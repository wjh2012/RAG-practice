import json
import shutil
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.schemas import (
    DeleteResponse,
    DocumentListResponse,
    GraphStatsResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)
from app.services.embedder import EmbedderService
from app.services.graph_builder import GraphBuilder
from app.services.llm_chain import LLMChain
from app.services.neo4j_client import Neo4jClient
from app.services.pdf_processor import PDFProcessor
from app.services.retriever import RetrieverService

router = APIRouter(prefix="/api")


def get_neo4j(request: Request) -> Neo4jClient:
    return request.app.state.neo4j


def get_pdf_processor(request: Request) -> PDFProcessor:
    return request.app.state.pdf_processor


def get_embedder(request: Request) -> EmbedderService:
    return request.app.state.embedder


def get_graph_builder(request: Request) -> GraphBuilder:
    return request.app.state.graph_builder


def get_retriever(request: Request) -> RetrieverService:
    return request.app.state.retriever


def get_llm(request: Request) -> LLMChain:
    return request.app.state.llm_chain


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    processor: PDFProcessor = Depends(get_pdf_processor),
    embedder: EmbedderService = Depends(get_embedder),
    graph_builder: GraphBuilder = Depends(get_graph_builder),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = settings.upload_dir / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks, page_count, document_id = processor.process(file_path)
        await embedder.add_document(chunks, document_id, file.filename, page_count)
        await graph_builder.extract_and_store(chunks, document_id)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"PDF 처리 중 오류: {e}")

    # Count entities linked to this document
    neo4j: Neo4jClient = embedder.neo4j
    entity_result = await neo4j.run_query(
        "MATCH (d:Document {document_id: $doc_id})-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity) "
        "RETURN count(DISTINCT e) AS cnt",
        doc_id=document_id,
    )
    entity_count = entity_result[0]["cnt"] if entity_result else 0

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        page_count=page_count,
        chunk_count=len(chunks),
        entity_count=entity_count,
        message="문서가 성공적으로 업로드되었습니다.",
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(embedder: EmbedderService = Depends(get_embedder)):
    docs = await embedder.list_documents()
    return DocumentListResponse(documents=docs)


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    embedder: EmbedderService = Depends(get_embedder),
):
    try:
        await embedder.delete_document(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return DeleteResponse(message="문서가 삭제되었습니다.")


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    retriever: RetrieverService = Depends(get_retriever),
    llm: LLMChain = Depends(get_llm),
):
    context, sources, _ = await retriever.retrieve(req.query, k=req.k)
    if not context:
        return SearchResponse(answer="관련 문서를 찾을 수 없습니다.", sources=[])
    answer = await llm.generate(req.query, context)
    return SearchResponse(answer=answer, sources=sources)


@router.post("/search/stream")
async def search_stream(
    req: SearchRequest,
    retriever: RetrieverService = Depends(get_retriever),
    llm: LLMChain = Depends(get_llm),
):
    context, sources, graph_relations = await retriever.retrieve(req.query, k=req.k)

    async def event_generator():
        if not context:
            yield f"data: {json.dumps({'type': 'error', 'content': '관련 문서를 찾을 수 없습니다.'}, ensure_ascii=False)}\n\n"
            return

        yield f"data: {json.dumps({'type': 'sources', 'content': sources}, ensure_ascii=False)}\n\n"

        if graph_relations:
            yield f"data: {json.dumps({'type': 'graph', 'content': graph_relations}, ensure_ascii=False)}\n\n"

        async for chunk in llm.stream(req.query, context):
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/graph/stats", response_model=GraphStatsResponse)
async def graph_stats(neo4j: Neo4jClient = Depends(get_neo4j)):
    result = await neo4j.run_query(
        "MATCH (d:Document) WITH count(d) AS docs "
        "MATCH (c:Chunk) WITH docs, count(c) AS chunks "
        "MATCH (e:Entity) WITH docs, chunks, count(e) AS entities "
        "OPTIONAL MATCH ()-[r:RELATES_TO]->() "
        "RETURN docs, chunks, entities, count(r) AS rels"
    )
    if result:
        r = result[0]
        return GraphStatsResponse(
            document_count=r["docs"],
            chunk_count=r["chunks"],
            entity_count=r["entities"],
            relationship_count=r["rels"],
        )
    return GraphStatsResponse(
        document_count=0, chunk_count=0, entity_count=0, relationship_count=0
    )


@router.get("/health", response_model=HealthResponse)
async def health(embedder: EmbedderService = Depends(get_embedder)):
    return HealthResponse(
        status="healthy",
        document_count=await embedder.document_count(),
        timestamp=datetime.now().isoformat(),
    )
