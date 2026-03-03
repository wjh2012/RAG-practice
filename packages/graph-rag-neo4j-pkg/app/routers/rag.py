import asyncio
import json
import shutil
import uuid
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
from app.services.kg_service import KGService
from app.services.llm_chain import LLMChain
from app.services.retriever import RetrieverService

router = APIRouter(prefix="/api")


def get_kg(request: Request) -> KGService:
    return request.app.state.kg_service


def get_retriever(request: Request) -> RetrieverService:
    return request.app.state.retriever


def get_llm(request: Request) -> LLMChain:
    return request.app.state.llm_chain


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    kg: KGService = Depends(get_kg),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    document_id = uuid.uuid4().hex[:12]
    file_path = settings.upload_dir / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        page_count, chunk_count, entity_count = await kg.build_from_pdf(
            file_path, document_id, file.filename
        )
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"PDF 처리 중 오류: {e}")

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        page_count=page_count,
        chunk_count=chunk_count,
        entity_count=entity_count,
        message="문서가 성공적으로 업로드되었습니다.",
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(kg: KGService = Depends(get_kg)):
    docs = await asyncio.to_thread(kg.list_documents)
    return DocumentListResponse(documents=docs)


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    kg: KGService = Depends(get_kg),
):
    await asyncio.to_thread(kg.delete_document, document_id)
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
async def graph_stats(kg: KGService = Depends(get_kg)):
    stats = await asyncio.to_thread(kg.graph_stats)
    return GraphStatsResponse(**stats)


@router.get("/health", response_model=HealthResponse)
async def health(kg: KGService = Depends(get_kg)):
    count = await asyncio.to_thread(kg.document_count)
    return HealthResponse(
        status="healthy",
        document_count=count,
        timestamp=datetime.now().isoformat(),
    )
