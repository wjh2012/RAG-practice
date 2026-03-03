import json
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.schemas import (
    DeleteResponse,
    DocumentListResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)
from app.services.embedder import EmbedderService
from app.services.llm_chain import LLMChain
from app.services.pdf_processor import PDFProcessor
from app.services.retriever import RetrieverService

router = APIRouter(prefix="/api")


def get_pdf_processor(request: Request) -> PDFProcessor:
    return request.app.state.pdf_processor


def get_embedder(request: Request) -> EmbedderService:
    return request.app.state.embedder


def get_retriever(request: Request) -> RetrieverService:
    return request.app.state.retriever


def get_llm(request: Request) -> LLMChain:
    return request.app.state.llm_chain


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    processor: PDFProcessor = Depends(get_pdf_processor),
    embedder: EmbedderService = Depends(get_embedder),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = settings.upload_dir / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks, page_count, document_id = processor.process(file_path)
        embedder.add_document(chunks, document_id, file.filename, page_count)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"PDF 처리 중 오류: {str(e)}")

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        page_count=page_count,
        chunk_count=len(chunks),
        message="문서가 성공적으로 업로드되었습니다.",
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(embedder: EmbedderService = Depends(get_embedder)):
    return DocumentListResponse(documents=embedder.list_documents())


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    embedder: EmbedderService = Depends(get_embedder),
):
    try:
        embedder.delete_document(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return DeleteResponse(message="문서가 삭제되었습니다.")


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    retriever: RetrieverService = Depends(get_retriever),
    llm: LLMChain = Depends(get_llm),
):
    context, sources = retriever.retrieve(req.query, k=req.k)
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
    context, sources = retriever.retrieve(req.query, k=req.k)

    async def event_generator():
        if not context:
            yield f"data: {json.dumps({'type': 'error', 'content': '관련 문서를 찾을 수 없습니다.'}, ensure_ascii=False)}\n\n"
            return

        # Send sources first
        yield f"data: {json.dumps({'type': 'sources', 'content': sources}, ensure_ascii=False)}\n\n"

        # Stream answer chunks
        async for chunk in llm.stream(req.query, context):
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/health", response_model=HealthResponse)
async def health(embedder: EmbedderService = Depends(get_embedder)):
    return HealthResponse(
        status="healthy",
        document_count=embedder.document_count,
        timestamp=datetime.now().isoformat(),
    )
