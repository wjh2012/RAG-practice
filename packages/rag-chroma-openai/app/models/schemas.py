from datetime import datetime
from pydantic import BaseModel


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SourceInfo(BaseModel):
    document: str
    page: int
    content_type: str
    snippet: str


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    message: str


class DeleteResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    document_count: int
    timestamp: str
