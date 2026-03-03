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
    retrieval_method: str = "vector"


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    entity_count: int
    message: str


class DeleteResponse(BaseModel):
    message: str


class GraphStatsResponse(BaseModel):
    document_count: int
    chunk_count: int
    entity_count: int
    relationship_count: int


class HealthResponse(BaseModel):
    status: str
    document_count: int
    timestamp: str
