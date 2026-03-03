from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    upload_dir: Path = base_dir / "uploads"
    chroma_dir: Path = base_dir / "chroma_db"
    registry_path: Path = base_dir / "documents_registry.json"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embedding
    embedding_model: str = "text-embedding-3-small"

    # LLM
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1

    # Retrieval
    search_k: int = 5

    # Chroma
    chroma_collection: str = "pdf_documents"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
