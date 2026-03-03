from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    upload_dir: Path = base_dir / "uploads"

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # LLM
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1

    # Retrieval
    search_k: int = 5
    graph_traversal_depth: int = 2

    # KG Builder entity/relation types
    entity_types: list[str] = [
        "Person",
        "Organization",
        "Location",
        "Concept",
        "Technology",
        "Event",
    ]
    relation_types: list[str] = [
        "RELATES_TO",
        "PART_OF",
        "WORKS_FOR",
        "LOCATED_IN",
        "CREATED_BY",
        "DEPENDS_ON",
    ]

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
