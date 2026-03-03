import json
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import settings


class EmbedderService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self.vectorstore = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=self.embeddings,
            persist_directory=str(settings.chroma_dir),
        )
        self.registry_path = settings.registry_path
        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {}

    def _save_registry(self):
        self.registry_path.write_text(
            json.dumps(self._registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_document(self, chunks: list[dict], document_id: str, filename: str, page_count: int):
        """Add document chunks to vectorstore and registry."""
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]

        self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        self._registry[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "uploaded_at": datetime.now().isoformat(),
        }
        self._save_registry()

    def delete_document(self, document_id: str):
        """Remove all chunks for a document from vectorstore and registry."""
        if document_id not in self._registry:
            raise ValueError(f"Document {document_id} not found")

        chunk_count = self._registry[document_id]["chunk_count"]
        ids = [f"{document_id}_{i}" for i in range(chunk_count)]
        self.vectorstore.delete(ids=ids)

        del self._registry[document_id]
        self._save_registry()

    def list_documents(self) -> list[dict]:
        return list(self._registry.values())

    def get_document(self, document_id: str) -> dict | None:
        return self._registry.get(document_id)

    def search(self, query: str, k: int = 5) -> list:
        """MMR search to avoid duplicate chunks."""
        return self.vectorstore.max_marginal_relevance_search(query, k=k)

    @property
    def document_count(self) -> int:
        return len(self._registry)
