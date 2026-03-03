from app.services.embedder import EmbedderService


class RetrieverService:
    def __init__(self, embedder: EmbedderService):
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> tuple[str, list[dict]]:
        """Search and return (formatted_context, sources)."""
        docs = self.embedder.search(query, k=k)

        sources = []
        context_parts = []

        for i, doc in enumerate(docs):
            meta = doc.metadata
            context_parts.append(
                f"[출처 {i + 1}] (문서: {meta.get('filename', '?')}, "
                f"페이지: {meta.get('page', '?')}, "
                f"유형: {meta.get('content_type', '?')})\n"
                f"{doc.page_content}"
            )
            sources.append({
                "document": meta.get("filename", "unknown"),
                "page": meta.get("page", 0),
                "content_type": meta.get("content_type", "text"),
                "snippet": doc.page_content[:200],
            })

        context = "\n\n---\n\n".join(context_parts)
        return context, sources
