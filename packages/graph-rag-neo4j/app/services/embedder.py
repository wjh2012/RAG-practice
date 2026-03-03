from openai import AsyncOpenAI

from app.config import settings
from app.services.neo4j_client import Neo4jClient

BATCH_SIZE = 100


class EmbedderService:
    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts in batches."""
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = await self.client.embeddings.create(
                model=self.model, input=batch
            )
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        response = await self.client.embeddings.create(model=self.model, input=query)
        return response.data[0].embedding

    async def add_document(
        self,
        chunks: list[dict],
        document_id: str,
        filename: str,
        page_count: int,
    ):
        """Create Document node, Chunk nodes with embeddings, and HAS_CHUNK / NEXT_CHUNK relationships."""
        # Create Document node
        await self.neo4j.run_write(
            "MERGE (d:Document {document_id: $doc_id}) "
            "SET d.filename = $filename, d.page_count = $page_count, "
            "    d.chunk_count = $chunk_count, d.uploaded_at = datetime()",
            doc_id=document_id,
            filename=filename,
            page_count=page_count,
            chunk_count=len(chunks),
        )

        # Embed all chunk texts
        texts = [c["text"] for c in chunks]
        embeddings = await self.embed_texts(texts)

        # Batch-create Chunk nodes with UNWIND
        chunk_data = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_data.append(
                {
                    "chunk_id": f"{document_id}_{i}",
                    "text": chunk["text"],
                    "page": chunk["metadata"].get("page", 0),
                    "content_type": chunk["metadata"].get("content_type", "text"),
                    "embedding": emb,
                    "seq": i,
                }
            )

        await self.neo4j.run_write(
            "UNWIND $chunks AS c "
            "CREATE (ch:Chunk {chunk_id: c.chunk_id}) "
            "SET ch.text = c.text, ch.page = c.page, "
            "    ch.content_type = c.content_type, "
            "    ch.embedding = c.embedding, ch.seq = c.seq "
            "WITH ch, c "
            "MATCH (d:Document {document_id: $doc_id}) "
            "CREATE (d)-[:HAS_CHUNK]->(ch)",
            chunks=chunk_data,
            doc_id=document_id,
        )

        # Create NEXT_CHUNK relationships for sequential chunks
        await self.neo4j.run_write(
            "MATCH (d:Document {document_id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) "
            "WITH c ORDER BY c.seq "
            "WITH collect(c) AS chunks "
            "UNWIND range(0, size(chunks)-2) AS i "
            "WITH chunks[i] AS a, chunks[i+1] AS b "
            "CREATE (a)-[:NEXT_CHUNK]->(b)",
            doc_id=document_id,
        )

    async def delete_document(self, document_id: str):
        """Delete Document, its Chunks, and related Entity edges."""
        # Delete chunks and their relationships
        await self.neo4j.run_write(
            "MATCH (d:Document {document_id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) "
            "DETACH DELETE c",
            doc_id=document_id,
        )
        # Delete document node
        await self.neo4j.run_write(
            "MATCH (d:Document {document_id: $doc_id}) DETACH DELETE d",
            doc_id=document_id,
        )
        # Clean up orphan entities (no MENTIONS from any chunk)
        await self.neo4j.run_write(
            "MATCH (e:Entity) "
            "WHERE NOT EXISTS { MATCH (:Chunk)-[:MENTIONS]->(e) } "
            "DETACH DELETE e"
        )

    async def list_documents(self) -> list[dict]:
        return await self.neo4j.run_query(
            "MATCH (d:Document) "
            "RETURN d.document_id AS document_id, d.filename AS filename, "
            "       d.page_count AS page_count, d.chunk_count AS chunk_count, "
            "       toString(d.uploaded_at) AS uploaded_at "
            "ORDER BY d.uploaded_at DESC"
        )

    async def document_count(self) -> int:
        result = await self.neo4j.run_query("MATCH (d:Document) RETURN count(d) AS cnt")
        return result[0]["cnt"] if result else 0
