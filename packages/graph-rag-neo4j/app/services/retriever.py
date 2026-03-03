from app.config import settings
from app.services.embedder import EmbedderService
from app.services.graph_builder import GraphBuilder
from app.services.neo4j_client import Neo4jClient


class RetrieverService:
    def __init__(
        self,
        neo4j: Neo4jClient,
        embedder: EmbedderService,
        graph_builder: GraphBuilder,
    ):
        self.neo4j = neo4j
        self.embedder = embedder
        self.graph_builder = graph_builder

    async def retrieve(
        self, query: str, k: int = 5
    ) -> tuple[str, list[dict], list[dict]]:
        """Hybrid retrieval: vector search + graph traversal.

        Returns (formatted_context, sources, graph_relations).
        """
        # 1. Vector search
        vector_chunks = await self._vector_search(query, k=k)

        # 2. Extract entity names from query via LLM
        entity_names = await self.graph_builder.extract_entity_names(query)

        # 3. Fulltext entity matching + graph traversal
        graph_chunks, graph_relations = await self._graph_search(
            entity_names, depth=settings.graph_traversal_depth
        )

        # 4. Merge and deduplicate
        seen_ids = set()
        merged_chunks = []

        for chunk in vector_chunks:
            cid = chunk["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                chunk["retrieval_method"] = "vector"
                merged_chunks.append(chunk)

        for chunk in graph_chunks:
            cid = chunk["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                chunk["retrieval_method"] = "graph"
                merged_chunks.append(chunk)
            elif cid in seen_ids:
                # Mark as both if found by both methods
                for mc in merged_chunks:
                    if mc["chunk_id"] == cid:
                        mc["retrieval_method"] = "hybrid"
                        break

        # 5. Build context
        sources = []
        context_parts = []

        for i, chunk in enumerate(merged_chunks):
            method = chunk.get("retrieval_method", "vector")
            context_parts.append(
                f"[출처 {i + 1}] (문서: {chunk.get('filename', '?')}, "
                f"페이지: {chunk.get('page', '?')}, "
                f"유형: {chunk.get('content_type', '?')}, "
                f"검색: {method})\n"
                f"{chunk['text']}"
            )
            sources.append(
                {
                    "document": chunk.get("filename", "unknown"),
                    "page": chunk.get("page", 0),
                    "content_type": chunk.get("content_type", "text"),
                    "snippet": chunk["text"][:200],
                    "retrieval_method": method,
                }
            )

        # 6. Append graph relations to context
        if graph_relations:
            rel_lines = []
            for rel in graph_relations:
                rel_lines.append(
                    f"- {rel['source']} --[{rel['type']}]--> {rel['target']}"
                )
            context_parts.append(
                "[지식 그래프 관계]\n" + "\n".join(rel_lines)
            )

        context = "\n\n---\n\n".join(context_parts)
        return context, sources, graph_relations

    async def _vector_search(self, query: str, k: int = 5) -> list[dict]:
        """Neo4j vector index search."""
        query_embedding = await self.embedder.embed_query(query)
        results = await self.neo4j.run_query(
            "CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding) "
            "YIELD node, score "
            "MATCH (d:Document)-[:HAS_CHUNK]->(node) "
            "RETURN node.chunk_id AS chunk_id, node.text AS text, "
            "       node.page AS page, node.content_type AS content_type, "
            "       d.filename AS filename, score "
            "ORDER BY score DESC",
            k=k,
            embedding=query_embedding,
        )
        return results

    async def _graph_search(
        self, entity_names: list[str], depth: int = 2
    ) -> tuple[list[dict], list[dict]]:
        """Fulltext entity search + N-hop graph traversal."""
        if not entity_names:
            return [], []

        # Fulltext search for entities
        matched_entities = []
        for name in entity_names:
            results = await self.neo4j.run_query(
                "CALL db.index.fulltext.queryNodes('entity_fulltext', $name) "
                "YIELD node, score "
                "WHERE score > 0.5 "
                "RETURN node.name AS name LIMIT 3",
                name=name,
            )
            matched_entities.extend([r["name"] for r in results])

        if not matched_entities:
            return [], []

        # Graph traversal: find chunks and relations within N hops
        chunks = await self.neo4j.run_query(
            "UNWIND $entities AS ename "
            "MATCH (e:Entity {name: ename})<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document) "
            "RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text, "
            "       c.page AS page, c.content_type AS content_type, "
            "       d.filename AS filename "
            "LIMIT 10",
            entities=matched_entities,
        )

        # Collect relationships between matched entities (up to depth hops)
        # Note: Neo4j doesn't support parameterized variable-length bounds,
        # so we interpolate depth into the query string (it's an int from config).
        relations = await self.neo4j.run_query(
            "UNWIND $entities AS ename "
            f"MATCH (e:Entity {{name: ename}})-[r:RELATES_TO*1..{int(depth)}]-(other:Entity) "
            "UNWIND r AS rel "
            "WITH startNode(rel) AS s, endNode(rel) AS t, rel "
            "RETURN DISTINCT s.name AS source, t.name AS target, rel.type AS type "
            "LIMIT 20",
            entities=matched_entities,
        )

        return chunks, relations
