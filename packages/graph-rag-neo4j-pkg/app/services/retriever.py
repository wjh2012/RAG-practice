"""Retriever service using neo4j-graphrag VectorCypherRetriever.

Performs vector similarity search on Chunk embeddings, then traverses
the graph to collect related entities and their relationships.
"""

import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from app.config import settings

# Cypher query executed after vector search finds matching Chunk nodes.
# `node` = matched Chunk, `score` = similarity score.
RETRIEVAL_QUERY = """
MATCH (node)-[:FROM_DOCUMENT]->(d:Document)
WITH node, score, d
OPTIONAL MATCH (e:__Entity__)-[:FROM_CHUNK]->(node)
WITH node, score, d, collect(DISTINCT e.name) AS entities
OPTIONAL MATCH (e2:__Entity__)-[:FROM_CHUNK]->(node)
OPTIONAL MATCH (e2)-[r]-(other:__Entity__)
    WHERE type(r) <> 'FROM_CHUNK' AND other <> e2
WITH node, score, d, entities,
     collect(DISTINCT {
         source: startNode(r).name,
         type: type(r),
         target: endNode(r).name
     }) AS relations
RETURN node.text AS text, score,
       coalesce(d.filename, d.path) AS document,
       entities, relations
"""


def _format_result(record) -> RetrieverResultItem:
    """Custom formatter to preserve structured metadata."""
    return RetrieverResultItem(
        content=record.get("text", ""),
        metadata={
            "score": record.get("score", 0),
            "document": record.get("document", "unknown"),
            "entities": record.get("entities", []),
            "relations": record.get("relations", []),
        },
    )


class RetrieverService:
    def __init__(self, driver: GraphDatabase.driver, embedder: OpenAIEmbeddings):
        self.retriever = VectorCypherRetriever(
            driver=driver,
            index_name="chunk_vector",
            retrieval_query=RETRIEVAL_QUERY,
            embedder=embedder,
            result_formatter=_format_result,
        )

    async def retrieve(
        self, query: str, k: int = 5
    ) -> tuple[str, list[dict], list[dict]]:
        """Search and return (formatted_context, sources, graph_relations)."""
        # VectorCypherRetriever.search() is synchronous — run in thread
        result = await asyncio.to_thread(
            self.retriever.search, query_text=query, top_k=k
        )

        sources = []
        context_parts = []
        all_relations = []
        seen_relations = set()

        for i, item in enumerate(result.items):
            meta = item.metadata

            context_parts.append(
                f"[출처 {i + 1}] (문서: {meta.get('document', '?')})\n"
                f"{item.content}"
            )

            entities = meta.get("entities", [])
            sources.append(
                {
                    "document": meta.get("document", "unknown"),
                    "page": 0,
                    "content_type": "text",
                    "snippet": item.content[:200],
                    "retrieval_method": "graph" if entities else "vector",
                    "entities": entities,
                }
            )

            # Collect unique relations
            for rel in meta.get("relations", []):
                key = (rel.get("source", ""), rel.get("type", ""), rel.get("target", ""))
                if key[0] and key[2] and key not in seen_relations:
                    seen_relations.add(key)
                    all_relations.append(rel)

        # Append graph relations to context for LLM
        if all_relations:
            rel_lines = [
                f"- {r['source']} --[{r['type']}]--> {r['target']}"
                for r in all_relations
            ]
            context_parts.append("[지식 그래프 관계]\n" + "\n".join(rel_lines))

        context = "\n\n---\n\n".join(context_parts)
        return context, sources, all_relations
