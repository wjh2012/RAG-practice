"""Knowledge Graph service using neo4j-graphrag SimpleKGPipeline.

Uses the official neo4j-graphrag package for:
- PDF text extraction (built-in PdfLoader)
- Text chunking (FixedSizeSplitter)
- LLM entity/relation extraction
- Neo4j graph writing
- Entity resolution (deduplication)
"""

import logging
import uuid

import fitz  # PyMuPDF — only for page counting
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.llm import OpenAILLM

from app.config import settings

logger = logging.getLogger(__name__)


class KGService:
    def __init__(self, driver: GraphDatabase.driver):
        self.driver = driver
        self.llm = OpenAILLM(
            model_name=settings.llm_model,
            model_params={"temperature": 0},
            api_key=settings.openai_api_key,
        )
        self.embedder = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        self.pipeline = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            embedder=self.embedder,
            from_pdf=True,
            entities=settings.entity_types,
            relations=settings.relation_types,
            perform_entity_resolution=True,
        )

    def init_indexes(self):
        """Create vector and fulltext indexes (idempotent)."""
        try:
            create_vector_index(
                self.driver,
                "chunk_vector",
                label="Chunk",
                embedding_property="embedding",
                dimensions=settings.embedding_dimensions,
                similarity_fn="cosine",
            )
            logger.info("Vector index 'chunk_vector' created")
        except Exception as e:
            logger.debug("Vector index creation skipped: %s", e)

        try:
            create_fulltext_index(
                self.driver,
                "entity_fulltext",
                label="__Entity__",
                node_properties=["name"],
            )
            logger.info("Fulltext index 'entity_fulltext' created")
        except Exception as e:
            logger.debug("Fulltext index creation skipped: %s", e)

    async def build_from_pdf(
        self, file_path, document_id: str, filename: str
    ) -> tuple[int, int, int]:
        """Build knowledge graph from PDF. Returns (page_count, chunk_count, entity_count)."""
        # Get page count via PyMuPDF (fast, no full extraction)
        doc = fitz.open(str(file_path))
        page_count = len(doc)
        doc.close()

        # Run neo4j-graphrag pipeline (PDF load → chunk → extract → write → resolve)
        await self.pipeline.run_async(file_path=str(file_path))

        # Add our metadata to the Document node created by the pipeline
        with self.driver.session() as session:
            result = session.run(
                "MATCH (d:Document {path: $path}) "
                "SET d.document_id = $doc_id, d.filename = $filename, "
                "    d.page_count = $page_count, d.uploaded_at = datetime() "
                "WITH d "
                "OPTIONAL MATCH (d)<-[:FROM_DOCUMENT]-(c:Chunk) "
                "WITH d, count(c) AS chunks "
                "SET d.chunk_count = chunks "
                "RETURN chunks",
                path=str(file_path),
                doc_id=document_id,
                filename=filename,
                page_count=page_count,
            )
            record = result.single()
            chunk_count = record["chunks"] if record else 0

        # Count entities linked to this document's chunks
        with self.driver.session() as session:
            result = session.run(
                "MATCH (d:Document {document_id: $doc_id})"
                "<-[:FROM_DOCUMENT]-(c:Chunk)"
                "<-[:FROM_CHUNK]-(e:__Entity__) "
                "RETURN count(DISTINCT e) AS cnt",
                doc_id=document_id,
            )
            record = result.single()
            entity_count = record["cnt"] if record else 0

        return page_count, chunk_count, entity_count

    def list_documents(self) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (d:Document) WHERE d.document_id IS NOT NULL "
                "RETURN d.document_id AS document_id, d.filename AS filename, "
                "       d.page_count AS page_count, d.chunk_count AS chunk_count, "
                "       toString(d.uploaded_at) AS uploaded_at "
                "ORDER BY d.uploaded_at DESC"
            )
            return [record.data() for record in result]

    def delete_document(self, document_id: str):
        with self.driver.session() as session:
            # Delete entities extracted from this document's chunks
            session.run(
                "MATCH (d:Document {document_id: $doc_id})"
                "<-[:FROM_DOCUMENT]-(c:Chunk)"
                "<-[:FROM_CHUNK]-(e:__Entity__) "
                "DETACH DELETE e",
                doc_id=document_id,
            )
            # Delete chunks
            session.run(
                "MATCH (d:Document {document_id: $doc_id})"
                "<-[:FROM_DOCUMENT]-(c:Chunk) "
                "DETACH DELETE c",
                doc_id=document_id,
            )
            # Delete document
            session.run(
                "MATCH (d:Document {document_id: $doc_id}) DETACH DELETE d",
                doc_id=document_id,
            )
            # Clean orphan entities (not linked to any chunk)
            session.run(
                "MATCH (e:__Entity__) "
                "WHERE NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->(:Chunk) } "
                "DETACH DELETE e"
            )

    def document_count(self) -> int:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (d:Document) WHERE d.document_id IS NOT NULL "
                "RETURN count(d) AS cnt"
            )
            return result.single()["cnt"]

    def graph_stats(self) -> dict:
        with self.driver.session() as session:
            result = session.run(
                "OPTIONAL MATCH (d:Document) WHERE d.document_id IS NOT NULL "
                "WITH count(d) AS docs "
                "OPTIONAL MATCH (c:Chunk) WITH docs, count(c) AS chunks "
                "OPTIONAL MATCH (e:__Entity__) WITH docs, chunks, count(e) AS entities "
                "OPTIONAL MATCH (:__Entity__)-[r]->(:__Entity__) "
                "WHERE type(r) <> 'FROM_CHUNK' "
                "RETURN docs, chunks, entities, count(r) AS rels"
            )
            record = result.single()
            return {
                "document_count": record["docs"],
                "chunk_count": record["chunks"],
                "entity_count": record["entities"],
                "relationship_count": record["rels"],
            }
