from neo4j import AsyncGraphDatabase

from app.config import settings


class Neo4jClient:
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def verify_connectivity(self):
        await self.driver.verify_connectivity()

    async def close(self):
        await self.driver.close()

    async def init_schema(self):
        """Create constraints, vector index, and fulltext index."""
        async with self.driver.session() as session:
            # Uniqueness constraints
            await session.run(
                "CREATE CONSTRAINT doc_id IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.document_id IS UNIQUE"
            )
            await session.run(
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
            )
            await session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

            # Vector index on Chunk.embedding (cosine, 1536 dims)
            await session.run(
                "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.embedding) "
                "OPTIONS {indexConfig: {"
                " `vector.dimensions`: $dims,"
                " `vector.similarity_function`: 'cosine'"
                "}}",
                dims=settings.embedding_dimensions,
            )

            # Fulltext index on Entity.name for fuzzy matching
            await session.run(
                "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
                "FOR (e:Entity) ON EACH [e.name]"
            )

    async def run_query(self, query: str, **params):
        """Execute a single Cypher query and return list of records."""
        async with self.driver.session() as session:
            result = await session.run(query, **params)
            return [record.data() async for record in result]

    async def run_write(self, query: str, **params):
        """Execute a write transaction."""
        async with self.driver.session() as session:
            await session.run(query, **params)
