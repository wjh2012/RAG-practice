import json
import logging

from openai import AsyncOpenAI

from app.config import settings
from app.services.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
당신은 텍스트에서 엔티티(개체)와 관계를 추출하는 전문가입니다.
주어진 텍스트를 분석하여 핵심 엔티티와 엔티티 간의 관계를 JSON으로 추출하세요.

규칙:
1. 엔티티 유형: PERSON, ORGANIZATION, LOCATION, CONCEPT, TECHNOLOGY, EVENT, DOCUMENT, OTHER
2. 관계는 두 엔티티 사이의 의미적 연결을 나타냅니다
3. 한국어 고유명사는 원문 그대로 사용하세요
4. 최대 10개의 엔티티와 15개의 관계를 추출하세요
5. 반드시 아래 JSON 형식만 출력하세요

출력 형식:
{
  "entities": [
    {"name": "엔티티명", "type": "유형", "description": "간단한 설명"}
  ],
  "relationships": [
    {"source": "엔티티명1", "target": "엔티티명2", "type": "관계유형"}
  ]
}

텍스트:
"""


class GraphBuilder:
    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model

    async def extract_and_store(self, chunks: list[dict], document_id: str):
        """Extract entities/relationships from chunks and store in Neo4j."""
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_{i}"
            text = chunk["text"]

            # Skip very short chunks
            if len(text.strip()) < 50:
                continue

            try:
                extracted = await self._extract_entities(text)
            except Exception as e:
                logger.warning("Entity extraction failed for chunk %s: %s", chunk_id, e)
                continue

            entities = extracted.get("entities", [])
            relationships = extracted.get("relationships", [])

            # MERGE Entity nodes and create MENTIONS relationships
            for entity in entities:
                name = entity.get("name", "").strip()
                if not name:
                    continue
                await self.neo4j.run_write(
                    "MERGE (e:Entity {name: $name}) "
                    "ON CREATE SET e.type = $type, e.description = $desc "
                    "WITH e "
                    "MATCH (c:Chunk {chunk_id: $chunk_id}) "
                    "MERGE (c)-[:MENTIONS]->(e)",
                    name=name,
                    type=entity.get("type", "OTHER"),
                    desc=entity.get("description", ""),
                    chunk_id=chunk_id,
                )

            # MERGE RELATES_TO relationships between entities
            for rel in relationships:
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                rel_type = rel.get("type", "RELATES_TO")
                if not source or not target:
                    continue
                await self.neo4j.run_write(
                    "MATCH (a:Entity {name: $source}) "
                    "MATCH (b:Entity {name: $target}) "
                    "MERGE (a)-[r:RELATES_TO {type: $rel_type}]->(b)",
                    source=source,
                    target=target,
                    rel_type=rel_type,
                )

    async def extract_entity_names(self, query: str) -> list[str]:
        """Extract entity names from a user query for graph lookup."""
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "사용자 질문에서 핵심 엔티티(사람, 조직, 기술, 개념 등)의 이름을 추출하세요. "
                        'JSON 배열로만 응답하세요. 예: ["엔티티1", "엔티티2"]'
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        text = response.choices[0].message.content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return []

    async def _extract_entities(self, text: str) -> dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You extract entities and relationships from text as JSON."},
                {"role": "user", "content": EXTRACTION_PROMPT + text},
            ],
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
        return json.loads(content)
