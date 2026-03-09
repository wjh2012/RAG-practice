# Step 1: GraphRAG 개념 & Local Search 구현

## 일반 RAG vs GraphRAG

```
일반 RAG:  질문 → 벡터 검색 → "비슷한 텍스트" 찾기 → LLM 답변
GraphRAG: 질문 → 엔티티 추출 → 그래프 관계 탐색 → 컨텍스트 수집 → LLM 답변
```

## 핵심 차이

```
질문: "이란 전쟁이 한국 경제에 미치는 영향은?"

벡터 검색: "이란 전쟁" + "한국 경제" 함께 나오는 청크만 찾음

GraphRAG:
  이란 --[전쟁]--> 미국
  이란 --[사태로]--> 국제유가 상승
  국제유가 --[영향]--> 원달러 환율
  → 관계를 따라가며 연결고리 추론
```

## Microsoft GraphRAG 두 가지 검색

| | Local Search | Global Search |
|---|---|---|
| 질문 | "트럼프가 이란에 대해 뭐라 했어?" | "중동 사태의 전체 흐름은?" |
| 방식 | 특정 엔티티 중심 주변 탐색 | 커뮤니티 요약 기반 |

## 실습 코드

### 셀 0: 초기화

```python
import json
import numpy as np
import chromadb
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="news")
```

### 셀 1: Local Search

```python
def extract_entities(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "질문에서 핵심 엔티티를 추출하세요. 쉼표로 구분."},
            {"role": "user", "content": question}
        ]
    )
    return [e.strip() for e in response.choices[0].message.content.split(",")]

def graph_local_search(question, hops=2):
    entities = extract_entities(question)
    print(f"추출된 엔티티: {entities}")
    graph_context = []

    with driver.session() as session:
        for entity in entities:
            result = session.run("""
                MATCH (n {name: $name})-[r]-(m)
                RETURN n.name AS source, type(r) AS relation, m.name AS target
            """, name=entity)
            for record in result:
                graph_context.append(f"{record['source']} -{record['relation']}-> {record['target']}")

            if hops >= 2:
                result = session.run("""
                    MATCH (n {name: $name})-[r1]-(mid)-[r2]-(end)
                    WHERE end.name <> n.name
                    RETURN n.name AS source, type(r1) AS r1, mid.name AS mid, type(r2) AS r2, end.name AS target
                    LIMIT 10
                """, name=entity)
                for record in result:
                    graph_context.append(f"{record['source']} -{record['r1']}-> {record['mid']} -{record['r2']}-> {record['target']}")

    return list(set(graph_context))
```

### 셀 2: 벡터 + 그래프 결합 (GraphRAG)

```python
def graph_rag(question):
    graph_context = graph_local_search(question)
    graph_text = "\n".join(graph_context)

    q_resp = client.embeddings.create(input=[question], model="text-embedding-3-small")
    vec_results = collection.query(query_embeddings=[q_resp.data[0].embedding], n_results=3)
    vector_text = "\n\n".join(vec_results['documents'][0])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """두 가지 정보를 활용해 답변하세요:
1. [그래프 관계]: 엔티티 간 관계
2. [관련 문서]: 뉴스 원문
종합해서 답변. 없는 내용은 지어내지 마세요."""},
            {"role": "user", "content": f"[그래프 관계]\n{graph_text}\n\n[관련 문서]\n{vector_text}\n\n질문: {question}"}
        ]
    )
    return response.choices[0].message.content
```

### 셀 3: Naive RAG vs GraphRAG 비교

```python
questions = [
    "이란 전쟁이 한국 경제에 미치는 영향은?",
    "트럼프와 이란의 관계를 설명해줘",
    "중동 사태와 관련된 주요 인물들은?",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"\n[Naive RAG]")
    print(naive_rag(q)[:200])
    print(f"\n[GraphRAG]")
    print(graph_rag(q)[:200])
    print("=" * 60)
```

## 관찰할 것

1. 직접 언급 안 된 질문에서 GraphRAG가 관계를 따라 답을 만드나?
2. 그래프 허브 엔티티 정보가 답변에 반영되나?
3. GraphRAG가 더 나은 질문 유형은?
