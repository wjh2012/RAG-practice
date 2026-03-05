# Step 2: Query Transformation (질문 변환)

## 문제

사용자 질문이 짧고 모호하면 벡터 검색이 엉뚱한 청크를 가져올 수 있다.

```
"중동 상황 어때?" → 너무 짧음 → 검색 품질 낮음
```

## 기법 1: Multi-Query

하나의 질문을 **여러 버전으로 바꿔서** 각각 검색, 결과를 합침.

```
원본: "중동 상황 어때?"
  → "중동 전쟁의 최신 동향은?"
  → "이란 분쟁 관련 뉴스는?"
  → "중동 지역 군사적 상황은?"
```

3번 검색 → 결과 합치기 → 더 다양한 청크 확보

## 기법 2: HyDE (Hypothetical Document Embedding)

질문 대신 **LLM이 만든 가상 답변**을 임베딩해서 검색.

```
질문: "중동 상황 어때?"
→ LLM 가상 답변: "현재 중동에서는 이란과 미국 간 군사적 긴장이 고조되고..."
→ 이 답변을 임베딩해서 검색
```

**질문은 짧지만, 답변은 문서와 비슷한 형태**이므로 유사도가 더 잘 잡힘.

## 실습 코드

### 셀 0: 벡터DB 불러오기 (1단계에서 저장한 것)

```python
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="news")
print(f"불러온 청크 수: {collection.count()}")
```

### 셀 1: Multi-Query 구현

```python
from openai import OpenAI

client = OpenAI()

def generate_multi_queries(question, n=3):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"주어진 질문을 {n}가지 다른 표현으로 바꿔주세요. 각 줄에 하나씩만 출력하세요."},
            {"role": "user", "content": question}
        ]
    )
    queries = response.choices[0].message.content.strip().split("\n")
    return [q.strip().lstrip("0123456789.-) ") for q in queries if q.strip()]

question = "중동 상황 어때?"
queries = generate_multi_queries(question)
print(f"원본: {question}")
for i, q in enumerate(queries):
    print(f"  변환 {i+1}: {q}")
```

### 셀 2: Multi-Query 검색

```python
def multi_query_search(question, collection, n_results=3):
    queries = generate_multi_queries(question)
    all_queries = [question] + queries

    all_docs = []
    seen_ids = set()

    for q in all_queries:
        q_resp = client.embeddings.create(input=[q], model="text-embedding-3-small")
        results = collection.query(query_embeddings=[q_resp.data[0].embedding], n_results=n_results)

        for doc_id, doc, meta in zip(results['ids'][0], results['documents'][0], results['metadatas'][0]):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append({"id": doc_id, "doc": doc, "meta": meta})

    return all_docs

results = multi_query_search("중동 상황 어때?", collection)
print(f"총 검색된 청크: {len(results)}개\n")
for r in results:
    print(f"[{r['meta']['category']}] {r['doc'][:80]}")
```

### 셀 3: HyDE 구현

```python
def hyde_search(question, collection, n_results=3):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "질문에 대해 뉴스 기사 스타일로 가상의 답변을 1문단으로 작성하세요."},
            {"role": "user", "content": question}
        ]
    )
    hypothetical_doc = response.choices[0].message.content
    print(f"가상 답변: {hypothetical_doc[:100]}...\n")

    h_resp = client.embeddings.create(input=[hypothetical_doc], model="text-embedding-3-small")
    results = collection.query(query_embeddings=[h_resp.data[0].embedding], n_results=n_results)

    return results

results = hyde_search("중동 상황 어때?", collection)
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(f"[{meta['category']}] {doc[:80]}")
```

### 셀 4: 비교 - 일반 vs Multi-Query vs HyDE

```python
question = "경제에 미치는 영향은?"

print("=== 일반 검색 ===")
q_resp = client.embeddings.create(input=[question], model="text-embedding-3-small")
r1 = collection.query(query_embeddings=[q_resp.data[0].embedding], n_results=3)
for doc in r1['documents'][0]:
    print(f"  {doc[:80]}")

print("\n=== Multi-Query ===")
r2 = multi_query_search(question, collection)
for r in r2[:3]:
    print(f"  {r['doc'][:80]}")

print("\n=== HyDE ===")
r3 = hyde_search(question, collection)
for doc in r3['documents'][0]:
    print(f"  {doc[:80]}")
```

## 관찰할 것

1. Multi-Query가 일반 검색보다 **더 다양한 청크**를 가져오나?
2. HyDE가 **간접적으로 관련된 청크**를 더 잘 찾나?
3. 어떤 질문에서 차이가 크고, 어떤 질문에서 차이가 작은가?
