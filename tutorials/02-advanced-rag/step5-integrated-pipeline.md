# Step 5: Advanced RAG 통합 파이프라인

## 전체 흐름

```
질문 → Multi-Query (질문 확장)
     → Hybrid Search (BM25 + 벡터)
     → Re-ranking (LLM 재정렬)
     → LLM 답변 생성
```

## 실습 코드

### 셀 0: 초기화

```python
import chromadb
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
client = OpenAI()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="news")

all_data = collection.get(include=["documents", "metadatas"])
all_docs = all_data['documents']
all_metas = all_data['metadatas']

tokenized_docs = [doc.split() for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)

print(f"준비 완료! 청크 수: {len(all_docs)}")
```

### 셀 1: 각 기법을 함수로 정리

```python
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

def hybrid_search(question, top_k=10, alpha=0.5):
    tokenized_query = question.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    if max(bm25_scores) > 0:
        bm25_scores = bm25_scores / max(bm25_scores)

    q_resp = client.embeddings.create(input=[question], model="text-embedding-3-small")
    vec_results = collection.query(
        query_embeddings=[q_resp.data[0].embedding],
        n_results=len(all_docs),
        include=["distances"]
    )

    vec_scores = np.zeros(len(all_docs))
    for i, doc_id in enumerate(vec_results['ids'][0]):
        idx = all_data['ids'].index(doc_id)
        vec_scores[idx] = 1 - vec_results['distances'][0][i]
    if max(vec_scores) > 0:
        vec_scores = vec_scores / max(vec_scores)

    combined = alpha * vec_scores + (1 - alpha) * bm25_scores
    top_indices = np.argsort(combined)[::-1][:top_k]

    return [{"doc": all_docs[i], "meta": all_metas[i], "score": combined[i]} for i in top_indices]

def rerank(question, documents, top_k=3):
    doc_list = ""
    for i, doc in enumerate(documents):
        doc_list += f"[문서 {i+1}] {doc}\n\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"질문에 가장 관련성 높은 문서 {top_k}개를 골라 번호만 출력하세요. 관련성 높은 순서대로, 쉼표로 구분. 예: 3,7,1"},
            {"role": "user", "content": f"질문: {question}\n\n{doc_list}"}
        ]
    )
    answer = response.choices[0].message.content.strip()
    return [int(x.strip()) - 1 for x in answer.split(",") if x.strip().isdigit()]
```

### 셀 2: 통합 파이프라인

```python
def advanced_rag(question):
    print(f"질문: {question}\n")

    queries = [question] + generate_multi_queries(question)
    print(f"[1] Multi-Query: {len(queries)}개 질문 생성")

    all_results = []
    seen_docs = set()
    for q in queries:
        results = hybrid_search(q, top_k=5)
        for r in results:
            doc_key = r['doc'][:50]
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                all_results.append(r)
    print(f"[2] Hybrid Search: {len(all_results)}개 후보 확보")

    candidate_docs = [r['doc'] for r in all_results[:15]]
    top_indices = rerank(question, candidate_docs, top_k=3)
    final_docs = [candidate_docs[i] for i in top_indices]
    print(f"[3] Re-ranking: Top {len(final_docs)}개 선별\n")

    context = "\n\n".join([f"[참고자료 {i+1}]\n{doc}" for i, doc in enumerate(final_docs)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "참고자료를 기반으로 답변하세요. 없는 내용은 '해당 정보를 찾을 수 없습니다'라고 하세요."},
            {"role": "user", "content": f"{context}\n\n질문: {question}"}
        ]
    )

    return response.choices[0].message.content, final_docs
```

### 셀 3: 테스트

```python
answer, docs = advanced_rag("이란 전쟁의 경제적 영향은?")
print("=== 답변 ===")
print(answer)
```

### 셀 4: Naive RAG vs Advanced RAG 비교

```python
def naive_rag(question):
    q_resp = client.embeddings.create(input=[question], model="text-embedding-3-small")
    results = collection.query(query_embeddings=[q_resp.data[0].embedding], n_results=3)
    context = "\n\n".join([f"[참고자료 {i+1}]\n{doc}" for i, doc in enumerate(results['documents'][0])])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "참고자료를 기반으로 답변하세요. 없는 내용은 '해당 정보를 찾을 수 없습니다'라고 하세요."},
            {"role": "user", "content": f"{context}\n\n질문: {question}"}
        ]
    )
    return response.choices[0].message.content

questions = [
    "이란 전쟁의 경제적 영향은?",
    "천궁 미사일 요격 성공률은?",
    "AI 관련 뉴스 알려줘",
]

for q in questions:
    print(f"Q: {q}")
    print(f"\n[Naive RAG]")
    print(naive_rag(q)[:150])
    print(f"\n[Advanced RAG]")
    answer, _ = advanced_rag(q)
    print(answer[:150])
    print("=" * 60)
```

## 관찰할 것

1. 같은 질문인데 답변 품질이 다른가?
2. Advanced RAG가 특히 잘하는 질문 유형은?
3. API 호출이 더 많아 느리고 비용 큼 → 트레이드오프

## 2단계 Advanced RAG 완료!
