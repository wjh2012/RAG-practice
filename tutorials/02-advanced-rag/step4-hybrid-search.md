# Step 4: Hybrid Search (BM25 + 벡터 검색)

## 문제

| | 벡터 검색 | BM25 (키워드) |
|---|---|---|
| 장점 | 의미적 유사성 | 정확한 단어 매칭 |
| 단점 | 특정 키워드 놓침 | 동의어/유사 표현 못 찾음 |

→ 합치면 더 좋다!

## BM25란?

검색 엔진의 고전 알고리즘:
1. 질문 단어가 문서에 몇 번 나오나? (많을수록 관련)
2. 그 단어가 전체에서 흔한가? (흔하면 덜 중요)

## 실습 코드

### 셀 0: 설치 & 벡터DB 불러오기

```python
# pip install rank-bm25

import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import numpy as np

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="news")
client = OpenAI()

all_data = collection.get(include=["documents", "metadatas"])
all_docs = all_data['documents']
all_metas = all_data['metadatas']
print(f"총 문서 수: {len(all_docs)}")
```

### 셀 1: BM25 인덱스 만들기

```python
tokenized_docs = [doc.split() for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)

question = "천궁 요격 성공률"
tokenized_query = question.split()
bm25_scores = bm25.get_scores(tokenized_query)

top_indices = np.argsort(bm25_scores)[::-1][:5]

print("=== BM25 검색 결과 ===")
for rank, idx in enumerate(top_indices):
    print(f"{rank+1}. (점수: {bm25_scores[idx]:.2f}) {all_docs[idx][:80]}")
```

### 셀 2: 벡터 검색과 비교

```python
q_resp = client.embeddings.create(input=[question], model="text-embedding-3-small")
vec_results = collection.query(query_embeddings=[q_resp.data[0].embedding], n_results=5)

print("=== 벡터 검색 결과 ===")
for i, doc in enumerate(vec_results['documents'][0]):
    print(f"{i+1}. {doc[:80]}")
```

### 셀 3: Hybrid Search

```python
def hybrid_search(question, collection, all_docs, all_metas, bm25, top_k=5, alpha=0.5):
    """
    alpha: 벡터 검색 가중치 (0~1)
    1-alpha: BM25 가중치
    """
    tokenized_query = question.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    if max(bm25_scores) > 0:
        bm25_scores = bm25_scores / max(bm25_scores)

    q_resp = client.embeddings.create(input=[question], model="text-embedding-3-small")
    vec_results = collection.query(
        query_embeddings=[q_resp.data[0].embedding],
        n_results=len(all_docs),
        include=["distances", "documents"]
    )

    vec_scores = np.zeros(len(all_docs))
    for i, doc_id in enumerate(vec_results['ids'][0]):
        idx = all_data['ids'].index(doc_id)
        vec_scores[idx] = 1 - vec_results['distances'][0][i]
    if max(vec_scores) > 0:
        vec_scores = vec_scores / max(vec_scores)

    combined_scores = alpha * vec_scores + (1 - alpha) * bm25_scores
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "doc": all_docs[idx],
            "meta": all_metas[idx],
            "score": combined_scores[idx],
            "vec_score": vec_scores[idx],
            "bm25_score": bm25_scores[idx]
        })
    return results

results = hybrid_search(question, collection, all_docs, all_metas, bm25)

print(f"=== Hybrid Search: '{question}' ===\n")
for i, r in enumerate(results):
    print(f"{i+1}. [합산: {r['score']:.3f}] (벡터: {r['vec_score']:.3f}, BM25: {r['bm25_score']:.3f})")
    print(f"   [{r['meta']['category']}] {r['doc'][:70]}")
```

### 셀 4: alpha 값에 따른 차이

```python
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    results = hybrid_search(question, collection, all_docs, all_metas, bm25, alpha=alpha)
    print(f"--- alpha={alpha} (벡터 {int(alpha*100)}% + BM25 {int((1-alpha)*100)}%) ---")
    for r in results[:3]:
        print(f"  {r['doc'][:60]}")
    print()
```

## 관찰할 것

1. BM25와 벡터 검색이 **다른 결과**를 가져오나?
2. Hybrid가 둘의 장점을 합쳐서 더 좋은 결과를 내나?
3. **alpha 값**에 따라 결과가 어떻게 변하나?
