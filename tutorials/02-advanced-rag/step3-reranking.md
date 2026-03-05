# Step 3: Re-ranking (재정렬)

## 문제

벡터 검색 Top 3의 순서가 질문 의도와 안 맞을 수 있다.

```
질문: "이란 전쟁의 경제적 영향은?"
벡터 검색:
  1위: 전쟁 뉴스 (경제 아님)
  2위: 정치 뉴스
  3위: LNG 생산 차질 ← 진짜 경제적 영향!
```

## 해결: Re-ranker

```
벡터 검색 (빠르지만 대략적) → Top 10 후보 확보
    ↓
Re-ranker (느리지만 정밀) → Top 3 재정렬
```

## 실습 코드

### 셀 0: 벡터DB 불러오기

```python
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="news")
client = OpenAI()
```

### 셀 1: 벡터 검색으로 후보 많이 가져오기

```python
question = "이란 전쟁의 경제적 영향은?"

q_resp = client.embeddings.create(input=[question], model="text-embedding-3-small")
results = collection.query(
    query_embeddings=[q_resp.data[0].embedding],
    n_results=10
)

print("=== 벡터 검색 Top 10 ===")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"{i+1}. [{meta['category']}] {doc[:60]}")
```

### 셀 2: LLM Re-ranking

```python
def rerank(question, documents, top_k=3):
    doc_list = ""
    for i, doc in enumerate(documents):
        doc_list += f"[문서 {i+1}] {doc}\n\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""질문에 가장 관련성 높은 문서 {top_k}개를 골라 번호만 출력하세요.
관련성 높은 순서대로, 쉼표로 구분하세요.
예시: 3,7,1"""},
            {"role": "user", "content": f"질문: {question}\n\n{doc_list}"}
        ]
    )

    answer = response.choices[0].message.content.strip()
    indices = [int(x.strip()) - 1 for x in answer.split(",") if x.strip().isdigit()]
    return indices

indices = rerank(question, results['documents'][0])

print(f"\n=== Re-ranking 후 Top 3 ===")
for rank, idx in enumerate(indices):
    doc = results['documents'][0][idx]
    meta = results['metadatas'][0][idx]
    print(f"{rank+1}. (원래 {idx+1}위) [{meta['category']}] {doc[:80]}")
```

### 셀 3: 비교 - Re-ranking 있을 때 vs 없을 때

```python
def rag_with_rerank(question, n_candidates=10, top_k=3):
    q_resp = client.embeddings.create(input=[question], model="text-embedding-3-small")
    results = collection.query(query_embeddings=[q_resp.data[0].embedding], n_results=n_candidates)

    indices = rerank(question, results['documents'][0], top_k=top_k)
    reranked_docs = [results['documents'][0][i] for i in indices]

    context = "\n\n".join([f"[참고자료 {i+1}]\n{doc}" for i, doc in enumerate(reranked_docs)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "참고자료를 기반으로 답변하세요. 없는 내용은 '해당 정보를 찾을 수 없습니다'라고 하세요."},
            {"role": "user", "content": f"{context}\n\n질문: {question}"}
        ]
    )
    return response.choices[0].message.content

question = "이란 전쟁의 경제적 영향은?"

print("=== Re-ranking 없이 (Top 3) ===")
q_resp = client.embeddings.create(input=[question], model="text-embedding-3-small")
r = collection.query(query_embeddings=[q_resp.data[0].embedding], n_results=3)
context = "\n\n".join([f"[참고자료 {i+1}]\n{doc}" for i, doc in enumerate(r['documents'][0])])
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "참고자료를 기반으로 답변하세요."},
        {"role": "user", "content": f"{context}\n\n질문: {question}"}
    ]
)
print(resp.choices[0].message.content[:200])

print("\n=== Re-ranking 후 (Top 3) ===")
answer = rag_with_rerank(question)
print(answer[:200])
```

## 관찰할 것

1. 벡터 검색 Top 3과 Re-ranking 후 Top 3가 **다른가?**
2. Re-ranking 후 답변이 **질문 의도에 더 맞나?**
3. 단점: API 호출 추가 → **속도와 비용 트레이드오프**
