# Step 4: 검색 + LLM 답변 생성 (RAG 완성!)

## 핵심 구조

```
질문 → 벡터DB 검색 → 관련 청크 3개 획득 → 프롬프트에 끼워넣기 → LLM 답변
```

LLM에게 보내는 프롬프트:

```
[시스템] 아래 참고자료를 기반으로 답변하세요. 참고자료에 없으면 "모르겠습니다"라고 하세요.

[참고자료]
청크1: ...
청크2: ...
청크3: ...

[질문] 중동 전쟁 상황은?
```

> "참고자료에 없으면 모르겠다고 하라" → hallucination 방지 핵심

## 실습 코드

### 셀 10: RAG 함수 만들기

```python
def rag(question, n_results=3):
    # 1. 질문 임베딩
    q_response = openai_client.embeddings.create(
        input=[question],
        model="text-embedding-3-small"
    )
    q_embedding = q_response.data[0].embedding

    # 2. 벡터DB에서 검색
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results
    )

    # 3. 검색 결과를 프롬프트에 조립
    context = ""
    for i, doc in enumerate(results['documents'][0]):
        context += f"[참고자료 {i+1}]\n{doc}\n\n"

    # 4. LLM에게 질문
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "아래 참고자료를 기반으로 질문에 답변하세요. 참고자료에 없는 내용은 '해당 정보를 찾을 수 없습니다'라고 하세요."},
            {"role": "user", "content": f"{context}\n질문: {question}"}
        ]
    )

    return response.choices[0].message.content, results
```

### 셀 11: 질문해보기

```python
answer, results = rag("중동 전쟁 상황은?")

print("=== 답변 ===")
print(answer)
print("\n=== 참고한 청크 ===")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\n[{i+1}] {meta['title'][:40]}")
    print(doc[:100])
```

### 셀 12: 여러 질문 비교

```python
questions = [
    "오늘 날씨 어때?",
    "주식 시장 상황은?",
    "이란 전쟁 관련 뉴스 알려줘",
    "AI 관련 최신 뉴스는?"
]

for q in questions:
    answer, _ = rag(q)
    print(f"Q: {q}")
    print(f"A: {answer[:150]}...")
    print("-" * 60)
```

## 관찰할 것

1. 답변이 **참고자료 기반**으로 나오나? 지어내진 않나?
2. 크롤링하지 않은 주제를 질문하면 어떻게 답하나?
3. **검색 결과가 별로면 답변도 별로** → Naive RAG의 근본 한계

## Naive RAG 완성!

전체 파이프라인:

```
문서 로드 → 청킹 → 임베딩 → 벡터DB 저장 → 검색 → LLM 답변
```

다음 단계(Advanced RAG)에서 이 파이프라인의 약점을 개선한다.
