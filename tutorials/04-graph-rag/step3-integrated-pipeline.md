# Step 3: GraphRAG 통합 파이프라인

## 핵심: 질문 유형 자동 판단

```
질문 → LLM이 유형 판단
  → "구체적(local)" → Local Search + Hybrid 검색 → LLM 답변
  → "포괄적(global)" → Community 요약 + Hybrid 검색 → LLM 답변
```

## 실습 코드

### 질문 유형 판단기

```python
def classify_question(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """질문 유형을 판단하세요.
- "local": 특정 인물, 사건, 조직에 대한 구체적 질문
- "global": 전체 요약, 흐름, 주요 이슈 등 포괄적 질문
local 또는 global 한 단어만 출력."""},
            {"role": "user", "content": question}
        ]
    )
    answer = response.choices[0].message.content.strip().lower()
    return "global" if "global" in answer else "local"
```

### 통합 파이프라인

```python
def graph_rag_pipeline(question):
    q_type = classify_question(question)

    if q_type == "local":
        graph_context = graph_local_search(question)
        vector_docs = hybrid_search(question, top_k=3)
        context = f"[그래프 관계]\n{chr(10).join(graph_context[:20])}\n\n[관련 문서]\n{chr(10).join(vector_docs[:3])}"
    else:
        community_text = "\n\n".join([f"[주제 {s['id']+1}] {s['summary']}" for s in summaries])
        vector_docs = hybrid_search(question, top_k=3)
        context = f"[주제별 요약]\n{community_text}\n\n[관련 문서]\n{chr(10).join(vector_docs[:3])}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "제공된 정보를 종합해서 답변하세요. 없는 내용은 지어내지 마세요."},
            {"role": "user", "content": f"{context}\n\n질문: {question}"}
        ]
    )
    return response.choices[0].message.content
```

## 각 방식의 강점

| 질문 유형 | 최적 방식 |
|---|---|
| 특정 인물/사건 구체적 질문 | Local Search |
| 전체 요약/흐름/이슈 | Global Search |
| 단순 사실 확인 | Naive RAG |
