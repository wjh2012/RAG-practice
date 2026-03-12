# Step 4: Naive RAG vs Advanced RAG vs GraphRAG 최종 비교

## 3가지 파이프라인

```
Naive RAG:    문서 → 청킹 → 임베딩 → 벡터 검색 → LLM
Advanced RAG: + Semantic Chunking + Multi-Query + Hybrid Search + Rerank
GraphRAG:     + 지식 그래프 + Local/Global Search + 커뮤니티 탐지
```

## 비교 질문 5가지

1. 구체적 사실: "천궁 미사일의 요격 성공률은?"
2. 인물+관계: "트럼프와 이란의 관계를 설명해줘"
3. 간접 추론: "이란 전쟁이 한국 경제에 미치는 영향은?"
4. 전체 요약: "오늘 뉴스의 전체적인 흐름을 요약해줘"
5. 다수 엔티티: "중동 사태와 관련된 주요 인물들은?"

## 실습 코드

### 비교 실행

```python
questions = [
    "천궁 미사일의 요격 성공률은?",
    "트럼프와 이란의 관계를 설명해줘",
    "이란 전쟁이 한국 경제에 미치는 영향은?",
    "오늘 뉴스의 전체적인 흐름을 요약해줘",
    "중동 사태와 관련된 주요 인물들은?",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"\n[Naive RAG] {naive_rag(q)[:200]}")
    print(f"\n[Advanced RAG] {advanced_rag(q)[:200]}")
    print(f"\n[GraphRAG] {graph_rag_pipeline(q)[:200]}")
    print("=" * 60)
```

### LLM 자동 평가

```python
def evaluate(question, answers):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """세 가지 RAG 답변을 비교 평가.
각 답변: 정확성(1-5), 완성도(1-5), 추론력(1-5), 한줄평
형식: 방식명 | 정확성 | 완성도 | 추론력 | 한줄평"""},
            {"role": "user", "content": f"질문: {question}\n\n[Naive]{answers[0][:300]}\n\n[Advanced]{answers[1][:300]}\n\n[Graph]{answers[2][:300]}"}
        ]
    )
    return response.choices[0].message.content
```

## 최종 결론

| 방식 | 강점 | 약점 | 적합한 질문 |
|---|---|---|---|
| Naive RAG | 빠름, 단순 | 키워드/관계 추론 약함 | 단순 사실 확인 |
| Advanced RAG | 검색 품질 높음 | 관계 추론 불가 | 복잡한 검색 질문 |
| GraphRAG | 관계 추론, 전체 요약 | 느림, 비용 높음 | 관계/요약/추론 질문 |

**정답은 "하나만 쓰는 것"이 아니라 "질문에 맞게 골라 쓰는 것"**

## 로드맵 완주!
