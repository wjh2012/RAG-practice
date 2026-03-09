# Step 2: Global Search (커뮤니티 기반 검색)

## Local Search vs Global Search

```
Local Search:  "트럼프가 이란에 뭐라 했어?" → 특정 엔티티 중심
Global Search: "전체적인 흐름은?"          → 커뮤니티 요약 기반
```

## 커뮤니티 탐지 → 요약 → 검색

```
전체 그래프
    ↓ 커뮤니티 탐지
[커뮤니티 1: 이란-미국-트럼프]  → 요약: "미국과 이란 간 군사 충돌..."
[커뮤니티 2: 코스피-환율-유가]  → 요약: "중동 사태로 경제 악화..."
    ↓
질문에 관련된 커뮤니티 요약 → LLM → 포괄적 답변
```

## 실습 코드

### 셀 1: 커뮤니티 탐지 (Union-Find)

```python
def detect_communities():
    with driver.session() as session:
        result = session.run("""
            MATCH (n)-[r]-(m)
            RETURN n.name AS source, type(r) AS relation, m.name AS target
        """)
        edges = [(r['source'], r['relation'], r['target']) for r in result]

    parent = {}
    def find(x):
        if x not in parent: parent[x] = x
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        px, py = find(x), find(y)
        if px != py: parent[px] = py

    for src, rel, tgt in edges:
        union(src, tgt)

    communities = {}
    for src, rel, tgt in edges:
        root = find(src)
        if root not in communities:
            communities[root] = {"nodes": set(), "relations": []}
        communities[root]["nodes"].update([src, tgt])
        communities[root]["relations"].append(f"{src} -{rel}-> {tgt}")

    return sorted(communities.values(), key=lambda x: len(x["nodes"]), reverse=True)
```

### 셀 2: 커뮤니티 요약

```python
def summarize_communities(communities, max_communities=10):
    summaries = []
    for i, comm in enumerate(communities[:max_communities]):
        if len(comm['nodes']) < 3: continue
        relations_text = "\n".join(comm['relations'][:20])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "엔티티 관계들을 분석해서 이 그룹의 주제를 2-3문장으로 요약하세요."},
                {"role": "user", "content": f"엔티티: {list(comm['nodes'])[:15]}\n\n관계:\n{relations_text}"}
            ]
        )
        summaries.append({
            "id": i, "nodes": list(comm['nodes']),
            "summary": response.choices[0].message.content,
            "size": len(comm['nodes'])
        })
    return summaries
```

### 셀 3: Global Search

```python
def graph_global_search(question):
    all_summaries = "\n\n".join([
        f"[주제 그룹 {s['id']+1}] ({s['size']}개 엔티티)\n{s['summary']}"
        for s in summaries
    ])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "주제 그룹 요약을 종합해서 질문에 포괄적으로 답변하세요. 관련 없는 그룹은 무시."},
            {"role": "user", "content": f"{all_summaries}\n\n질문: {question}"}
        ]
    )
    return response.choices[0].message.content
```

## 최적 방식 정리

| 질문 유형 | 최적 방식 |
|---|---|
| 특정 엔티티 관련 | Local Search |
| 전체 요약/흐름 | Global Search |
| 단순 사실 확인 | Naive RAG |
