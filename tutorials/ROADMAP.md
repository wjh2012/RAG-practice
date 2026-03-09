# RAG → GraphRAG 학습 로드맵

## 1단계: Naive RAG (1~2주)

기본 파이프라인을 직접 구현. "벡터 검색만으로는 뭐가 부족한지" 체감하기.

- [x] **Step 0** - RAG 큰 그림 이해 → [step0-overview.md](01-naive-rag/step0-overview.md)
- [x] **Step 1** - 문서 로드 & 청킹 → [step1-chunking.md](01-naive-rag/step1-chunking.md)
- [x] **Step 2** - 임베딩 & 코사인 유사도 → [step2-embedding.md](01-naive-rag/step2-embedding.md)
- [x] **Step 3** - 벡터DB (ChromaDB) → [step3-vectordb.md](01-naive-rag/step3-vectordb.md)
- [x] **Step 4** - 검색 + LLM 답변 생성 → [step4-rag-complete.md](01-naive-rag/step4-rag-complete.md)

## 2단계: Advanced RAG (1~2주)

Naive RAG의 한계를 개선하는 기법들. "검색 품질"에 대한 감각 키우기.

- [x] **Step 0** - Naive RAG 한계 분석 → [step0-naive-rag-limits.md](02-advanced-rag/step0-naive-rag-limits.md)
- [x] **Step 1** - Semantic Chunking → [step1-semantic-chunking.md](02-advanced-rag/step1-semantic-chunking.md)
- [x] **Step 2** - Query Transformation (HyDE, Multi-Query) → [step2-query-transformation.md](02-advanced-rag/step2-query-transformation.md)
- [x] **Step 3** - Re-ranking (Cross-encoder) → [step3-reranking.md](02-advanced-rag/step3-reranking.md)
- [x] **Step 4** - Hybrid Search (BM25 + 벡터 검색) → [step4-hybrid-search.md](02-advanced-rag/step4-hybrid-search.md)
- [x] **Step 5** - Advanced RAG 통합 파이프라인 → [step5-integrated-pipeline.md](02-advanced-rag/step5-integrated-pipeline.md)

## 3단계: Knowledge Graph 기초 (1주)

GraphRAG 전에 지식 그래프 자체를 이해하기.

- [x] **Step 1** - 노드, 엣지, 트리플 개념 → [step1-node-edge-triple.md](03-knowledge-graph/step1-node-edge-triple.md)
- [x] **Step 2** - Neo4j 설치 & Cypher 쿼리 기초 → [step2-neo4j-cypher.md](03-knowledge-graph/step2-neo4j-cypher.md)
- [x] **Step 3** - 뉴스 데이터로 간단한 그래프 만들기 → [step3-news-graph.md](03-knowledge-graph/step3-news-graph.md)

## 4단계: GraphRAG (2~3주)

엔티티·관계 추출 → 그래프 구조화 → 커뮤니티 탐지 → 그래프 기반 검색.

- [x] **Step 1** - GraphRAG 개념 & Local Search 구현 → [step1-concept.md](04-graph-rag/step1-concept.md)
- [x] **Step 2** - Global Search (커뮤니티 기반 검색) → [step2-global-search.md](04-graph-rag/step2-global-search.md)
- [ ] **Step 3** - GraphRAG 통합 파이프라인
- [ ] **Step 4** - Naive RAG vs Advanced RAG vs GraphRAG 최종 비교

---

## 데이터셋

모든 단계에서 동일한 데이터셋 사용 → 각 방식의 답변 품질 비교 가능

- `Articles_20260305_125404.xlsx`: 네이버 뉴스 60개 기사 (정치/경제/사회/생활문화/IT과학/세계)

## 기술 스택

- Python, OpenAI API (`text-embedding-3-small`, `gpt-4o-mini`)
- ChromaDB (벡터DB)
- Neo4j (그래프DB) - 3단계부터
- uv 워크스페이스 (모노레포)
