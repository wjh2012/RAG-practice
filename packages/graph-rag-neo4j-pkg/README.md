# Graph RAG Neo4j (neo4j-graphrag 패키지)

Neo4j 공식 [`neo4j-graphrag`](https://pypi.org/project/neo4j-graphrag/) 패키지를 활용한 Graph RAG 시스템.
패키지가 제공하는 `SimpleKGPipeline`, `VectorCypherRetriever` 등을 사용하여 지식 그래프 구축과 하이브리드 검색을 구현한다.

> **참고**: 같은 기능을 직접 구현한 버전은 `packages/graph-rag-neo4j/`에 있다.
> 이 패키지는 공식 라이브러리를 최대한 활용하여 코드량을 줄이고 Entity Resolution 등 고급 기능을 자동으로 사용한다.

## neo4j-graphrag 활용 내역

| 영역 | 사용한 컴포넌트 | 설명 |
|------|-----------------|------|
| KG 구축 | `SimpleKGPipeline` | PDF 로드 → 청킹 → LLM 엔티티 추출 → 그래프 저장 → 엔티티 중복 해소 |
| 인덱스 | `create_vector_index`, `create_fulltext_index` | Chunk 벡터 인덱스 + Entity 풀텍스트 인덱스 |
| 검색 | `VectorCypherRetriever` | 벡터 유사도 검색 + Cypher 그래프 탐색을 한 번에 수행 |
| LLM | `OpenAILLM` | SimpleKGPipeline 내부 엔티티 추출용 |
| 임베딩 | `OpenAIEmbeddings` | Chunk 임베딩 + 검색 쿼리 임베딩 |

## 직접 구현 vs neo4j-graphrag 패키지 비교

| 영역 | `graph-rag-neo4j` (직접 구현) | `graph-rag-neo4j-pkg` (이 패키지) |
|------|------------------------------|----------------------------------|
| PDF 파싱 | PyMuPDF + pdfplumber + EasyOCR | 패키지 내장 PdfLoader |
| 청크 분할 | langchain RecursiveCharacterTextSplitter | 패키지 내장 FixedSizeSplitter |
| 엔티티 추출 | 직접 프롬프트 + JSON 파싱 | `LLMEntityRelationExtractor` (structured output) |
| 엔티티 중복 해소 | MERGE만 사용 | `SinglePropertyExactMatchResolver` 자동 적용 |
| 그래프 저장 | 직접 Cypher (UNWIND 배치) | `Neo4jWriter` (배치 자동 처리) |
| 벡터 검색 | 직접 Cypher 쿼리 | `VectorCypherRetriever` |
| 하이브리드 검색 | 벡터 + 풀텍스트 + 그래프 직접 조합 | `VectorCypherRetriever`의 retrieval_query로 그래프 탐색 |
| SSE 스트리밍 | OpenAI async 직접 호출 | 동일 (패키지 미지원) |

## 기술 스택

| 구분 | 기술 |
|------|------|
| Core | `neo4j-graphrag[openai]` >= 1.5.0 |
| Graph DB | Neo4j 5.x (Docker) |
| LLM | OpenAI GPT-4o |
| Embedding | OpenAI text-embedding-3-small (1536차원) |
| Backend | FastAPI + Jinja2 |
| Frontend | Tailwind CSS |

## 프로젝트 구조

```
packages/graph-rag-neo4j-pkg/
├── pyproject.toml                  # neo4j-graphrag[openai] 의존성
├── docker-compose.yml              # Neo4j 컨테이너
├── main.py                         # uvicorn 진입점 (port 8003)
│
├── app/
│   ├── config.py                   # 설정 (entity_types, relation_types 포함)
│   ├── main.py                     # FastAPI + lifespan (sync Neo4j 드라이버)
│   │
│   ├── models/
│   │   └── schemas.py              # 요청/응답 스키마
│   │
│   ├── routers/
│   │   └── rag.py                  # API 엔드포인트
│   │
│   ├── services/
│   │   ├── kg_service.py           # SimpleKGPipeline 래퍼 + 문서/그래프 관리
│   │   ├── retriever.py            # VectorCypherRetriever 래퍼
│   │   └── llm_chain.py            # SSE 스트리밍 LLM (OpenAI async)
│   │
│   ├── templates/                  # 웹 UI (그래프 패널 포함)
│   └── static/                     # JS + CSS
```

## Neo4j 그래프 스키마 (neo4j-graphrag 생성)

`SimpleKGPipeline`이 자동으로 생성하는 스키마:

```
(:Document {path, document_id, filename, page_count, ...})
    <-[:FROM_DOCUMENT]-
(:Chunk {text, embedding, index})
    -[:NEXT_CHUNK]->  (순서 연결)
    <-[:FROM_CHUNK]-
(:__Entity__:Person|Organization|... {name})
    -[:WORKS_FOR|LOCATED_IN|RELATES_TO|...]->  (엔티티 간 관계)
```

- `__Entity__`: 모든 엔티티의 공통 라벨
- 구체적 타입(Person, Organization 등)은 추가 라벨로 부여
- Entity Resolution이 자동으로 중복 엔티티를 병합

## 실행 방법

### 1. 사전 요구사항

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- [Docker](https://www.docker.com/)

### 2. 환경변수 설정

프로젝트 루트(`RAG-practice/`) 또는 패키지 디렉토리에 `.env`:

```env
OPENAI_API_KEY=sk-...
```

Neo4j 기본값 변경 시:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### 3. Neo4j 실행

```bash
cd packages/graph-rag-neo4j-pkg
docker compose up -d
```

Neo4j Browser: http://localhost:7474

### 4. 의존성 설치

```bash
# 프로젝트 루트에서
uv sync
```

### 5. 서버 실행

```bash
cd packages/graph-rag-neo4j-pkg
uv run python main.py
```

웹 UI: http://localhost:8003

### 6. 사용 흐름

1. PDF 업로드 → `SimpleKGPipeline`이 자동으로:
   - PDF 텍스트 추출
   - 청크 분할 + 임베딩 생성
   - LLM 엔티티/관계 추출
   - 그래프 저장 + Entity Resolution
2. 질문 입력 → `VectorCypherRetriever`가:
   - 벡터 유사도로 관련 청크 검색
   - Cypher로 그래프 탐색 (엔티티, 관계 수집)
3. 검색 결과 + 그래프 관계 → LLM 스트리밍 답변

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/api/documents/upload` | PDF 업로드 (SimpleKGPipeline 실행) |
| `GET` | `/api/documents` | 업로드된 문서 목록 |
| `DELETE` | `/api/documents/{document_id}` | 문서 삭제 |
| `POST` | `/api/search` | 질의응답 (동기) |
| `POST` | `/api/search/stream` | 질의응답 (SSE 스트리밍) |
| `GET` | `/api/graph/stats` | 그래프 통계 |
| `GET` | `/api/health` | 헬스 체크 |

## 환경변수 목록

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OPENAI_API_KEY` | (필수) | OpenAI API 키 |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt 주소 |
| `NEO4J_USER` | `neo4j` | Neo4j 사용자 |
| `NEO4J_PASSWORD` | `password` | Neo4j 비밀번호 |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 임베딩 모델 |
| `LLM_MODEL` | `gpt-4o` | LLM 모델 |
| `LLM_TEMPERATURE` | `0.1` | LLM 온도 |
| `SEARCH_K` | `5` | 벡터 검색 top-k |
| `GRAPH_TRAVERSAL_DEPTH` | `2` | 그래프 탐색 홉 수 |

## 종료 및 정리

```bash
# 서버 종료: Ctrl+C

# Neo4j 중지
docker compose down

# Neo4j 데이터까지 삭제
docker compose down -v
```
