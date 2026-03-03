# Graph RAG Neo4j

Neo4j 지식 그래프 기반 Graph RAG 시스템. PDF 문서에서 LLM으로 엔티티/관계를 추출하여 지식 그래프를 구축하고, **벡터 검색 + 그래프 탐색** 하이브리드 검색으로 질의응답을 수행한다.

## 기술 스택

| 구분 | 기술 |
|------|------|
| Graph DB | Neo4j 5.x (Docker) — 그래프 저장 + 내장 벡터 인덱스 |
| Driver | neo4j (Python async) |
| LLM | OpenAI GPT-4o — 답변 생성 + 엔티티/관계 추출 |
| Embedding | OpenAI text-embedding-3-small (1536차원) |
| Backend | FastAPI + Jinja2 |
| Frontend | Tailwind CSS |
| PDF 파싱 | PyMuPDF + pdfplumber + EasyOCR |

## 프로젝트 구조

```
packages/graph-rag-neo4j/
├── pyproject.toml                  # 패키지 설정 및 의존성
├── docker-compose.yml              # Neo4j 컨테이너
├── main.py                         # uvicorn 진입점 (port 8002)
│
├── app/
│   ├── config.py                   # 환경변수 설정 (Neo4j, OpenAI 등)
│   ├── main.py                     # FastAPI 앱 + lifespan (Neo4j 연결 관리)
│   │
│   ├── models/
│   │   └── schemas.py              # 요청/응답 스키마 (GraphStatsResponse 등)
│   │
│   ├── routers/
│   │   └── rag.py                  # API 엔드포인트 (업로드, 검색, 삭제, 그래프 통계)
│   │
│   ├── services/
│   │   ├── neo4j_client.py         # Neo4j 드라이버 래퍼 + 스키마/인덱스 초기화
│   │   ├── pdf_processor.py        # PDF → 텍스트/테이블/OCR → 청크 분할
│   │   ├── embedder.py             # 임베딩 생성 → Neo4j Document/Chunk 노드 저장
│   │   ├── graph_builder.py        # LLM 엔티티 추출 → Entity/RELATES_TO 저장
│   │   ├── retriever.py            # 하이브리드 검색 (벡터 + 그래프 탐색)
│   │   └── llm_chain.py            # 그래프 컨텍스트 인식 LLM 체인 (스트리밍)
│   │
│   ├── templates/
│   │   ├── base.html               # 공통 레이아웃 (Tailwind CDN)
│   │   ├── index.html              # 메인 페이지 (3컬럼: 문서|검색|그래프)
│   │   └── components/
│   │       ├── upload.html          # 파일 업로드 (드래그 앤 드롭)
│   │       ├── search.html          # 검색 입력
│   │       ├── results.html         # 답변 + 출처 + 그래프 관계 표시
│   │       └── graph_panel.html     # 지식 그래프 통계 패널
│   │
│   └── static/
│       ├── css/style.css           # 검색 방법 뱃지, 그래프 관계 스타일
│       └── js/app.js               # SSE 스트리밍, 문서 관리, 그래프 통계
```

## Neo4j 그래프 스키마

```
(:Document {document_id, filename, page_count, chunk_count, uploaded_at})
    -[:HAS_CHUNK]->
(:Chunk {chunk_id, text, page, content_type, embedding, seq})
    -[:NEXT_CHUNK]->  (다음 청크)
    -[:MENTIONS]->
(:Entity {name, type, description})
    -[:RELATES_TO {type}]->  (엔티티 간 관계)
```

**인덱스**:
- 벡터 인덱스: `Chunk.embedding` (cosine, 1536차원)
- 풀텍스트 인덱스: `Entity.name`
- 유니크 제약조건: `Document.document_id`, `Chunk.chunk_id`, `Entity.name`

## 하이브리드 검색 알고리즘

1. 질문을 벡터화 → Neo4j 벡터 인덱스 top-k 검색
2. 질문에서 LLM으로 엔티티명 추출
3. 풀텍스트 인덱스로 엔티티 매칭
4. 매칭된 엔티티에서 N홉 그래프 탐색 → 관련 청크 + 관계 수집
5. 벡터 + 그래프 결과 병합/중복제거 (hybrid 태깅)
6. 지식 그래프 관계 + 청크 텍스트 → 컨텍스트로 구성 → LLM 전달

## 실행 방법

### 1. 사전 요구사항

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (패키지 매니저)
- [Docker](https://www.docker.com/) (Neo4j 실행용)

### 2. 환경변수 설정

프로젝트 루트(`RAG-practice/`) 또는 패키지 디렉토리에 `.env` 파일 생성:

```env
OPENAI_API_KEY=sk-...
```

Neo4j 접속 정보는 기본값 사용 시 별도 설정 불필요 (변경하려면 `.env`에 추가):

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### 3. Neo4j 실행

```bash
cd packages/graph-rag-neo4j
docker compose up -d
```

Neo4j Browser: http://localhost:7474 (ID: `neo4j` / PW: `password`)

### 4. 의존성 설치

```bash
# 프로젝트 루트에서
uv sync
```

### 5. 서버 실행

```bash
cd packages/graph-rag-neo4j
uv run python main.py
```

웹 UI: http://localhost:8002

### 6. 사용법

1. 웹 UI에서 PDF 파일 업로드 (드래그 앤 드롭 또는 클릭)
   - 자동으로 텍스트 추출, 임베딩 생성, 엔티티/관계 추출이 수행됨
2. 질문 입력 후 검색
   - 벡터 + 그래프 하이브리드 검색 결과로 스트리밍 답변 생성
   - 각 출처에 검색 방법 뱃지 표시 (`vector` / `graph` / `hybrid`)
3. 좌측 패널에서 지식 그래프 통계 확인
4. Neo4j Browser에서 그래프 직접 탐색 가능

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/api/documents/upload` | PDF 업로드 (임베딩 + 그래프 구축) |
| `GET` | `/api/documents` | 업로드된 문서 목록 |
| `DELETE` | `/api/documents/{document_id}` | 문서 삭제 (청크 + 엔티티 정리) |
| `POST` | `/api/search` | 질의응답 (동기) |
| `POST` | `/api/search/stream` | 질의응답 (SSE 스트리밍) |
| `GET` | `/api/graph/stats` | 그래프 통계 (문서/청크/엔티티/관계 수) |
| `GET` | `/api/health` | 헬스 체크 |

## 환경변수 목록

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OPENAI_API_KEY` | (필수) | OpenAI API 키 |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt 주소 |
| `NEO4J_USER` | `neo4j` | Neo4j 사용자 |
| `NEO4J_PASSWORD` | `password` | Neo4j 비밀번호 |
| `CHUNK_SIZE` | `1000` | 텍스트 청크 크기 |
| `CHUNK_OVERLAP` | `200` | 청크 간 겹침 |
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
