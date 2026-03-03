# PDF RAG Q&A System

PDF 문서를 업로드하고, 문서 내용을 기반으로 질문에 답변하는 RAG(Retrieval-Augmented Generation) 시스템.

## 기술 스택

| 구분 | 기술 |
|------|------|
| Backend | FastAPI + Jinja2 |
| LLM | OpenAI GPT-4o |
| Embedding | OpenAI text-embedding-3-small |
| Vector DB | Chroma (HNSW 인덱스) |
| PDF 처리 | PyMuPDF (텍스트/이미지) + pdfplumber (테이블) + EasyOCR (OCR) |
| Frontend | Tailwind CSS (CDN) + 바닐라 JS + SSE 스트리밍 |

## 프로젝트 구조

```
packages/rag-chroma-openai/
├── main.py                      # uvicorn 서버 실행 진입점
├── pyproject.toml               # 패키지 의존성 관리 (uv workspace member)
│
├── app/
│   ├── main.py                  # FastAPI 앱 (lifespan, 라우팅, 정적파일)
│   ├── config.py                # Pydantic Settings (환경변수, 경로, 모델 설정)
│   │
│   ├── models/
│   │   └── schemas.py           # 요청/응답 Pydantic 모델
│   │
│   ├── routers/
│   │   └── rag.py               # API 엔드포인트 (업로드, 검색, 문서관리)
│   │
│   ├── services/
│   │   ├── pdf_processor.py     # PDF 파싱 (텍스트 + 테이블 + 이미지 OCR)
│   │   ├── embedder.py          # 임베딩 생성 + Chroma 벡터 DB 관리
│   │   ├── retriever.py         # MMR 검색 + 컨텍스트 포맷팅
│   │   └── llm_chain.py         # LLM 호출 (동기/스트리밍)
│   │
│   ├── templates/
│   │   ├── base.html            # 기본 레이아웃 (Tailwind + Noto Sans KR)
│   │   ├── index.html           # 메인 페이지 (2컬럼 레이아웃)
│   │   └── components/
│   │       ├── upload.html      # 드래그앤드롭 업로드 영역
│   │       ├── search.html      # 검색 입력 바
│   │       └── results.html     # 스트리밍 답변 + 출처 표시
│   │
│   └── static/
│       ├── css/style.css        # 커스텀 스타일
│       └── js/app.js            # 업로드, SSE 스트리밍, 문서 CRUD
│
├── uploads/                     # 업로드된 PDF 저장 (자동 생성)
├── chroma_db/                   # Chroma 벡터 DB 저장소 (자동 생성)
└── documents_registry.json      # 문서 메타데이터 레지스트리 (자동 생성)
```

## RAG 파이프라인 상세

### 전체 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                     Indexing Pipeline                        │
│                                                             │
│  PDF 업로드                                                  │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────┐                   │
│  │         pdf_processor.py             │                   │
│  │                                      │                   │
│  │  PyMuPDF ──→ 텍스트 추출 (페이지별)    │                   │
│  │  PyMuPDF ──→ 이미지 추출 ──→ EasyOCR  │                   │
│  │  pdfplumber ──→ 테이블 → Markdown     │                   │
│  └──────────┬───────────────────────────┘                   │
│             │ raw_blocks (text/table/image_ocr)              │
│             ▼                                                │
│  ┌──────────────────────────────────────┐                   │
│  │   RecursiveCharacterTextSplitter     │                   │
│  │                                      │                   │
│  │  텍스트 블록 → 1000자 단위 분할        │                   │
│  │  테이블/OCR 블록 → 분할하지 않음       │                   │
│  └──────────┬───────────────────────────┘                   │
│             │ chunks (text + metadata)                       │
│             ▼                                                │
│  ┌──────────────────────────────────────┐                   │
│  │          embedder.py                 │                   │
│  │                                      │                   │
│  │  text-embedding-3-small로 벡터 변환   │                   │
│  │  Chroma DB에 벡터 + 메타데이터 저장    │                   │
│  │  JSON 레지스트리에 문서 정보 기록       │                   │
│  └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Query Pipeline                          │
│                                                             │
│  사용자 질문                                                  │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────────┐                   │
│  │          retriever.py                │                   │
│  │                                      │                   │
│  │  질문 → text-embedding-3-small 벡터화  │                   │
│  │  Chroma MMR 검색 (상위 k개)           │                   │
│  │  출처 메타데이터 추출                   │                   │
│  │  컨텍스트 문자열 포맷팅                 │                   │
│  └──────────┬───────────────────────────┘                   │
│             │ (context, sources)                             │
│             ▼                                                │
│  ┌──────────────────────────────────────┐                   │
│  │          llm_chain.py                │                   │
│  │                                      │                   │
│  │  시스템 프롬프트 + 컨텍스트 + 질문     │                   │
│  │  GPT-4o 호출                          │                   │
│  │  SSE 스트리밍으로 실시간 응답 전송      │                   │
│  └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### 1단계: PDF 파싱 (`pdf_processor.py`)

PDF에서 세 가지 유형의 콘텐츠를 추출한다.

| 콘텐츠 유형 | 추출 도구 | 처리 방식 |
|-------------|----------|----------|
| **텍스트** | PyMuPDF (`fitz`) | 페이지별 `get_text("text")`로 추출 |
| **테이블** | pdfplumber | `extract_tables()` → Markdown 표 변환 |
| **이미지 텍스트** | PyMuPDF + EasyOCR | 이미지 추출 → 한국어/영어 OCR |

**EasyOCR lazy loading**: OCR 모델(~250MB)은 이미지가 포함된 PDF를 처리할 때만 메모리에 로드된다.
첫 호출 시 `@property`로 초기화하고 이후 재사용.

```python
@property
def ocr_reader(self):
    if self._ocr_reader is None:
        import easyocr
        self._ocr_reader = easyocr.Reader(["ko", "en"], gpu=False)
    return self._ocr_reader
```

### 2단계: 청킹 (Chunking)

`RecursiveCharacterTextSplitter`로 텍스트를 의미 단위로 분할한다.

| 설정 | 값 | 설명 |
|------|---|------|
| `chunk_size` | 1000자 | 한 청크의 최대 크기 |
| `chunk_overlap` | 200자 | 청크 간 겹치는 영역 (문맥 유지) |
| `separators` | `\n\n` → `\n` → `. ` → ` ` | 우선순위별 분할 지점 |

**선택적 청킹**: 텍스트 블록만 분할하고, 테이블과 OCR 블록은 구조 보존을 위해 원본 그대로 유지한다.

```
텍스트 블록 (3000자) → [청크1: 1000자] [청크2: 1000자] [청크3: 1000자]
테이블 블록 (500자)  → [그대로 1개 청크로 저장]
OCR 블록 (300자)     → [그대로 1개 청크로 저장]
```

### 3단계: 임베딩 + 벡터 저장 (`embedder.py`)

각 청크를 벡터로 변환하여 Chroma DB에 저장한다.

- **임베딩 모델**: `text-embedding-3-small` (1536차원, OpenAI 최신 경량 모델)
- **벡터 DB**: Chroma (로컬 파일 기반, HNSW 인덱스로 ANN 검색)
- **청크 ID**: `{document_id}_{index}` 형식 → 문서 단위 삭제 가능

**메타데이터 구조** (각 청크에 포함):
```json
{
    "document_id": "a1b2c3d4e5f6",
    "filename": "보고서.pdf",
    "page": 3,
    "content_type": "text"  // "text" | "table" | "image_ocr"
}
```

**문서 레지스트리** (`documents_registry.json`):
Chroma는 문서 단위 관리를 지원하지 않으므로, JSON 사이드카 파일로 문서 메타데이터를 별도 관리한다.
```json
{
    "a1b2c3d4e5f6": {
        "document_id": "a1b2c3d4e5f6",
        "filename": "보고서.pdf",
        "page_count": 15,
        "chunk_count": 47,
        "uploaded_at": "2026-03-03T12:00:00"
    }
}
```

### 4단계: 검색 (`retriever.py`)

사용자 질문을 벡터로 변환하여 유사한 청크를 검색한다.

**MMR (Maximal Marginal Relevance) 검색**:
일반 유사도 검색은 비슷한 청크를 중복 반환할 수 있다. MMR은 관련성과 다양성을 동시에 고려하여
서로 다른 정보를 담은 청크를 반환한다.

```
일반 검색: [청크A, 청크A', 청크A'', 청크B, 청크C]  ← 비슷한 내용 중복
MMR 검색:  [청크A, 청크B, 청크C, 청크D, 청크E]     ← 다양한 정보 포함
```

**컨텍스트 포맷팅**: 검색된 청크를 출처 정보와 함께 하나의 문자열로 조합한다.
```
[출처 1] (문서: 보고서.pdf, 페이지: 3, 유형: text)
검색된 텍스트 내용...

---

[출처 2] (문서: 보고서.pdf, 페이지: 7, 유형: table)
[테이블]
| 항목 | 수치 |
| --- | --- |
| 매출 | 100억 |
```

### 5단계: LLM 응답 생성 (`llm_chain.py`)

검색된 컨텍스트와 질문을 GPT-4o에 전달하여 답변을 생성한다.

**시스템 프롬프트 핵심 규칙**:
1. 컨텍스트에 없는 내용은 답변하지 않음 (hallucination 방지)
2. 출처(문서명, 페이지)를 명시
3. 테이블 구조 유지
4. 한국어 답변, 고유명사/기술 용어는 원문 유지

**프롬프트 구성**:
```
[System] 시스템 프롬프트 (답변 규칙)
[User]   ## 컨텍스트
          {검색된 청크들}
          ## 질문
          {사용자 질문}
          ## 답변
```

**SSE 스트리밍**: OpenAI의 `stream=True` 옵션으로 토큰 단위 생성.
각 토큰을 `text/event-stream` 형식으로 브라우저에 실시간 전송한다.

```
data: {"type": "sources", "content": [...]}     ← 출처 먼저 전송
data: {"type": "chunk", "content": "답변"}       ← 토큰 단위 스트리밍
data: {"type": "chunk", "content": "의 첫"}
data: {"type": "chunk", "content": " 번째"}
data: {"type": "done"}                           ← 완료 신호
```

### 서비스 초기화 (DI)

모든 서비스는 FastAPI `lifespan`에서 싱글톤으로 한 번만 초기화되고,
`Request.app.state`를 통해 각 엔드포인트에 주입된다.

```
앱 시작 (lifespan)
    ├── PDFProcessor()          ← 텍스트 스플리터 초기화
    ├── EmbedderService()       ← OpenAI Embeddings + Chroma 연결
    ├── RetrieverService(embedder)  ← embedder 참조
    └── LLMChain()              ← AsyncOpenAI 클라이언트
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/api/documents/upload` | PDF 업로드 + 처리 |
| `GET` | `/api/documents` | 문서 목록 조회 |
| `DELETE` | `/api/documents/{id}` | 문서 삭제 |
| `POST` | `/api/search` | 일반 검색 (전체 응답) |
| `POST` | `/api/search/stream` | SSE 스트리밍 검색 |
| `GET` | `/api/health` | 헬스체크 |

## 설정 및 실행

```bash
# 1. 워크스페이스 루트에서 의존성 설치
uv sync --all-packages

# 2. 환경변수 설정
# 워크스페이스 루트의 .env 파일에 OPENAI_API_KEY 입력

# 3. 서버 실행 (packages/rag-chroma-openai 디렉터리에서)
cd packages/rag-chroma-openai
uv run python main.py

# 또는 워크스페이스 루트에서 직접 실행
uv run --package rag-chroma-openai python packages/rag-chroma-openai/main.py

# 4. 브라우저 접속
# http://localhost:8000
```

## 핵심 설계 결정

- **OCR lazy loading**: EasyOCR 모델은 이미지가 포함된 PDF 처리 시에만 로드
- **서비스 싱글톤**: FastAPI lifespan으로 한 번만 초기화, Request.app.state로 주입
- **선택적 청킹**: 텍스트는 RecursiveCharacterTextSplitter로 분할, 테이블/OCR은 구조 보존
- **MMR 검색**: 중복 청크 방지를 위해 Maximal Marginal Relevance 사용
- **문서 레지스트리**: JSON 사이드카 파일로 문서 단위 메타데이터 관리
