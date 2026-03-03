from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.routers import rag
from app.services.embedder import EmbedderService
from app.services.llm_chain import LLMChain
from app.services.pdf_processor import PDFProcessor
from app.services.retriever import RetrieverService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize singletons
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    app.state.pdf_processor = PDFProcessor()
    app.state.embedder = EmbedderService()
    app.state.retriever = RetrieverService(app.state.embedder)
    app.state.llm_chain = LLMChain()

    yield
    # Shutdown: cleanup if needed


app = FastAPI(title="PDF RAG Q&A System", lifespan=lifespan)

# Static files and templates
app_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(app_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(app_dir / "templates"))

# Register router
app.include_router(rag.router)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
