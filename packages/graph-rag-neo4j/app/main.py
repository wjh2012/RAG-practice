import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.routers import rag
from app.services.embedder import EmbedderService
from app.services.graph_builder import GraphBuilder
from app.services.llm_chain import LLMChain
from app.services.neo4j_client import Neo4jClient
from app.services.pdf_processor import PDFProcessor
from app.services.retriever import RetrieverService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    neo4j = Neo4jClient()
    await neo4j.verify_connectivity()
    await neo4j.init_schema()
    logger.info("Neo4j connected and schema initialized")

    embedder = EmbedderService(neo4j)
    graph_builder = GraphBuilder(neo4j)

    app.state.neo4j = neo4j
    app.state.pdf_processor = PDFProcessor()
    app.state.embedder = embedder
    app.state.graph_builder = graph_builder
    app.state.retriever = RetrieverService(neo4j, embedder, graph_builder)
    app.state.llm_chain = LLMChain()

    yield

    # Shutdown
    await neo4j.close()
    logger.info("Neo4j connection closed")


app = FastAPI(title="Graph RAG Q&A System", lifespan=lifespan)

app_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(app_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(app_dir / "templates"))

app.include_router(rag.router)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
