import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from neo4j import GraphDatabase

from app.config import settings
from app.routers import rag
from app.services.kg_service import KGService
from app.services.llm_chain import LLMChain
from app.services.retriever import RetrieverService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    # neo4j-graphrag uses the synchronous Neo4j driver
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    driver.verify_connectivity()
    logger.info("Neo4j connected")

    # Initialize KG service and indexes
    kg_service = KGService(driver)
    await asyncio.to_thread(kg_service.init_indexes)
    logger.info("Neo4j indexes initialized")

    # Retriever (reuses the embedder from kg_service)
    retriever = RetrieverService(driver, kg_service.embedder)

    app.state.driver = driver
    app.state.kg_service = kg_service
    app.state.retriever = retriever
    app.state.llm_chain = LLMChain()

    yield

    # Shutdown
    driver.close()
    logger.info("Neo4j connection closed")


app = FastAPI(title="Graph RAG Q&A (neo4j-graphrag)", lifespan=lifespan)

app_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(app_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(app_dir / "templates"))

app.include_router(rag.router)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
