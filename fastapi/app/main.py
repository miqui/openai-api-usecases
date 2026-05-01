"""FastAPI application — OpenAI API learning demo.

Exposes three endpoint groups:
  /chat       — use case 1: completions + structured extraction
  /vision     — use case 5: image description (URL + file upload)
  /embeddings — use case 6: embed + cosine similarity ranking
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging import configure_logging, get_logger
from app.routers import chat, embeddings, vision

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    logger.info("startup", service="openai-fastapi-demo")
    yield
    logger.info("shutdown", service="openai-fastapi-demo")


app = FastAPI(
    title="OpenAI API Learning — FastAPI Demo",
    description=(
        "Demonstrates OpenAI use cases 1 (chat), 5 (vision), and 6 (embeddings) "
        "via REST endpoints."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(vision.router)
app.include_router(embeddings.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
