from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    t = time.perf_counter()
    logger.info("voicerag.startup", env=settings.APP_ENV)
    from storage.chroma_client import get_chroma_client
    from storage.query_log import init_query_log

    client = get_chroma_client()
    collections = client.list_collections()
    logger.info("chroma.ready", collections=[c.name for c in collections])
    init_query_log()
    logger.info("query_log.ready", path=settings.QUERY_LOG_PATH)
    logger.info("voicerag.ready", startup_seconds=round(time.perf_counter() - t, 2))
    yield
    logger.info("voicerag.shutdown")


def create_app() -> FastAPI:
    application = FastAPI(
        title="VoiceRAG",
        description="Bilingual Hindi-English voice-driven RAG API using Self-CRAG.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.APP_ENV == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.APP_ENV == "development",
        log_level=settings.LOG_LEVEL.lower(),
    )
