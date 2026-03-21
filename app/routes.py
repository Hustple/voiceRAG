from __future__ import annotations
import structlog
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from app.schemas import HealthResponse

logger = structlog.get_logger(__name__)
router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    from storage.chroma_client import get_chroma_client
    from storage.query_log import get_health_metrics
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        chroma_ok = True
        corpus_chunks = sum(c.count() for c in collections)
    except Exception as exc:
        logger.error("chroma.health_check_failed", error=str(exc))
        chroma_ok = False
        corpus_chunks = 0
    metrics = get_health_metrics()
    return HealthResponse(
        status="ok" if chroma_ok else "degraded",
        chroma_ok=chroma_ok,
        corpus_chunks=corpus_chunks,
        **metrics,
    )

@router.post("/query/text", tags=["query"])
async def query_text(body: dict) -> JSONResponse:
    return JSONResponse(
        content={"detail": "Not yet implemented (T6)"},
        status_code=501,
    )

@router.post("/query/voice", tags=["query"])
async def query_voice(audio: UploadFile) -> JSONResponse:
    return JSONResponse(
        content={"detail": "Not yet implemented (T7)"},
        status_code=501,
    )

@router.get("/explain/{query_id}", tags=["observability"])
async def explain(query_id: str) -> JSONResponse:
    from storage.query_log import get_query_record
    record = get_query_record(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No record for query_id={query_id}")
    return JSONResponse(content=record)
