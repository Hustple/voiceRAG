"""
VoiceRAG API routes.

POST /query/text  — text query → SSE stream
POST /query/voice — audio file → SSE stream (T7 stub)
GET  /explain/{id} — routing trace
GET  /health       — system metrics
"""
from __future__ import annotations
import json
import uuid
import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from app.schemas import HealthResponse, TextQueryRequest
from pipeline.state import PipelineState

logger = structlog.get_logger(__name__)
router = APIRouter()


async def _run_pipeline_stream(query: str, lang_hint: str | None, query_id: str):
    """
    Run the full pipeline and yield SSE events.

    Events:
      data: {"type": "token", "content": "..."}   — streamed answer tokens
      data: {"type": "done", "payload": {...}}     — final metadata
      data: {"type": "error", "message": "..."}   — on failure
    """
    from pipeline.graph import pipeline

    initial_state = PipelineState(
        query=query,
        lang_hint=lang_hint,
        query_id=query_id,
        latency_map={},
        self_rag_retries=0,
    )

    try:
        result = pipeline.invoke(initial_state)

        answer: str = result.get("answer", "")
        sources = result.get("sources", [])
        confidence = result.get("confidence", 0.0)
        routing_path = result.get("route", "unknown")
        crag_action = result.get("crag_action", "UNKNOWN")
        latency_map = result.get("latency_map", {})
        retries = result.get("self_rag_retries", 0)

        # Stream answer word by word for UX responsiveness
        words = answer.split()
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # Final metadata event
        payload = {
            "query_id": query_id,
            "answer": answer,
            "lang": result.get("lang", "en"),
            "sources": [
                {
                    "chunk_id": s.get("chunk_id", ""),
                    "doc_title": s.get("doc_title", ""),
                    "page_num": s.get("page_num", 0),
                    "chunk_text": s.get("chunk_text", "")[:300],
                    "relevance_score": s.get("relevance_score", 0.0),
                }
                for s in sources
            ],
            "confidence": confidence,
            "routing_path": routing_path,
            "crag_action": crag_action,
            "self_rag_retries": retries,
            "latency_ms": latency_map,
        }
        yield f"data: {json.dumps({'type': 'done', 'payload': payload})}\n\n"

    except Exception as exc:
        logger.error("pipeline.error", error=str(exc), query_id=query_id)
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"


def _write_log(result_payload: dict) -> None:
    """Background task — write query record to SQLite after response completes."""
    from storage.query_log import write_query_record
    try:
        write_query_record(result_payload)
    except Exception as exc:
        logger.error("query_log.write_failed", error=str(exc))


# ── Health ─────────────────────────────────────────────────────────────────
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


# ── Text query ─────────────────────────────────────────────────────────────
@router.post("/query/text", tags=["query"])
async def query_text(body: TextQueryRequest, background_tasks: BackgroundTasks):
    """
    Accept a text query and stream back the grounded answer via SSE.
    """
    query_id = str(uuid.uuid4())
    logger.info("query.text", query_id=query_id, query=body.query[:80])

    return StreamingResponse(
        _run_pipeline_stream(body.query, body.lang, query_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Query-ID": query_id,
        },
    )


# ── Voice query (T7 stub) ──────────────────────────────────────────────────
@router.post("/query/voice", tags=["query"])
async def query_voice(audio: UploadFile) -> JSONResponse:
    return JSONResponse(
        content={"detail": "Voice endpoint not yet implemented (T7)"},
        status_code=501,
    )


# ── Explain ────────────────────────────────────────────────────────────────
@router.get("/explain/{query_id}", tags=["observability"])
async def explain(query_id: str) -> JSONResponse:
    from storage.query_log import get_query_record
    record = get_query_record(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No record for query_id={query_id}")
    return JSONResponse(content=record)
