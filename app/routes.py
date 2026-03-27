"""
VoiceRAG API routes — all endpoints.
"""
from __future__ import annotations
import json
import uuid
import structlog
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from app.schemas import HealthResponse, TextQueryRequest
from pipeline.state import PipelineState
from typing import Optional

logger = structlog.get_logger(__name__)
router = APIRouter()


async def _run_pipeline_stream(
    query: str,
    lang_hint: Optional[str],
    query_id: str,
    audio_bytes: Optional[bytes] = None,
):
    """Run pipeline and yield SSE events."""
    from pipeline.graph import pipeline

    initial_state = PipelineState(
        query=query,
        lang_hint=lang_hint,
        query_id=query_id,
        latency_map={},
        self_rag_retries=0,
        audio_bytes=audio_bytes,
    )

    try:
        result = pipeline.invoke(initial_state)

        answer: str = result.get("answer", "")
        sources = result.get("sources", [])

        # Stream answer word by word
        words = answer.split()
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

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
            "confidence": result.get("confidence", 0.0),
            "routing_path": result.get("route", "unknown"),
            "crag_action": result.get("crag_action", "UNKNOWN"),
            "self_rag_retries": result.get("self_rag_retries", 0),
            "latency_ms": result.get("latency_map", {}),
            "transcript": result.get("transcript", query),
        }
        yield f"data: {json.dumps({'type': 'done', 'payload': payload})}\n\n"

        # Log to SQLite in background
        from storage.query_log import write_query_record
        try:
            write_query_record({
                "query_id": query_id,
                "transcript": result.get("transcript", query),
                "lang": result.get("lang", "en"),
                "routing_path": result.get("route", "unknown"),
                "crag_action": result.get("crag_action", "UNKNOWN"),
                "self_rag_retries": result.get("self_rag_retries", 0),
                "confidence": result.get("confidence", 0.0),
                "latency_ms": result.get("latency_map", {}),
                "sources": sources,
            })
        except Exception as log_exc:
            logger.error("query_log.write_failed", error=str(log_exc))

    except Exception as exc:
        logger.error("pipeline.error", error=str(exc), query_id=query_id)
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"


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
async def query_text(body: TextQueryRequest):
    query_id = str(uuid.uuid4())
    logger.info("query.text", query_id=query_id, query=body.query[:80])
    return StreamingResponse(
        _run_pipeline_stream(body.query, body.lang, query_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Query-ID": query_id},
    )


# ── Voice query ────────────────────────────────────────────────────────────
@router.post("/query/voice", tags=["query"])
async def query_voice(
    audio: UploadFile = File(...),
    lang: Optional[str] = Form(default=None),
):
    """
    Accept an audio file and optional lang hint, stream back grounded answer.
    Supports WAV, MP3, M4A, OGG. Recommended: WAV 16kHz mono.
    """
    query_id = str(uuid.uuid4())
    logger.info(
        "query.voice",
        query_id=query_id,
        filename=audio.filename,
        content_type=audio.content_type,
        lang_hint=lang,
    )

    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    try:
        audio_bytes = await audio.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {exc}")

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    return StreamingResponse(
        _run_pipeline_stream(
            query="",           # transcript will be filled by ASR node
            lang_hint=lang,
            query_id=query_id,
            audio_bytes=audio_bytes,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Query-ID": query_id},
    )


# ── Explain ────────────────────────────────────────────────────────────────
@router.get("/explain/{query_id}", tags=["observability"])
async def explain(query_id: str) -> JSONResponse:
    from storage.query_log import get_query_record
    record = get_query_record(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No record for query_id={query_id}")
    return JSONResponse(content=record)
