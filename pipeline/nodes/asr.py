"""
ASR node — converts audio bytes to text transcript.

Routing logic:
  1. If SARVAM_API_KEY is set and lang is Hindi → use Sarvam Saaras API
  2. Otherwise → use OpenAI Whisper (local, no API key needed)

Language detection runs on the first 100 chars of transcript using langdetect.
If lang_hint is provided by the caller, it overrides detection.

Both ASR calls are wrapped in try/except — failures return empty transcript
so the pipeline can still attempt a text-based fallback.
"""
from __future__ import annotations
import io
import time
import tempfile
import os
import structlog

from app.config import settings
from pipeline.state import PipelineState

logger = structlog.get_logger(__name__)

SUPPORTED_LANGS = {"hi", "en"}


def _detect_language(text: str) -> str:
    """Detect language from first 100 chars. Falls back to 'en'."""
    try:
        from langdetect import detect
        lang = detect(text[:100])
        return lang if lang in SUPPORTED_LANGS else "en"
    except Exception:
        return "en"


def _transcribe_whisper(audio_bytes: bytes) -> str:
    """Transcribe audio using local Whisper model."""
    import whisper
    model = whisper.load_model(settings.WHISPER_MODEL)

    # Write to temp file — Whisper requires a file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, fp16=False)
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)


def _transcribe_saaras(audio_bytes: bytes, lang: str = "hi-IN") -> str:
    """Transcribe audio using Sarvam Saaras API."""
    import httpx
    response = httpx.post(
        "https://api.sarvam.ai/speech-to-text",
        headers={"API-Subscription-Key": settings.SARVAM_API_KEY},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"language_code": lang, "model": "saaras:v2"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json().get("transcript", "").strip()


def asr_node(state: PipelineState) -> PipelineState:
    """
    Transcribe audio_bytes → transcript + detected lang.
    If no audio_bytes present, pass query through as transcript (text path).
    """
    t_start = time.perf_counter()

    audio_bytes: bytes | None = state.get("audio_bytes")

    # Text path — no audio, just pass query through
    if not audio_bytes:
        query = state.get("query", "")
        lang_hint = state.get("lang_hint")
        lang = lang_hint if lang_hint in SUPPORTED_LANGS else _detect_language(query)
        state["transcript"] = query
        state["lang"] = lang
        latency_map = dict(state.get("latency_map") or {})
        latency_map["asr"] = 0.0
        state["latency_map"] = latency_map
        return state

    # Audio path
    lang_hint = state.get("lang_hint")
    transcript = ""

    # Try Saaras for Hindi if key available
    if settings.SARVAM_API_KEY and lang_hint == "hi":
        try:
            logger.info("asr.saaras", lang=lang_hint)
            transcript = _transcribe_saaras(audio_bytes, lang="hi-IN")
        except Exception as exc:
            logger.warning("asr.saaras_failed", error=str(exc), fallback="whisper")

    # Fallback to Whisper
    if not transcript:
        try:
            logger.info("asr.whisper", model=settings.WHISPER_MODEL)
            transcript = _transcribe_whisper(audio_bytes)
        except Exception as exc:
            logger.error("asr.whisper_failed", error=str(exc))
            transcript = ""

    # Detect language from transcript
    if lang_hint in SUPPORTED_LANGS:
        lang = lang_hint
    else:
        lang = _detect_language(transcript) if transcript else "en"

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
    logger.info(
        "asr.done",
        lang=lang,
        transcript_len=len(transcript),
        latency_ms=elapsed_ms,
    )

    latency_map = dict(state.get("latency_map") or {})
    latency_map["asr"] = elapsed_ms
    state["transcript"] = transcript
    state["lang"] = lang
    state["latency_map"] = latency_map
    return state
