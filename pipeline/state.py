"""
PipelineState — the single shared TypedDict that flows through every
LangGraph node. All nodes read from and write to this dict.

Field naming convention:
  - Input fields   : query, lang_hint, audio_bytes
  - Computed fields: transcript, lang, route, hyde_query
  - Retrieval      : chunks, scores, crag_action
  - Generation     : answer, sources, confidence
  - Meta           : query_id, latency_map, self_rag_retries
"""

from __future__ import annotations

from typing import Optional

from typing_extensions import TypedDict


class SourceChunk(TypedDict):
    chunk_id: str
    doc_title: str
    page_num: int
    chunk_text: str
    relevance_score: float


class PipelineState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str  # raw text query (or ASR transcript)
    lang_hint: Optional[str]  # caller-provided hint: 'hi' | 'en' | None
    audio_bytes: Optional[bytes]  # raw audio for ASR node

    # ── ASR output ─────────────────────────────────────────────────────────
    transcript: str  # text from ASR (or passthrough of query)
    lang: str  # detected language: 'hi' | 'en'

    # ── Routing ────────────────────────────────────────────────────────────
    route: str  # 'simple' | 'moderate' | 'complex'

    # ── HyDE ───────────────────────────────────────────────────────────────
    hyde_query: Optional[str]  # hypothetical answer used as retrieval query

    # ── Retrieval ──────────────────────────────────────────────────────────
    chunks: list[SourceChunk]  # top-k chunks after MMR reranking
    raw_scores: list[float]  # cosine scores before MMR

    # ── CRAG ───────────────────────────────────────────────────────────────
    crag_action: str  # 'CORRECT' | 'AMBIGUOUS' | 'INCORRECT'
    crag_score: float  # aggregate corpus confidence score

    # ── Self-RAG ───────────────────────────────────────────────────────────
    self_rag_retries: int  # retry count (max = SELF_RAG_MAX_RETRIES)

    # ── Generation ─────────────────────────────────────────────────────────
    answer: str  # final grounded answer
    sources: list[SourceChunk]  # chunks cited in answer
    confidence: float  # overall response confidence 0-1

    # ── Meta ───────────────────────────────────────────────────────────────
    query_id: str  # UUID for /explain lookup
    latency_map: dict[str, float]  # per-node ms: {asr, classifier, ...}
    error: Optional[str]  # set if pipeline hits an unrecoverable error
