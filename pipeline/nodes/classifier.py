"""
Classifier node — determines query complexity using a single fast Groq call.

Returns one of three routes:
  simple   — direct factual lookup, no reflection needed
  moderate — some reasoning required, CRAG evaluate then conditionally Self-RAG
  complex  — multi-hop reasoning, HyDE expansion + full Self-CRAG loop

Uses temp=0 for deterministic output and a strict one-word response format.
"""

from __future__ import annotations

import time

import structlog
from groq import Groq

from app.config import settings
from pipeline.state import PipelineState

logger = structlog.get_logger(__name__)

VALID_ROUTES = {"simple", "moderate", "complex"}

CLASSIFIER_SYSTEM = """You are a query complexity classifier for a financial document RAG system.
Classify the user query into exactly one category:

simple   — single fact lookup (what is X, define Y, when did Z)
moderate — requires connecting 2-3 pieces of information
complex  — requires comparing multiple schemes, multi-step reasoning, or synthesis

Respond with ONLY one word: simple, moderate, or complex. No punctuation, no explanation."""


def classifier_node(state: PipelineState) -> PipelineState:
    t_start = time.perf_counter()
    query = state.get("transcript") or state.get("query", "")

    client = Groq(api_key=settings.GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=5,
        )
        raw = response.choices[0].message.content.strip().lower()
        # Strip punctuation and take first word only
        route = raw.split()[0].rstrip(".,;:") if raw else "moderate"
        if route not in VALID_ROUTES:
            logger.warning("classifier.invalid_route", raw=raw, fallback="moderate")
            route = "moderate"
    except Exception as exc:
        logger.error("classifier.error", error=str(exc), fallback="moderate")
        route = "moderate"

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
    logger.info("classifier.done", query=query[:60], route=route, latency_ms=elapsed_ms)

    latency_map = dict(state.get("latency_map") or {})
    latency_map["classifier"] = elapsed_ms
    state["route"] = route
    state["latency_map"] = latency_map
    return state
