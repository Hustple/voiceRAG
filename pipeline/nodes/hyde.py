"""
HyDE node — Hypothetical Document Embedding (Gao et al., 2022).

For complex queries, generates a short hypothetical answer and uses that
as the retrieval query instead of the raw user query.

Why this works for Hindi queries against English PDFs:
  - User asks: "MSME loan ke liye eligibility kya hai?"
  - HyDE generates: "MSME loan eligibility requires turnover below 250 crore..."
  - The hypothetical answer is in English domain language → better embedding match

Only activated for 'complex' route. Moderate and simple routes skip this node.
"""
from __future__ import annotations
import time
import structlog
from groq import Groq
from app.config import settings
from pipeline.state import PipelineState

logger = structlog.get_logger(__name__)

HYDE_SYSTEM = """You are a financial document expert on Indian government schemes.
Given a user question, write a short 2-3 sentence hypothetical answer as if it
came from an official Indian government document.

Write in English regardless of the question language.
Be specific — mention scheme names, amounts, eligibility criteria if relevant.
Do NOT say you don't know. Always generate a plausible answer."""


def hyde_node(state: PipelineState) -> PipelineState:
    t_start = time.perf_counter()

    # Only run for complex queries
    if state.get("route") != "complex":
        state["hyde_query"] = None
        latency_map = dict(state.get("latency_map") or {})
        latency_map["hyde"] = 0.0
        state["latency_map"] = latency_map
        return state

    query = state.get("transcript") or state.get("query", "")
    client = Groq(api_key=settings.GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        hyde_query = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("hyde.error", error=str(exc), fallback="original query")
        hyde_query = None  # Fall back to original query in retriever

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
    logger.info(
        "hyde.done",
        original=query[:60],
        hypothesis=hyde_query[:80] if hyde_query else "none",
        latency_ms=elapsed_ms,
    )

    latency_map = dict(state.get("latency_map") or {})
    latency_map["hyde"] = elapsed_ms
    state["hyde_query"] = hyde_query
    state["latency_map"] = latency_map
    return state
