"""
CRAG evaluator node — Corrective RAG (Yan et al., 2024).

Scores each retrieved chunk for relevance to the query using a Groq
grader call. Aggregates to a corpus-level confidence score and returns
one of three actions:

  CORRECT   (score >= CRAG_CORRECT_THRESHOLD)   → pass chunks to generator
  AMBIGUOUS (score between thresholds)           → strip low-relevance sentences
  INCORRECT (score < CRAG_INCORRECT_THRESHOLD)  → abstain, return structured response

The INCORRECT path short-circuits the pipeline — no LLM generation occurs,
preventing hallucination on out-of-corpus queries.
"""
from __future__ import annotations
import time
import structlog
from groq import Groq
from app.config import settings
from pipeline.state import PipelineState, SourceChunk

logger = structlog.get_logger(__name__)

GRADER_SYSTEM = """You are a relevance grader for a financial document RAG system.
Given a user query and a retrieved document chunk, score its relevance.

Respond with ONLY a number between 0.0 and 1.0:
  1.0 = perfectly relevant, directly answers the query
  0.5 = partially relevant, contains some useful information
  0.0 = not relevant at all

No explanation. Just the number."""


def _grade_chunk(client: Groq, query: str, chunk_text: str) -> float:
    """Score a single chunk for relevance to the query. Returns 0.0-1.0."""
    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": GRADER_SYSTEM},
                {"role": "user", "content": f"Query: {query}\n\nChunk: {chunk_text[:800]}"},
            ],
            temperature=0,
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        score = float(raw.split()[0].rstrip(".,"))
        return max(0.0, min(1.0, score))
    except Exception as exc:
        logger.warning("crag.grade_failed", error=str(exc))
        return 0.5  # neutral score on error


def _strip_irrelevant_sentences(text: str, query: str) -> str:
    """
    AMBIGUOUS path: remove sentences that are clearly off-topic.
    Simple heuristic — keep sentences containing query keywords.
    """
    query_words = set(query.lower().split())
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    kept = []
    for sent in sentences:
        sent_words = set(sent.lower().split())
        overlap = len(query_words & sent_words)
        if overlap > 0 or len(kept) == 0:  # always keep at least one sentence
            kept.append(sent)
    return ". ".join(kept) + "." if kept else text


def crag_evaluator_node(state: PipelineState) -> PipelineState:
    """
    Grade each retrieved chunk, compute aggregate score, set crag_action.
    """
    t_start = time.perf_counter()
    query = state.get("transcript") or state.get("query", "")
    chunks: list[SourceChunk] = state.get("chunks", [])

    if not chunks:
        logger.warning("crag.no_chunks", action="INCORRECT")
        state["crag_action"] = "INCORRECT"
        state["crag_score"] = 0.0
        latency_map = dict(state.get("latency_map") or {})
        latency_map["crag"] = round((time.perf_counter() - t_start) * 1000, 1)
        state["latency_map"] = latency_map
        return state

    client = Groq(api_key=settings.GROQ_API_KEY)

    # Grade each chunk individually
    scores: list[float] = []
    for chunk in chunks:
        score = _grade_chunk(client, query, chunk["chunk_text"])
        scores.append(score)

    # Aggregate: weighted average (top chunk weighted more)
    if scores:
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        total_weight = sum(weights)
        agg_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    else:
        agg_score = 0.0

    agg_score = round(agg_score, 4)
    correct_threshold = settings.CRAG_CORRECT_THRESHOLD
    incorrect_threshold = settings.CRAG_INCORRECT_THRESHOLD

    if agg_score >= correct_threshold:
        action = "CORRECT"
    elif agg_score >= incorrect_threshold:
        action = "AMBIGUOUS"
        # Strip low-relevance sentences from each chunk
        cleaned_chunks: list[SourceChunk] = []
        for chunk, score in zip(chunks, scores):
            if score >= incorrect_threshold:
                cleaned_text = _strip_irrelevant_sentences(chunk["chunk_text"], query)
                cleaned_chunks.append({**chunk, "chunk_text": cleaned_text})
        state["chunks"] = cleaned_chunks if cleaned_chunks else chunks
    else:
        action = "INCORRECT"

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
    logger.info(
        "crag.done",
        action=action,
        agg_score=agg_score,
        n_chunks=len(chunks),
        latency_ms=elapsed_ms,
    )

    latency_map = dict(state.get("latency_map") or {})
    latency_map["crag"] = elapsed_ms
    state["crag_action"] = action
    state["crag_score"] = agg_score
    state["latency_map"] = latency_map
    return state
