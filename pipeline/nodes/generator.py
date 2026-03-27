"""
Generator node — streams grounded answers from Groq.

Handles two cases:
  1. Normal path (CORRECT/AMBIGUOUS): generate answer from chunks
  2. INCORRECT path: return structured abstain response immediately

Grounding prompt enforces that every claim references a chunk number [1], [2] etc.
Response is stored in state.answer as a complete string.
Streaming to the HTTP client is handled in app/routes.py, not here.
"""
from __future__ import annotations
import time
import structlog
from groq import Groq
from app.config import settings
from pipeline.state import PipelineState, SourceChunk

logger = structlog.get_logger(__name__)

GENERATOR_SYSTEM = """You are a financial document assistant specializing in Indian government schemes.
Answer the user's query using ONLY the provided context chunks.

Rules:
1. Cite every claim with chunk numbers like [1], [2] etc.
2. If the context does not contain enough information, say so explicitly.
3. Keep answers concise and factual — 3-5 sentences maximum.
4. Never invent facts not present in the context.
5. If the query is in Hindi, respond in Hindi. Otherwise respond in English."""

ABSTAIN_RESPONSE_EN = (
    "I don't have enough information in my knowledge base to answer this question accurately. "
    "Please consult official government sources like msme.gov.in or startupindia.gov.in."
)

ABSTAIN_RESPONSE_HI = (
    "मेरे पास इस प्रश्न का सटीक उत्तर देने के लिए पर्याप्त जानकारी नहीं है। "
    "कृपया msme.gov.in या startupindia.gov.in जैसे आधिकारिक सरकारी स्रोतों से परामर्श करें।"
)


def _format_context(chunks: list[SourceChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {chunk['doc_title']} (p.{chunk['page_num']}):\n{chunk['chunk_text']}"
        )
    return "\n\n".join(parts)


def _compute_confidence(crag_score: float, crag_action: str, self_rag_retries: int) -> float:
    """
    Derive a 0-1 confidence score from pipeline signals.
    Higher CRAG score + no retries = higher confidence.
    """
    base = crag_score if crag_action != "INCORRECT" else 0.0
    retry_penalty = 0.1 * self_rag_retries
    return round(max(0.0, min(1.0, base - retry_penalty)), 3)


def generator_node(state: PipelineState) -> PipelineState:
    t_start = time.perf_counter()

    lang = state.get("lang", "en")
    crag_action = state.get("crag_action", "CORRECT")
    chunks: list[SourceChunk] = state.get("chunks", [])
    crag_score = state.get("crag_score", 0.5)
    retries = state.get("self_rag_retries", 0)

    # INCORRECT path — abstain without calling the LLM
    if crag_action == "INCORRECT":
        answer = ABSTAIN_RESPONSE_HI if lang == "hi" else ABSTAIN_RESPONSE_EN
        state["answer"] = answer
        state["sources"] = []
        state["confidence"] = 0.0
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
        latency_map = dict(state.get("latency_map") or {})
        latency_map["generation"] = elapsed_ms
        state["latency_map"] = latency_map
        logger.info("generator.abstained", lang=lang)
        return state

    # If Self-RAG already produced an answer, use it directly
    existing_answer = state.get("answer", "")
    if existing_answer:
        state["sources"] = chunks
        state["confidence"] = _compute_confidence(crag_score, crag_action, retries)
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
        latency_map = dict(state.get("latency_map") or {})
        latency_map["generation"] = elapsed_ms
        state["latency_map"] = latency_map
        return state

    # Normal generation path
    if not chunks:
        state["answer"] = ABSTAIN_RESPONSE_HI if lang == "hi" else ABSTAIN_RESPONSE_EN
        state["sources"] = []
        state["confidence"] = 0.0
        return state

    context = _format_context(chunks)
    query = state.get("transcript") or state.get("query", "")
    client = Groq(api_key=settings.GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": GENERATOR_SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("generator.error", error=str(exc))
        answer = ABSTAIN_RESPONSE_HI if lang == "hi" else ABSTAIN_RESPONSE_EN
        chunks = []

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
    logger.info("generator.done", lang=lang, answer_len=len(answer), latency_ms=elapsed_ms)

    latency_map = dict(state.get("latency_map") or {})
    latency_map["generation"] = elapsed_ms
    state["answer"] = answer
    state["sources"] = chunks
    state["confidence"] = _compute_confidence(crag_score, crag_action, retries)
    state["latency_map"] = latency_map
    return state
