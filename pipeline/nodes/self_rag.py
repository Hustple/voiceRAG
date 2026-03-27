"""
Self-RAG reflection node — Asai et al., 2023.

Only activated for moderate and complex routes (CRAG action = CORRECT or AMBIGUOUS).
Generates with special reflection tokens and checks if the output is useful.

Reflection tokens checked:
  [IsREL]  — are the retrieved chunks relevant?
  [IsSUP]  — is the generation supported by the chunks?
  [IsUSE]  — is the answer useful to the user?

If [IsUSE] = no, triggers one retrieval retry (up to SELF_RAG_MAX_RETRIES).

Combined with CRAG evaluator = Self-CRAG architecture (Wang et al., 2024).
"""
from __future__ import annotations
import time
import structlog
from groq import Groq
from app.config import settings
from pipeline.state import PipelineState, SourceChunk

logger = structlog.get_logger(__name__)

SELF_RAG_SYSTEM = """You are a financial document assistant for Indian government schemes.
Answer the query using ONLY the provided context chunks.

After your answer, append these reflection tokens on a new line:
[IsREL: yes/no] - are the chunks relevant to the query?
[IsSUP: yes/no] - is your answer supported by the chunks?
[IsUSE: yes/no] - is your answer useful and complete?

Format:
<your answer here>

[IsREL: yes] [IsSUP: yes] [IsUSE: yes]"""


def _parse_reflection_tokens(text: str) -> dict[str, str]:
    """Extract IsREL, IsSUP, IsUSE values from generated text."""
    import re
    tokens = {}
    for token in ["IsREL", "IsSUP", "IsUSE"]:
        match = re.search(rf"\[{token}:\s*(yes|no)\]", text, re.IGNORECASE)
        tokens[token] = match.group(1).lower() if match else "yes"
    return tokens


def _extract_answer(text: str) -> str:
    """Strip reflection tokens from the answer text."""
    import re
    # Remove the reflection token line
    clean = re.sub(r"\[Is(REL|SUP|USE):[^\]]+\]", "", text)
    return clean.strip()


def _format_context(chunks: list[SourceChunk]) -> str:
    """Format chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {chunk['doc_title']} (p.{chunk['page_num']}):\n{chunk['chunk_text']}"
        )
    return "\n\n".join(parts)


def self_rag_node(state: PipelineState) -> PipelineState:
    """
    Generate with reflection tokens. If IsUSE=no, flag for retry.
    The retry loop is handled by the graph edge back to retriever (T5 graph update).
    """
    t_start = time.perf_counter()

    # Skip Self-RAG for simple routes — not worth the extra latency
    route = state.get("route", "moderate")
    if route == "simple":
        state["self_rag_retries"] = state.get("self_rag_retries", 0)
        latency_map = dict(state.get("latency_map") or {})
        latency_map["self_rag"] = 0.0
        state["latency_map"] = latency_map
        return state

    query = state.get("transcript") or state.get("query", "")
    chunks: list[SourceChunk] = state.get("chunks", [])
    retries = state.get("self_rag_retries", 0)

    if not chunks:
        state["answer"] = ""
        state["self_rag_retries"] = retries
        return state

    context = _format_context(chunks)
    client = Groq(api_key=settings.GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SELF_RAG_SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"},
            ],
            temperature=0.1,
            max_tokens=600,
        )
        raw_output = response.choices[0].message.content.strip()
        tokens = _parse_reflection_tokens(raw_output)
        answer = _extract_answer(raw_output)

        logger.info(
            "self_rag.reflection",
            IsREL=tokens["IsREL"],
            IsSUP=tokens["IsSUP"],
            IsUSE=tokens["IsUSE"],
            retries=retries,
        )

        # If answer not useful and we have retries left, signal retry
        if tokens["IsUSE"] == "no" and retries < settings.SELF_RAG_MAX_RETRIES:
            logger.info("self_rag.retry_triggered", retries=retries + 1)
            state["self_rag_retries"] = retries + 1
            state["answer"] = ""  # Clear so generator knows to retry
        else:
            state["answer"] = answer
            state["self_rag_retries"] = retries

    except Exception as exc:
        logger.error("self_rag.error", error=str(exc))
        state["answer"] = ""
        state["self_rag_retries"] = retries

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
    latency_map = dict(state.get("latency_map") or {})
    latency_map["self_rag"] = latency_map.get("self_rag", 0.0) + elapsed_ms
    state["latency_map"] = latency_map
    return state
