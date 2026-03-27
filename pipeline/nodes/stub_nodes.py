"""
Stub implementations for T3 — replaced task by task in T4-T7.
Each stub passes state through without modification so the graph
compiles and can be tested end-to-end with the retriever.
"""
from __future__ import annotations
import uuid
from pipeline.state import PipelineState


def asr_node(state: PipelineState) -> PipelineState:
    """T7 — transcribe audio. Stub: pass query through as transcript."""
    state["transcript"] = state.get("query", "")
    state["lang"] = state.get("lang_hint") or "en"
    return state


def classifier_node(state: PipelineState) -> PipelineState:
    """T4 — classify query complexity. Stub: always 'moderate'."""
    state["route"] = "moderate"
    return state


def hyde_node(state: PipelineState) -> PipelineState:
    """T4 — HyDE query expansion. Stub: no expansion."""
    state["hyde_query"] = None
    return state


def crag_evaluator_node(state: PipelineState) -> PipelineState:
    """T5 — CRAG relevance grading. Stub: always CORRECT."""
    state["crag_action"] = "CORRECT"
    state["crag_score"] = 1.0
    return state


def self_rag_node(state: PipelineState) -> PipelineState:
    """T5 — Self-RAG reflection. Stub: no-op."""
    state["self_rag_retries"] = state.get("self_rag_retries", 0)
    return state


def generator_node(state: PipelineState) -> PipelineState:
    """T6 — LLM generation. Stub: echo query as answer."""
    state["answer"] = f"[STUB] Query was: {state.get('transcript', '')}"
    state["sources"] = state.get("chunks", [])
    state["confidence"] = 0.0
    return state


def citation_builder_node(state: PipelineState) -> PipelineState:
    """T6 — assemble citations. Stub: pass through."""
    if "query_id" not in state or not state.get("query_id"):
        state["query_id"] = str(uuid.uuid4())
    return state
