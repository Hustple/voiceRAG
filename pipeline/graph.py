"""
LangGraph StateGraph — full Self-CRAG pipeline.

T5 update: real CRAG routing + Self-RAG for moderate/complex queries.

Flow:
  asr → classifier → [hyde if complex] → retriever →
  crag_evaluator →
    INCORRECT → END (abstain)
    CORRECT/AMBIGUOUS + simple → generator → citation_builder → END
    CORRECT/AMBIGUOUS + moderate/complex → self_rag →
      IsUSE=yes → citation_builder → END
      IsUSE=no  → retriever (retry, max once)
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END
from pipeline.state import PipelineState
from pipeline.nodes.classifier import classifier_node
from pipeline.nodes.hyde import hyde_node
from pipeline.nodes.retriever import retriever_node
from pipeline.nodes.crag_evaluator import crag_evaluator_node
from pipeline.nodes.self_rag import self_rag_node
from pipeline.nodes.stub_nodes import (
    asr_node,
    generator_node,
    citation_builder_node,
)


def _route_after_classifier(state: PipelineState) -> str:
    return "hyde" if state.get("route") == "complex" else "retriever"


def _route_after_crag(state: PipelineState) -> str:
    """
    INCORRECT → end immediately (abstain).
    simple route → skip Self-RAG, go straight to generator.
    moderate/complex → run Self-RAG reflection.
    """
    action = state.get("crag_action", "CORRECT")
    if action == "INCORRECT":
        return "end"
    route = state.get("route", "moderate")
    if route == "simple":
        return "generator"
    return "self_rag"


def _route_after_self_rag(state: PipelineState) -> str:
    """
    If Self-RAG flagged answer as not useful and retries remain → retriever.
    Otherwise → generator.
    """
    answer = state.get("answer", "")
    retries = state.get("self_rag_retries", 0)
    if not answer and retries <= settings_max_retries():
        return "retriever"
    return "generator"


def settings_max_retries() -> int:
    from app.config import settings
    return settings.SELF_RAG_MAX_RETRIES


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("asr",              asr_node)
    graph.add_node("classifier",       classifier_node)
    graph.add_node("hyde",             hyde_node)
    graph.add_node("retriever",        retriever_node)
    graph.add_node("crag_evaluator",   crag_evaluator_node)
    graph.add_node("self_rag",         self_rag_node)
    graph.add_node("generator",        generator_node)
    graph.add_node("citation_builder", citation_builder_node)

    graph.set_entry_point("asr")
    graph.add_edge("asr", "classifier")

    graph.add_conditional_edges(
        "classifier",
        _route_after_classifier,
        {"hyde": "hyde", "retriever": "retriever"},
    )
    graph.add_edge("hyde", "retriever")
    graph.add_edge("retriever", "crag_evaluator")

    graph.add_conditional_edges(
        "crag_evaluator",
        _route_after_crag,
        {"generator": "generator", "self_rag": "self_rag", "end": END},
    )

    graph.add_conditional_edges(
        "self_rag",
        _route_after_self_rag,
        {"retriever": "retriever", "generator": "generator"},
    )

    graph.add_edge("generator",        "citation_builder")
    graph.add_edge("citation_builder", END)

    return graph


pipeline = build_graph().compile()
