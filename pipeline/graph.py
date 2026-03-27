"""
LangGraph StateGraph — full Self-CRAG pipeline.
T7 update: real ASR node replaces stub.
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END
from pipeline.state import PipelineState
from pipeline.nodes.asr import asr_node
from pipeline.nodes.classifier import classifier_node
from pipeline.nodes.hyde import hyde_node
from pipeline.nodes.retriever import retriever_node
from pipeline.nodes.crag_evaluator import crag_evaluator_node
from pipeline.nodes.self_rag import self_rag_node
from pipeline.nodes.generator import generator_node
from pipeline.nodes.citation_builder import citation_builder_node


def _route_after_classifier(state: PipelineState) -> str:
    return "hyde" if state.get("route") == "complex" else "retriever"


def _route_after_crag(state: PipelineState) -> str:
    action = state.get("crag_action", "CORRECT")
    if action == "INCORRECT":
        return "generator"
    route = state.get("route", "moderate")
    if route == "simple":
        return "generator"
    return "self_rag"


def _route_after_self_rag(state: PipelineState) -> str:
    answer = state.get("answer", "")
    retries = state.get("self_rag_retries", 0)
    from app.config import settings
    if not answer and retries <= settings.SELF_RAG_MAX_RETRIES:
        return "retriever"
    return "generator"


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
        {"generator": "generator", "self_rag": "self_rag"},
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
