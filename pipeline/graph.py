"""
LangGraph StateGraph — wires all pipeline nodes together.

T4 update: real conditional routing after classifier.
  simple/moderate → retriever directly
  complex         → hyde → retriever

CRAG and Self-RAG routing are still stubs (replaced in T5).
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END
from pipeline.state import PipelineState
from pipeline.nodes.classifier import classifier_node
from pipeline.nodes.hyde import hyde_node
from pipeline.nodes.retriever import retriever_node
from pipeline.nodes.stub_nodes import (
    asr_node,
    crag_evaluator_node,
    self_rag_node,
    generator_node,
    citation_builder_node,
)


def _route_after_classifier(state: PipelineState) -> str:
    """Route complex queries through HyDE; others go straight to retriever."""
    return "hyde" if state.get("route") == "complex" else "retriever"


def _route_after_crag(state: PipelineState) -> str:
    """T5 will implement real CRAG routing. Stub: always generate."""
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
        {"generator": "generator", "end": END},
    )

    graph.add_edge("self_rag",         "generator")
    graph.add_edge("generator",        "citation_builder")
    graph.add_edge("citation_builder", END)

    return graph


pipeline = build_graph().compile()
