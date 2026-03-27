#!/bin/bash
# VoiceRAG T4 — Query classifier + HyDE nodes
# Run from inside voicerag/: bash ../voicerag_t4_setup.sh

set -e
echo "==> Creating T4 files..."

# ── pipeline/nodes/classifier.py ───────────────────────────────────────────
cat > pipeline/nodes/classifier.py << 'CLASSIFIER'
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
CLASSIFIER

# ── pipeline/nodes/hyde.py ─────────────────────────────────────────────────
cat > pipeline/nodes/hyde.py << 'HYDE'
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
HYDE

# ── Update pipeline/graph.py — real routing after classifier ───────────────
cat > pipeline/graph.py << 'GRAPH'
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
GRAPH

# ── tests/test_t4_classifier_hyde.py ──────────────────────────────────────
cat > tests/test_t4_classifier_hyde.py << 'TESTS'
"""
T4 acceptance tests — classifier and HyDE nodes.
All Groq API calls are mocked — no real API key needed in tests.
"""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from pipeline.state import PipelineState


def _base_state(query: str = "What is MSME?", lang: str = "en") -> PipelineState:
    return PipelineState(
        query=query,
        transcript=query,
        lang=lang,
        lang_hint=lang,
        latency_map={},
        self_rag_retries=0,
    )


def _mock_groq(content: str):
    """Return a mock Groq client whose completion returns `content`."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


# ── Classifier tests ───────────────────────────────────────────────────────
class TestClassifierNode:
    def test_simple_query_classified_correctly(self):
        from pipeline.nodes.classifier import classifier_node
        state = _base_state("What is Mudra Yojana?")
        with patch("pipeline.nodes.classifier.Groq", return_value=_mock_groq("simple")):
            result = classifier_node(state)
        assert result["route"] == "simple"

    def test_moderate_query_classified_correctly(self):
        from pipeline.nodes.classifier import classifier_node
        state = _base_state("How do I apply for MSME registration?")
        with patch("pipeline.nodes.classifier.Groq", return_value=_mock_groq("moderate")):
            result = classifier_node(state)
        assert result["route"] == "moderate"

    def test_complex_query_classified_correctly(self):
        from pipeline.nodes.classifier import classifier_node
        state = _base_state("Compare MSME and Startup India eligibility criteria")
        with patch("pipeline.nodes.classifier.Groq", return_value=_mock_groq("complex")):
            result = classifier_node(state)
        assert result["route"] == "complex"

    def test_invalid_llm_response_falls_back_to_moderate(self):
        from pipeline.nodes.classifier import classifier_node
        state = _base_state("Some query")
        with patch("pipeline.nodes.classifier.Groq", return_value=_mock_groq("I cannot determine")):
            result = classifier_node(state)
        assert result["route"] == "moderate"

    def test_groq_exception_falls_back_to_moderate(self):
        from pipeline.nodes.classifier import classifier_node
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        state = _base_state("Some query")
        with patch("pipeline.nodes.classifier.Groq", return_value=mock_client):
            result = classifier_node(state)
        assert result["route"] == "moderate"

    def test_classifier_logs_latency(self):
        from pipeline.nodes.classifier import classifier_node
        state = _base_state("What is GST?")
        with patch("pipeline.nodes.classifier.Groq", return_value=_mock_groq("simple")):
            result = classifier_node(state)
        assert "classifier" in result["latency_map"]
        assert result["latency_map"]["classifier"] >= 0

    def test_route_is_always_valid(self):
        from pipeline.nodes.classifier import classifier_node, VALID_ROUTES
        for response in ["simple", "SIMPLE", "simple.", "moderate\n", "complex "]:
            state = _base_state("test")
            with patch("pipeline.nodes.classifier.Groq", return_value=_mock_groq(response)):
                result = classifier_node(state)
            assert result["route"] in VALID_ROUTES

    def test_uses_transcript_over_query(self):
        from pipeline.nodes.classifier import classifier_node
        state = _base_state("original query")
        state["transcript"] = "transcribed version"
        captured = {}
        def fake_create(**kwargs):
            captured["messages"] = kwargs["messages"]
            return _mock_groq("simple").chat.completions.create(**kwargs)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create
        mock_client.chat.completions.create.return_value = _mock_groq("simple").chat.completions.create()
        with patch("pipeline.nodes.classifier.Groq", return_value=_mock_groq("simple")):
            result = classifier_node(state)
        assert result["route"] in ("simple", "moderate", "complex")


# ── HyDE tests ─────────────────────────────────────────────────────────────
class TestHydeNode:
    def test_hyde_skipped_for_simple_route(self):
        from pipeline.nodes.hyde import hyde_node
        state = _base_state("What is Mudra?")
        state["route"] = "simple"
        with patch("pipeline.nodes.hyde.Groq") as mock_groq_cls:
            result = hyde_node(state)
        mock_groq_cls.assert_not_called()
        assert result["hyde_query"] is None

    def test_hyde_skipped_for_moderate_route(self):
        from pipeline.nodes.hyde import hyde_node
        state = _base_state("How to apply for MSME?")
        state["route"] = "moderate"
        with patch("pipeline.nodes.hyde.Groq") as mock_groq_cls:
            result = hyde_node(state)
        mock_groq_cls.assert_not_called()
        assert result["hyde_query"] is None

    def test_hyde_runs_for_complex_route(self):
        from pipeline.nodes.hyde import hyde_node
        state = _base_state("Compare all Indian MSME schemes")
        state["route"] = "complex"
        hypothesis = "MSME schemes include Mudra, CGTMSE, and Credit Linked Capital Subsidy."
        with patch("pipeline.nodes.hyde.Groq", return_value=_mock_groq(hypothesis)):
            result = hyde_node(state)
        assert result["hyde_query"] == hypothesis

    def test_hyde_fallback_on_api_error(self):
        from pipeline.nodes.hyde import hyde_node
        state = _base_state("Complex query")
        state["route"] = "complex"
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("timeout")
        with patch("pipeline.nodes.hyde.Groq", return_value=mock_client):
            result = hyde_node(state)
        assert result["hyde_query"] is None

    def test_hyde_logs_latency(self):
        from pipeline.nodes.hyde import hyde_node
        state = _base_state("Complex query")
        state["route"] = "complex"
        with patch("pipeline.nodes.hyde.Groq", return_value=_mock_groq("A hypothesis.")):
            result = hyde_node(state)
        assert "hyde" in result["latency_map"]

    def test_hyde_latency_zero_when_skipped(self):
        from pipeline.nodes.hyde import hyde_node
        state = _base_state("Simple query")
        state["route"] = "simple"
        with patch("pipeline.nodes.hyde.Groq"):
            result = hyde_node(state)
        assert result["latency_map"]["hyde"] == 0.0


# ── Graph routing tests ────────────────────────────────────────────────────
class TestGraphRouting:
    def test_complex_routes_through_hyde(self):
        from pipeline.graph import _route_after_classifier
        state = _base_state()
        state["route"] = "complex"
        assert _route_after_classifier(state) == "hyde"

    def test_simple_skips_hyde(self):
        from pipeline.graph import _route_after_classifier
        state = _base_state()
        state["route"] = "simple"
        assert _route_after_classifier(state) == "retriever"

    def test_moderate_skips_hyde(self):
        from pipeline.graph import _route_after_classifier
        state = _base_state()
        state["route"] = "moderate"
        assert _route_after_classifier(state) == "retriever"
TESTS

echo ""
echo "==> T4 files created."
echo ""
echo "Run tests (no new deps needed):"
echo ""
echo "  python3.11 -m pytest tests/test_t4_classifier_hyde.py -v --no-cov"
