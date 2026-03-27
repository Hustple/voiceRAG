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
