"""T6 acceptance tests — generator, citation builder, SSE endpoint."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pipeline.state import PipelineState, SourceChunk


def _make_chunk(n: int = 0) -> SourceChunk:
    return SourceChunk(
        chunk_id=f"c{n}",
        doc_title="MSME Guide",
        page_num=n + 1,
        chunk_text="MSME loans are available for small enterprises.",
        relevance_score=0.85,
    )


def _base_state(**kwargs) -> PipelineState:
    defaults = dict(
        query="What is MSME loan?",
        transcript="What is MSME loan?",
        lang="en",
        lang_hint="en",
        route="moderate",
        latency_map={},
        self_rag_retries=0,
        crag_action="CORRECT",
        crag_score=0.85,
        chunks=[_make_chunk()],
    )
    defaults.update(kwargs)
    return PipelineState(**defaults)


def _mock_groq(content: str):
    choice = MagicMock()
    choice.message.content = content
    completion = MagicMock()
    completion.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    return client


# ── Generator tests ────────────────────────────────────────────────────────
class TestGeneratorNode:
    def test_generates_answer_from_chunks(self):
        from pipeline.nodes.generator import generator_node

        state = _base_state()
        with patch(
            "pipeline.nodes.generator.Groq",
            return_value=_mock_groq("MSME loans [1] are for small businesses."),
        ):
            result = generator_node(state)
        assert "MSME" in result["answer"]

    def test_abstain_on_incorrect_crag(self):
        from pipeline.nodes.generator import ABSTAIN_RESPONSE_EN, generator_node

        state = _base_state(crag_action="INCORRECT", chunks=[])
        with patch("pipeline.nodes.generator.Groq") as mock:
            result = generator_node(state)
        mock.assert_not_called()
        assert result["answer"] == ABSTAIN_RESPONSE_EN
        assert result["confidence"] == 0.0

    def test_abstain_hindi_on_incorrect_hindi(self):
        from pipeline.nodes.generator import ABSTAIN_RESPONSE_HI, generator_node

        state = _base_state(crag_action="INCORRECT", lang="hi", chunks=[])
        with patch("pipeline.nodes.generator.Groq"):
            result = generator_node(state)
        assert result["answer"] == ABSTAIN_RESPONSE_HI

    def test_uses_existing_self_rag_answer(self):
        from pipeline.nodes.generator import generator_node

        state = _base_state(answer="Pre-existing Self-RAG answer.")
        with patch("pipeline.nodes.generator.Groq") as mock:
            result = generator_node(state)
        mock.assert_not_called()
        assert result["answer"] == "Pre-existing Self-RAG answer."

    def test_confidence_computed(self):
        from pipeline.nodes.generator import generator_node

        state = _base_state()
        with patch("pipeline.nodes.generator.Groq", return_value=_mock_groq("Answer [1].")):
            result = generator_node(state)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_sources_populated(self):
        from pipeline.nodes.generator import generator_node

        state = _base_state()
        with patch("pipeline.nodes.generator.Groq", return_value=_mock_groq("Answer [1].")):
            result = generator_node(state)
        assert len(result["sources"]) > 0

    def test_latency_logged(self):
        from pipeline.nodes.generator import generator_node

        state = _base_state()
        with patch("pipeline.nodes.generator.Groq", return_value=_mock_groq("Answer.")):
            result = generator_node(state)
        assert "generation" in result["latency_map"]

    def test_groq_error_returns_abstain(self):
        from pipeline.nodes.generator import ABSTAIN_RESPONSE_EN, generator_node

        state = _base_state()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("timeout")
        with patch("pipeline.nodes.generator.Groq", return_value=mock_client):
            result = generator_node(state)
        assert result["answer"] == ABSTAIN_RESPONSE_EN

    def test_confidence_lower_with_retry(self):
        from pipeline.nodes.generator import _compute_confidence

        c0 = _compute_confidence(0.8, "CORRECT", 0)
        c1 = _compute_confidence(0.8, "CORRECT", 1)
        assert c0 > c1


# ── Citation builder tests ─────────────────────────────────────────────────
class TestCitationBuilder:
    def test_query_id_set(self):
        from pipeline.nodes.citation_builder import citation_builder_node

        state = _base_state(answer="Answer.", sources=[_make_chunk()])
        result = citation_builder_node(state)
        assert "query_id" in result
        assert len(result["query_id"]) > 0

    def test_total_latency_computed(self):
        from pipeline.nodes.citation_builder import citation_builder_node

        state = _base_state(
            answer="Answer.",
            sources=[],
            latency_map={"asr": 100.0, "retrieval": 200.0, "generation": 300.0},
        )
        result = citation_builder_node(state)
        assert result["latency_map"]["total"] == 600.0

    def test_existing_query_id_preserved(self):
        from pipeline.nodes.citation_builder import citation_builder_node

        state = _base_state(answer="A.", sources=[], query_id="my-fixed-id")
        result = citation_builder_node(state)
        assert result["query_id"] == "my-fixed-id"


# ── SSE endpoint tests ─────────────────────────────────────────────────────
class TestQueryTextEndpoint:
    @pytest.fixture(autouse=True)
    def _env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))
        monkeypatch.setenv("QUERY_LOG_PATH", str(tmp_path / "logs" / "q.db"))
        monkeypatch.setenv("CORPUS_PATH", str(tmp_path / "corpus"))
        monkeypatch.setenv("APP_ENV", "development")
        (tmp_path / "logs").mkdir()
        (tmp_path / "corpus").mkdir()

    @pytest.fixture
    def client(self, tmp_path):
        mock_chroma = MagicMock()
        mock_chroma.list_collections.return_value = []
        with (
            patch("storage.chroma_client._client", mock_chroma),
            patch("storage.chroma_client.get_chroma_client", return_value=mock_chroma),
        ):
            from app.main import app

            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    def _mock_pipeline_result(self):
        return {
            "answer": "MSME loans are available for small enterprises [1].",
            "lang": "en",
            "route": "moderate",
            "crag_action": "CORRECT",
            "crag_score": 0.85,
            "confidence": 0.82,
            "self_rag_retries": 0,
            "sources": [_make_chunk()],
            "latency_map": {
                "classifier": 120.0,
                "retrieval": 300.0,
                "generation": 800.0,
                "total": 1220.0,
            },
            "query_id": "test-query-id",
        }

    def test_text_endpoint_returns_200(self, client):
        with patch("pipeline.graph.pipeline") as mock_pipe:
            mock_pipe.invoke.return_value = self._mock_pipeline_result()
            resp = client.post("/query/text", json={"query": "What is MSME?"})
        assert resp.status_code == 200

    def test_text_endpoint_streams_sse(self, client):
        with patch("pipeline.graph.pipeline") as mock_pipe:
            mock_pipe.invoke.return_value = self._mock_pipeline_result()
            resp = client.post("/query/text", json={"query": "What is MSME?"})
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_sse_contains_done_event(self, client):
        with patch("pipeline.graph.pipeline") as mock_pipe:
            mock_pipe.invoke.return_value = self._mock_pipeline_result()
            resp = client.post("/query/text", json={"query": "What is MSME?"})
        events = [line for line in resp.text.split("\n") if line.startswith("data:")]
        done_events = [e for e in events if '"type": "done"' in e]
        assert len(done_events) == 1

    def test_done_payload_has_required_fields(self, client):
        with patch("pipeline.graph.pipeline") as mock_pipe:
            mock_pipe.invoke.return_value = self._mock_pipeline_result()
            resp = client.post("/query/text", json={"query": "What is MSME?"})
        events = [line for line in resp.text.split("\n") if line.startswith("data:")]
        done_raw = next(e for e in events if '"type": "done"' in e)
        payload = json.loads(done_raw[5:])["payload"]
        for field in ("query_id", "answer", "sources", "confidence", "routing_path", "latency_ms"):
            assert field in payload, f"Missing field: {field}"

    def test_query_id_in_response_header(self, client):
        with patch("pipeline.graph.pipeline") as mock_pipe:
            mock_pipe.invoke.return_value = self._mock_pipeline_result()
            resp = client.post("/query/text", json={"query": "What is MSME?"})
        assert "x-query-id" in resp.headers
