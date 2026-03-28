"""
T8 acceptance tests — query log, GET /explain, GET /health.
Uses a real SQLite in-memory path so no mocking needed for storage.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Query log unit tests ───────────────────────────────────────────────────
class TestQueryLog:
    @pytest.fixture(autouse=True)
    def _db(self, tmp_path, monkeypatch):
        db = str(tmp_path / "test.db")
        monkeypatch.setenv("QUERY_LOG_PATH", db)
        monkeypatch.setenv("GROQ_API_KEY", "test")
        import importlib

        import app.config as cfg
        import storage.query_log as ql

        importlib.reload(cfg)
        importlib.reload(ql)
        ql.init_query_log()
        self.ql = ql

    def _record(self, **kwargs) -> dict:
        defaults = dict(
            query_id="qid-001",
            transcript="What is MSME?",
            lang="en",
            routing_path="simple",
            crag_action="CORRECT",
            self_rag_retries=0,
            confidence=0.85,
            latency_ms={"asr": 0.0, "retrieval": 200.0, "generation": 800.0, "total": 1000.0},
            sources=[
                {
                    "chunk_id": "c1",
                    "doc_title": "MSME Guide",
                    "page_num": 1,
                    "chunk_text": "MSME text.",
                    "relevance_score": 0.9,
                }
            ],
            ragas={},
        )
        defaults.update(kwargs)
        return defaults

    def test_write_and_read_record(self):
        rec = self._record()
        self.ql.write_query_record(rec)
        result = self.ql.get_query_record("qid-001")
        assert result is not None
        assert result["transcript"] == "What is MSME?"
        assert result["routing_path"] == "simple"
        assert result["confidence"] == 0.85

    def test_latency_map_deserialised(self):
        self.ql.write_query_record(self._record())
        result = self.ql.get_query_record("qid-001")
        assert isinstance(result["latency_ms"], dict)
        assert result["latency_ms"]["total"] == 1000.0

    def test_sources_deserialised(self):
        self.ql.write_query_record(self._record())
        result = self.ql.get_query_record("qid-001")
        assert isinstance(result["sources"], list)
        assert result["sources"][0]["doc_title"] == "MSME Guide"

    def test_missing_query_returns_none(self):
        assert self.ql.get_query_record("nonexistent") is None

    def test_upsert_overwrites_existing(self):
        self.ql.write_query_record(self._record(confidence=0.5))
        self.ql.write_query_record(self._record(confidence=0.9))
        result = self.ql.get_query_record("qid-001")
        assert result["confidence"] == 0.9

    def test_health_metrics_empty_db(self):
        metrics = self.ql.get_health_metrics()
        assert metrics["total_queries"] == 0
        assert metrics["avg_latency_ms"] is None
        assert metrics["routing_distribution"] == {}

    def test_health_metrics_with_data(self):
        for i in range(3):
            self.ql.write_query_record(
                self._record(
                    query_id=f"qid-{i:03d}",
                    routing_path="simple" if i < 2 else "moderate",
                    confidence=0.8,
                )
            )
        metrics = self.ql.get_health_metrics()
        assert metrics["total_queries"] == 3
        assert metrics["avg_confidence"] == pytest.approx(0.8, rel=0.01)
        assert metrics["routing_distribution"]["simple"] == 2
        assert metrics["routing_distribution"]["moderate"] == 1

    def test_health_metrics_node_latency(self):
        self.ql.write_query_record(
            self._record(latency_ms={"retrieval": 200.0, "generation": 800.0, "total": 1000.0})
        )
        metrics = self.ql.get_health_metrics()
        assert "retrieval" in metrics["node_latency_avg_ms"]
        assert metrics["node_latency_avg_ms"]["retrieval"] == pytest.approx(200.0)

    def test_health_metrics_caps_at_100(self):
        for i in range(120):
            self.ql.write_query_record(
                self._record(
                    query_id=f"qid-{i:03d}",
                    confidence=0.5,
                )
            )
        metrics = self.ql.get_health_metrics()
        assert metrics["total_queries"] == 100


# ── API endpoint tests ─────────────────────────────────────────────────────
class TestObservabilityEndpoints:
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

    def _seed_record(self, tmp_path, monkeypatch):
        """Write a record directly to the log for endpoint tests."""
        import importlib

        import app.config as cfg
        import storage.query_log as ql

        importlib.reload(cfg)
        importlib.reload(ql)
        ql.init_query_log()
        ql.write_query_record(
            {
                "query_id": "explain-test-id",
                "transcript": "What is MSME?",
                "lang": "en",
                "routing_path": "simple",
                "crag_action": "CORRECT",
                "self_rag_retries": 0,
                "confidence": 0.85,
                "latency_ms": {
                    "asr": 0.0,
                    "retrieval": 200.0,
                    "generation": 800.0,
                    "total": 1000.0,
                },
                "sources": [],
                "ragas": {},
            }
        )

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_schema_fields(self, client):
        data = client.get("/health").json()
        for field in (
            "status",
            "chroma_ok",
            "corpus_chunks",
            "total_queries",
            "routing_distribution",
            "node_latency_avg_ms",
        ):
            assert field in data, f"Missing: {field}"

    def test_health_status_ok_or_degraded(self, client):
        assert client.get("/health").json()["status"] in ("ok", "degraded")

    def test_health_total_queries_is_int(self, client):
        assert isinstance(client.get("/health").json()["total_queries"], int)

    def test_explain_404_for_missing(self, client):
        assert client.get("/explain/nonexistent-id").status_code == 404

    def test_explain_returns_record(self, client, tmp_path, monkeypatch):
        self._seed_record(tmp_path, monkeypatch)
        resp = client.get("/explain/explain-test-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["transcript"] == "What is MSME?"
        assert data["routing_path"] == "simple"
        assert data["confidence"] == 0.85

    def test_explain_latency_map_is_dict(self, client, tmp_path, monkeypatch):
        self._seed_record(tmp_path, monkeypatch)
        data = client.get("/explain/explain-test-id").json()
        assert isinstance(data["latency_ms"], dict)
        assert data["latency_ms"]["total"] == 1000.0

    def test_health_reflects_logged_queries(self, client, tmp_path, monkeypatch):
        self._seed_record(tmp_path, monkeypatch)
        data = client.get("/health").json()
        assert data["total_queries"] >= 1
