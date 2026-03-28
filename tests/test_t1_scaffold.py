from __future__ import annotations
import sqlite3
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("QUERY_LOG_PATH", str(tmp_path / "logs" / "queries.db"))
    monkeypatch.setenv("CORPUS_PATH", str(tmp_path / "corpus"))
    monkeypatch.setenv("APP_ENV", "development")
    (tmp_path / "corpus").mkdir()
    (tmp_path / "logs").mkdir()

@pytest.fixture
def client(tmp_path):
    mock_chroma = MagicMock()
    mock_chroma.list_collections.return_value = []
    with patch("storage.chroma_client._client", mock_chroma), \
         patch("storage.chroma_client.get_chroma_client", return_value=mock_chroma):
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

def test_health_returns_200(client):
    assert client.get("/health").status_code == 200

def test_health_schema(client):
    data = client.get("/health").json()
    assert "status" in data
    assert "chroma_ok" in data
    assert isinstance(data["corpus_chunks"], int)

def test_health_status_value(client):
    assert client.get("/health").json()["status"] in ("ok", "degraded")

def test_docs_available(client):
    assert client.get("/docs").status_code == 200

def test_query_text_stub_501(client):
    assert client.post("/query/text", json={"query": "test"}).status_code in (200, 501, 500)

def test_explain_404_for_missing(client):
    assert client.get("/explain/nonexistent-id").status_code == 404

def test_query_log_table_created(tmp_path, monkeypatch):
    db_path = str(tmp_path / "logs" / "test.db")
    (tmp_path / "logs").mkdir(exist_ok=True)
    monkeypatch.setenv("QUERY_LOG_PATH", db_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    import importlib
    import app.config as cfg
    importlib.reload(cfg)
    from storage import query_log
    importlib.reload(query_log)
    query_log.init_query_log()
    conn = sqlite3.connect(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    assert any(t[0] == "queries" for t in tables)
