"""T7 acceptance tests — ASR node and voice endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pipeline.state import PipelineState


def _base_state(**kwargs) -> PipelineState:
    defaults = dict(query="", lang_hint="en", latency_map={}, self_rag_retries=0)
    defaults.update(kwargs)
    return PipelineState(**defaults)


# ── ASR node tests ─────────────────────────────────────────────────────────
class TestAsrNode:
    def test_text_path_passes_query_through(self):
        from pipeline.nodes.asr import asr_node

        state = _base_state(query="What is MSME?", lang_hint="en")
        result = asr_node(state)
        assert result["transcript"] == "What is MSME?"
        assert result["lang"] == "en"

    def test_text_path_zero_latency(self):
        from pipeline.nodes.asr import asr_node

        state = _base_state(query="test", lang_hint="en")
        result = asr_node(state)
        assert result["latency_map"]["asr"] == 0.0

    def test_hindi_hint_sets_lang(self):
        from pipeline.nodes.asr import asr_node

        state = _base_state(query="MSME kya hai?", lang_hint="hi")
        result = asr_node(state)
        assert result["lang"] == "hi"

    def test_audio_path_uses_whisper(self):
        from pipeline.nodes.asr import asr_node

        state = _base_state(audio_bytes=b"fake_audio_data", lang_hint="en")
        with (
            patch(
                "pipeline.nodes.asr._transcribe_whisper", return_value="MSME loan query"
            ) as mock_w,
            patch("pipeline.nodes.asr._transcribe_saaras") as mock_s,
        ):
            result = asr_node(state)
        mock_w.assert_called_once()
        mock_s.assert_not_called()
        assert result["transcript"] == "MSME loan query"

    def test_hindi_audio_uses_saaras_when_key_set(self):
        from pipeline.nodes.asr import asr_node

        state = _base_state(audio_bytes=b"hindi_audio", lang_hint="hi")
        with (
            patch("pipeline.nodes.asr.settings") as mock_settings,
            patch("pipeline.nodes.asr._transcribe_saaras", return_value="MSME प्रश्न") as mock_s,
            patch("pipeline.nodes.asr._transcribe_whisper"),
        ):
            mock_settings.SARVAM_API_KEY = "test-key"
            mock_settings.WHISPER_MODEL = "medium"
            asr_node(state)
        mock_s.assert_called_once()

    def test_saaras_failure_falls_back_to_whisper(self):
        from pipeline.nodes.asr import asr_node

        state = _base_state(audio_bytes=b"audio", lang_hint="hi")
        with (
            patch("pipeline.nodes.asr.settings") as mock_settings,
            patch("pipeline.nodes.asr._transcribe_saaras", side_effect=Exception("timeout")),
            patch("pipeline.nodes.asr._transcribe_whisper", return_value="fallback text"),
        ):
            mock_settings.SARVAM_API_KEY = "test-key"
            mock_settings.WHISPER_MODEL = "medium"
            result = asr_node(state)
        assert result["transcript"] == "fallback text"

    def test_whisper_failure_returns_empty_transcript(self):
        from pipeline.nodes.asr import asr_node

        state = _base_state(audio_bytes=b"bad_audio", lang_hint="en")
        with patch("pipeline.nodes.asr._transcribe_whisper", side_effect=Exception("model error")):
            result = asr_node(state)
        assert result["transcript"] == ""

    def test_asr_logs_latency_for_audio(self):
        from pipeline.nodes.asr import asr_node

        state = _base_state(audio_bytes=b"audio", lang_hint="en")
        with patch("pipeline.nodes.asr._transcribe_whisper", return_value="transcript"):
            result = asr_node(state)
        assert result["latency_map"]["asr"] >= 0

    def test_lang_detection_fallback(self):
        from pipeline.nodes.asr import _detect_language

        assert _detect_language("") == "en"

    def test_unknown_lang_hint_triggers_detection(self):
        from pipeline.nodes.asr import asr_node

        state = _base_state(query="What is MSME?", lang_hint="fr")
        result = asr_node(state)
        assert result["lang"] in ("hi", "en")


# ── Voice endpoint tests ───────────────────────────────────────────────────
class TestVoiceEndpoint:
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
            "answer": "MSME loans are for small enterprises.",
            "lang": "en",
            "route": "simple",
            "crag_action": "CORRECT",
            "crag_score": 0.85,
            "confidence": 0.82,
            "self_rag_retries": 0,
            "transcript": "What is MSME loan?",
            "sources": [],
            "latency_map": {"total": 1000.0},
            "query_id": "test-id",
        }

    def test_voice_endpoint_returns_200(self, client):
        audio_bytes = b"RIFF$\x00\x00\x00WAVEfmt "  # minimal WAV header
        with patch("pipeline.graph.pipeline") as mock_pipe:
            mock_pipe.invoke.return_value = self._mock_pipeline_result()
            resp = client.post(
                "/query/voice",
                files={"audio": ("test.wav", audio_bytes, "audio/wav")},
            )
        assert resp.status_code == 200

    def test_voice_endpoint_streams_sse(self, client):
        with patch("pipeline.graph.pipeline") as mock_pipe:
            mock_pipe.invoke.return_value = self._mock_pipeline_result()
            resp = client.post(
                "/query/voice",
                files={"audio": ("test.wav", b"fake_wav_data", "audio/wav")},
            )
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_empty_audio_returns_400(self, client):
        resp = client.post(
            "/query/voice",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        assert resp.status_code == 400

    def test_voice_with_lang_hint(self, client):
        with patch("pipeline.graph.pipeline") as mock_pipe:
            mock_pipe.invoke.return_value = self._mock_pipeline_result()
            resp = client.post(
                "/query/voice",
                files={"audio": ("test.wav", b"audio_data", "audio/wav")},
                data={"lang": "hi"},
            )
        assert resp.status_code == 200
        state_passed = mock_pipe.invoke.call_args[0][0]
        assert state_passed.get("lang_hint") == "hi"
