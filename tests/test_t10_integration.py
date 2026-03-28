"""T10 integration tests — all modules import, result files exist, contracts valid."""
from __future__ import annotations
import json
from pathlib import Path
import pytest


class TestAllModulesImport:
    def test_app_imports(self):
        from app.main import create_app
        from app.config import settings
        from app.schemas import HealthResponse, TextQueryRequest
        assert True

    def test_pipeline_imports(self):
        from pipeline.state import PipelineState
        from pipeline.graph import build_graph, pipeline
        assert pipeline is not None

    def test_all_nodes_import(self):
        from pipeline.nodes.asr import asr_node
        from pipeline.nodes.classifier import classifier_node
        from pipeline.nodes.hyde import hyde_node
        from pipeline.nodes.retriever import retriever_node
        from pipeline.nodes.crag_evaluator import crag_evaluator_node
        from pipeline.nodes.self_rag import self_rag_node
        from pipeline.nodes.generator import generator_node
        from pipeline.nodes.citation_builder import citation_builder_node
        assert True

    def test_storage_imports(self):
        from storage.chroma_client import get_chroma_client, COLLECTION_NAME
        from storage.query_log import init_query_log, write_query_record
        assert True

    def test_ingestion_imports(self):
        from ingestion.loader import load_pdf, RawPage
        from ingestion.chunker import chunk_pages, CHUNK_SIZE
        from ingestion.embedder import embed_and_upsert, E5_PASSAGE_PREFIX
        assert True

    def test_evaluation_imports(self):
        from evaluation.eval_runner import run_evaluation, _load_queries
        assert True


class TestResultFilesExist:
    def test_selfcrag_results_exist(self):
        assert Path("evaluation/results/selfcrag.json").exists(), \
            "Run: python -m evaluation.eval_runner --lang en"

    def test_baseline_results_exist(self):
        assert Path("evaluation/results/baseline_naive.json").exists(), \
            "Run: python -m evaluation.eval_runner --baseline --lang en"

    def test_selfcrag_results_valid(self):
        data = json.loads(Path("evaluation/results/selfcrag.json").read_text())
        assert "aggregate" in data
        assert data["total_queries"] > 0

    def test_results_have_ragas_scores(self):
        data = json.loads(Path("evaluation/results/selfcrag.json").read_text())
        for m in ("faithfulness", "answer_relevancy", "context_precision"):
            assert m in data["aggregate"]


class TestStateContract:
    def test_all_node_fields_in_state(self):
        from pipeline.state import PipelineState
        import typing
        hints = typing.get_type_hints(PipelineState)
        for field in ("query", "transcript", "lang", "route", "chunks",
                      "crag_action", "answer", "sources", "query_id", "latency_map"):
            assert field in hints, f"Missing: {field}"


class TestInfraFilesPresent:
    def test_dockerfile(self):
        assert Path("Dockerfile").exists()

    def test_docker_compose(self):
        assert Path("docker-compose.yml").exists()

    def test_env_example(self):
        assert Path(".env.example").exists()

    def test_pyproject(self):
        assert Path("pyproject.toml").exists()

    def test_ci_workflow(self):
        assert Path(".github/workflows/ci.yml").exists()

    def test_readme(self):
        readme = Path("README.md").read_text()
        assert "VoiceRAG" in readme
        assert "Self-CRAG" in readme
        assert "curl" in readme
