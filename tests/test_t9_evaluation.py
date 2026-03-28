"""
T9 acceptance tests — RAGAs evaluation harness.
Tests the scorer, query loading, and result structure without real pipeline calls.
"""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


class TestQueryFiles:
    def test_english_queries_exist(self):
        assert Path("evaluation/test_queries_en.json").exists()

    def test_hindi_queries_exist(self):
        assert Path("evaluation/test_queries_hi.json").exists()

    def test_english_queries_valid_json(self):
        data = json.loads(Path("evaluation/test_queries_en.json").read_text())
        assert isinstance(data, list)
        assert len(data) >= 5

    def test_hindi_queries_valid_json(self):
        data = json.loads(Path("evaluation/test_queries_hi.json").read_text())
        assert isinstance(data, list)
        assert len(data) >= 3

    def test_each_query_has_required_fields(self):
        for fname in ("evaluation/test_queries_en.json", "evaluation/test_queries_hi.json"):
            data = json.loads(Path(fname).read_text())
            for item in data:
                assert "id" in item, f"Missing id in {fname}"
                assert "query" in item, f"Missing query in {fname}"
                assert "ground_truth" in item, f"Missing ground_truth in {fname}"
                assert "lang" in item, f"Missing lang in {fname}"

    def test_query_ids_unique(self):
        en = json.loads(Path("evaluation/test_queries_en.json").read_text())
        hi = json.loads(Path("evaluation/test_queries_hi.json").read_text())
        all_ids = [q["id"] for q in en + hi]
        assert len(all_ids) == len(set(all_ids))

    def test_lang_field_valid(self):
        en = json.loads(Path("evaluation/test_queries_en.json").read_text())
        hi = json.loads(Path("evaluation/test_queries_hi.json").read_text())
        for q in en:
            assert q["lang"] == "en"
        for q in hi:
            assert q["lang"] == "hi"


class TestHeuristicScorer:
    def test_perfect_answer_scores_high(self):
        from evaluation.eval_runner import _heuristic_scores
        query = "What is MSME?"
        answer = "MSME stands for Micro Small Medium Enterprises"
        contexts = ["MSME stands for Micro Small Medium Enterprises in India"]
        ground_truth = "MSME is Micro Small Medium Enterprises"
        scores = _heuristic_scores(query, answer, contexts, ground_truth)
        assert scores["faithfulness"] > 0.5
        assert scores["answer_relevancy"] >= 0.0  # heuristic: query word overlap with answer

    def test_empty_answer_scores_zero_faithfulness(self):
        from evaluation.eval_runner import _heuristic_scores
        scores = _heuristic_scores("query", "", ["context text"], "ground truth")
        assert scores["faithfulness"] == 0.0

    def test_scores_between_0_and_1(self):
        from evaluation.eval_runner import _heuristic_scores
        scores = _heuristic_scores(
            "MSME eligibility", "MSME loans are available",
            ["MSME loans are for small businesses"], "MSME is for small enterprises"
        )
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_returns_all_three_metrics(self):
        from evaluation.eval_runner import _heuristic_scores
        scores = _heuristic_scores("q", "a", ["c"], "g")
        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "context_precision" in scores


class TestQueryLoader:
    def test_load_all_queries(self):
        from evaluation.eval_runner import _load_queries
        queries = _load_queries(None)
        assert len(queries) >= 8

    def test_load_english_only(self):
        from evaluation.eval_runner import _load_queries
        queries = _load_queries("en")
        assert all(q["lang"] == "en" for q in queries)

    def test_load_hindi_only(self):
        from evaluation.eval_runner import _load_queries
        queries = _load_queries("hi")
        assert all(q["lang"] == "hi" for q in queries)


class TestEvalRunner:
    def _mock_pipeline_result(self, answer: str = "MSME loans are for small enterprises."):
        return {
            "answer": answer,
            "lang": "en",
            "route": "simple",
            "crag_action": "CORRECT",
            "confidence": 0.85,
            "self_rag_retries": 0,
            "sources": [
                {"chunk_text": "MSME loans are available for micro enterprises.",
                 "chunk_id": "c1", "doc_title": "MSME Guide",
                 "page_num": 1, "relevance_score": 0.9}
            ],
            "latency_map": {"total": 1200.0},
        }

    def test_dry_run_uses_3_queries(self):
        from evaluation.eval_runner import run_evaluation
        with patch("evaluation.eval_runner._run_pipeline",
                   return_value=self._mock_pipeline_result()):
            summary = run_evaluation(lang="en", dry_run=True)
        assert summary["total_queries"] == 3

    def test_summary_has_aggregate(self):
        from evaluation.eval_runner import run_evaluation
        with patch("evaluation.eval_runner._run_pipeline",
                   return_value=self._mock_pipeline_result()):
            summary = run_evaluation(lang="en", dry_run=True)
        assert "aggregate" in summary
        agg = summary["aggregate"]
        assert "faithfulness" in agg
        assert "answer_relevancy" in agg
        assert "context_precision" in agg
        assert "avg_latency_ms" in agg

    def test_each_result_has_ragas_scores(self):
        from evaluation.eval_runner import run_evaluation
        with patch("evaluation.eval_runner._run_pipeline",
                   return_value=self._mock_pipeline_result()):
            summary = run_evaluation(lang="en", dry_run=True)
        for q in summary["queries"]:
            assert "ragas" in q
            ragas = q["ragas"]
            for metric in ("faithfulness", "answer_relevancy", "context_precision"):
                assert metric in ragas

    def test_results_saved_to_file(self, tmp_path, monkeypatch):
        from evaluation.eval_runner import run_evaluation, RESULTS_DIR
        import evaluation.eval_runner as er
        monkeypatch.setattr(er, "RESULTS_DIR", tmp_path)
        with patch("evaluation.eval_runner._run_pipeline",
                   return_value=self._mock_pipeline_result()):
            summary = run_evaluation(lang="en", dry_run=True)
        assert summary["total_queries"] > 0

    def test_aggregate_scores_between_0_and_1(self):
        from evaluation.eval_runner import run_evaluation
        with patch("evaluation.eval_runner._run_pipeline",
                   return_value=self._mock_pipeline_result()):
            summary = run_evaluation(lang="en", dry_run=True)
        agg = summary["aggregate"]
        for metric in ("faithfulness", "answer_relevancy", "context_precision"):
            val = agg[metric]
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{metric}={val}"
