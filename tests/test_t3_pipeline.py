"""
T3 acceptance tests — LangGraph state machine + retriever node.

Retriever tests use a mocked ChromaDB so no real index is needed.
Graph compilation tests verify the wiring is correct.
"""
from __future__ import annotations
import uuid
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pipeline.state import PipelineState
from pipeline.nodes.stub_nodes import (
    asr_node, classifier_node, hyde_node,
    crag_evaluator_node, generator_node, citation_builder_node,
)


# ── Graph compilation ──────────────────────────────────────────────────────
class TestGraphCompilation:
    def test_graph_compiles_without_error(self):
        from pipeline.graph import build_graph
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_pipeline_import_succeeds(self):
        from pipeline.graph import pipeline
        assert pipeline is not None

    def test_all_nodes_registered(self):
        from pipeline.graph import build_graph
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        expected = {
            "asr", "classifier", "hyde", "retriever",
            "crag_evaluator", "self_rag", "generator", "citation_builder",
        }
        assert expected.issubset(node_names)


# ── State flows through stub nodes ─────────────────────────────────────────
class TestStubNodes:
    def _base_state(self) -> PipelineState:
        return PipelineState(
            query="What is MSME loan eligibility?",
            lang_hint="en",
            latency_map={},
            self_rag_retries=0,
        )

    def test_asr_stub_populates_transcript(self):
        state = self._base_state()
        result = asr_node(state)
        assert result["transcript"] == "What is MSME loan eligibility?"
        assert result["lang"] == "en"

    def test_classifier_stub_sets_route(self):
        state = self._base_state()
        state["transcript"] = "What is MSME?"
        result = classifier_node(state)
        assert result["route"] in ("simple", "moderate", "complex")

    def test_hyde_stub_sets_hyde_query(self):
        state = self._base_state()
        result = hyde_node(state)
        assert "hyde_query" in result

    def test_crag_stub_sets_action(self):
        state = self._base_state()
        state["chunks"] = []
        result = crag_evaluator_node(state)
        assert result["crag_action"] in ("CORRECT", "AMBIGUOUS", "INCORRECT")

    def test_generator_stub_sets_answer(self):
        state = self._base_state()
        state["transcript"] = "test query"
        state["chunks"] = []
        result = generator_node(state)
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_citation_builder_sets_query_id(self):
        state = self._base_state()
        state["answer"] = "some answer"
        state["chunks"] = []
        result = citation_builder_node(state)
        assert "query_id" in result
        assert len(result["query_id"]) > 0


# ── Retriever node (mocked ChromaDB) ───────────────────────────────────────
class TestRetrieverNode:
    def _mock_collection(self, n_results: int = 3):
        """Build a mock ChromaDB collection returning n_results fake chunks."""
        col = MagicMock()
        col.count.return_value = 100

        dim = 1024
        fake_embs = np.random.rand(n_results, dim).astype("float32")
        # Normalize so cosine sim works
        fake_embs = fake_embs / np.linalg.norm(fake_embs, axis=1, keepdims=True)

        col.query.return_value = {
            "ids": [[f"chunk-{i}" for i in range(n_results)]],
            "documents": [[f"MSME loan eligibility text chunk {i}." for i in range(n_results)]],
            "metadatas": [[
                {"doc_title": "MSME Registration", "page_num": i + 1,
                 "lang_hint": "en", "token_count": 50, "chunk_index": i}
                for i in range(n_results)
            ]],
            "embeddings": [fake_embs.tolist()],
            "distances": [[0.1 + i * 0.05 for i in range(n_results)]],
        }
        return col

    def _mock_model(self):
        model = MagicMock()
        dim = 1024
        emb = np.random.rand(1, dim).astype("float32")
        emb = emb / np.linalg.norm(emb)
        model.encode.return_value = emb
        return model

    def _base_state(self) -> PipelineState:
        return PipelineState(
            query="MSME loan eligibility",
            transcript="MSME loan eligibility",
            lang="en",
            lang_hint="en",
            route="simple",
            latency_map={},
            self_rag_retries=0,
        )

    def test_retriever_returns_chunks(self):
        from pipeline.nodes.retriever import retriever_node
        state = self._base_state()
        with patch("pipeline.nodes.retriever._get_model", return_value=self._mock_model()), \
             patch("pipeline.nodes.retriever.get_collection", return_value=self._mock_collection(3)):
            result = retriever_node(state)
        assert len(result["chunks"]) > 0

    def test_retriever_chunks_have_required_fields(self):
        from pipeline.nodes.retriever import retriever_node
        state = self._base_state()
        with patch("pipeline.nodes.retriever._get_model", return_value=self._mock_model()), \
             patch("pipeline.nodes.retriever.get_collection", return_value=self._mock_collection(3)):
            result = retriever_node(state)
        for chunk in result["chunks"]:
            assert "chunk_id" in chunk
            assert "doc_title" in chunk
            assert "page_num" in chunk
            assert "chunk_text" in chunk
            assert "relevance_score" in chunk

    def test_retriever_logs_latency(self):
        from pipeline.nodes.retriever import retriever_node
        state = self._base_state()
        with patch("pipeline.nodes.retriever._get_model", return_value=self._mock_model()), \
             patch("pipeline.nodes.retriever.get_collection", return_value=self._mock_collection(3)):
            result = retriever_node(state)
        assert "retrieval" in result["latency_map"]
        assert result["latency_map"]["retrieval"] > 0

    def test_retriever_uses_hyde_query_when_set(self):
        from pipeline.nodes.retriever import retriever_node
        state = self._base_state()
        state["hyde_query"] = "A hypothetical answer about MSME loans"
        mock_model = self._mock_model()
        with patch("pipeline.nodes.retriever._get_model", return_value=mock_model), \
             patch("pipeline.nodes.retriever.get_collection",
                   return_value=self._mock_collection(3)):
            retriever_node(state)
        encoded_text = mock_model.encode.call_args[0][0][0]
        assert "hypothetical" in encoded_text

    def test_retriever_handles_empty_collection(self):
        from pipeline.nodes.retriever import retriever_node
        state = self._base_state()
        col = MagicMock()
        col.count.return_value = 0
        col.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]],
            "embeddings": [[]], "distances": [[]],
        }
        with patch("pipeline.nodes.retriever._get_model", return_value=self._mock_model()), \
             patch("pipeline.nodes.retriever.get_collection", return_value=col):
            result = retriever_node(state)
        assert result["chunks"] == []

    def test_mmr_returns_diverse_chunks(self):
        """MMR diversity guarantee: two selected chunks must not be near-identical."""
        from pipeline.nodes.retriever import _mmr, _normalize
        emb_a = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        emb_b = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)  # identical to a
        emb_c = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=float)   # fully diverse
        query  = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        doc_embs = _normalize(np.array([emb_a, emb_b, emb_c]))
        selected = _mmr(query, doc_embs, [0, 1, 2], k=2)
        sim = float(np.dot(doc_embs[selected[0]], doc_embs[selected[1]]))
        assert sim < 0.95, f"Selected chunks too similar: cosine={sim:.3f}"

    def test_full_pipeline_runs_end_to_end(self):
        """Run the compiled graph with mocked retriever."""
        from pipeline.graph import build_graph
        from pipeline.nodes import retriever as ret_module

        mock_model = self._mock_model()
        mock_col = self._mock_collection(3)

        with patch.object(ret_module, "_get_model", return_value=mock_model), \
             patch.object(ret_module, "get_collection", return_value=mock_col):
            compiled = build_graph().compile()
            result = compiled.invoke(PipelineState(
                query="What is MSME loan eligibility?",
                lang_hint="en",
                latency_map={},
                self_rag_retries=0,
            ))

        assert "answer" in result
        assert "query_id" in result
        assert "chunks" in result
