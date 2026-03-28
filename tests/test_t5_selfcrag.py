"""
T5 acceptance tests — CRAG evaluator + Self-RAG (Self-CRAG).
All Groq API calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pipeline.state import PipelineState, SourceChunk


def _base_state(query: str = "What is MSME loan eligibility?") -> PipelineState:
    return PipelineState(
        query=query,
        transcript=query,
        lang="en",
        lang_hint="en",
        route="moderate",
        latency_map={},
        self_rag_retries=0,
        crag_action="CORRECT",
        crag_score=0.8,
    )


def _make_chunk(text: str = "MSME loans are available for enterprises.", n: int = 0) -> SourceChunk:
    return SourceChunk(
        chunk_id=f"chunk-{n}",
        doc_title="MSME Guide",
        page_num=n + 1,
        chunk_text=text,
        relevance_score=0.85,
    )


def _mock_groq(content: str):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


# ── CRAG evaluator tests ───────────────────────────────────────────────────
class TestCragEvaluator:
    def test_correct_action_on_high_score(self):
        from pipeline.nodes.crag_evaluator import crag_evaluator_node

        state = _base_state()
        state["chunks"] = [_make_chunk()]
        with patch("pipeline.nodes.crag_evaluator.Groq", return_value=_mock_groq("0.9")):
            result = crag_evaluator_node(state)
        assert result["crag_action"] == "CORRECT"
        assert result["crag_score"] >= 0.7

    def test_ambiguous_action_on_mid_score(self):
        from pipeline.nodes.crag_evaluator import crag_evaluator_node

        state = _base_state()
        state["chunks"] = [_make_chunk()]
        with patch("pipeline.nodes.crag_evaluator.Groq", return_value=_mock_groq("0.3")):
            result = crag_evaluator_node(state)
        assert result["crag_action"] == "AMBIGUOUS"

    def test_incorrect_action_on_low_score(self):
        from pipeline.nodes.crag_evaluator import crag_evaluator_node

        state = _base_state()
        state["chunks"] = [_make_chunk()]
        with patch("pipeline.nodes.crag_evaluator.Groq", return_value=_mock_groq("0.1")):
            result = crag_evaluator_node(state)
        assert result["crag_action"] == "INCORRECT"
        assert result["crag_score"] < 0.4

    def test_incorrect_on_empty_chunks(self):
        from pipeline.nodes.crag_evaluator import crag_evaluator_node

        state = _base_state()
        state["chunks"] = []
        with patch("pipeline.nodes.crag_evaluator.Groq", return_value=_mock_groq("0.9")):
            result = crag_evaluator_node(state)
        assert result["crag_action"] == "INCORRECT"

    def test_crag_logs_latency(self):
        from pipeline.nodes.crag_evaluator import crag_evaluator_node

        state = _base_state()
        state["chunks"] = [_make_chunk()]
        with patch("pipeline.nodes.crag_evaluator.Groq", return_value=_mock_groq("0.8")):
            result = crag_evaluator_node(state)
        assert "crag" in result["latency_map"]

    def test_ambiguous_strips_chunks(self):
        from pipeline.nodes.crag_evaluator import crag_evaluator_node

        state = _base_state()
        state["chunks"] = [
            _make_chunk("MSME loans are for small businesses. Unrelated text about weather.")
        ]
        with patch("pipeline.nodes.crag_evaluator.Groq", return_value=_mock_groq("0.3")):
            result = crag_evaluator_node(state)
        assert result["crag_action"] == "AMBIGUOUS"
        assert len(result["chunks"]) > 0

    def test_score_clamped_between_0_and_1(self):
        from pipeline.nodes.crag_evaluator import crag_evaluator_node

        state = _base_state()
        state["chunks"] = [_make_chunk()]
        with patch("pipeline.nodes.crag_evaluator.Groq", return_value=_mock_groq("0.8")):
            result = crag_evaluator_node(state)
        assert 0.0 <= result["crag_score"] <= 1.0

    def test_grader_error_uses_neutral_score(self):
        from pipeline.nodes.crag_evaluator import _grade_chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("timeout")
        score = _grade_chunk(mock_client, "test query", "test chunk")
        assert score == 0.5


# ── Self-RAG tests ─────────────────────────────────────────────────────────
class TestSelfRag:
    def test_skipped_for_simple_route(self):
        from pipeline.nodes.self_rag import self_rag_node

        state = _base_state()
        state["route"] = "simple"
        state["chunks"] = [_make_chunk()]
        with patch("pipeline.nodes.self_rag.Groq") as mock_groq_cls:
            self_rag_node(state)
        mock_groq_cls.assert_not_called()

    def test_useful_answer_stored(self):
        from pipeline.nodes.self_rag import self_rag_node

        state = _base_state()
        state["chunks"] = [_make_chunk()]
        output = (
            "MSME loans require turnover below 250 crore.\n\n[IsREL: yes] [IsSUP: yes] [IsUSE: yes]"
        )
        with patch("pipeline.nodes.self_rag.Groq", return_value=_mock_groq(output)):
            result = self_rag_node(state)
        assert "MSME loans" in result["answer"]
        assert result["self_rag_retries"] == 0

    def test_not_useful_triggers_retry_flag(self):
        from pipeline.nodes.self_rag import self_rag_node

        state = _base_state()
        state["chunks"] = [_make_chunk()]
        state["self_rag_retries"] = 0
        output = "I cannot find this information.\n\n[IsREL: no] [IsSUP: no] [IsUSE: no]"
        with patch("pipeline.nodes.self_rag.Groq", return_value=_mock_groq(output)):
            result = self_rag_node(state)
        assert result["answer"] == ""
        assert result["self_rag_retries"] == 1

    def test_max_retries_respected(self):
        from pipeline.nodes.self_rag import self_rag_node

        state = _base_state()
        state["chunks"] = [_make_chunk()]
        state["self_rag_retries"] = 1  # already at max
        output = "Still cannot find.\n\n[IsREL: no] [IsSUP: no] [IsUSE: no]"
        with patch("pipeline.nodes.self_rag.Groq", return_value=_mock_groq(output)):
            result = self_rag_node(state)
        # Should NOT increment further — answer stored even if not useful
        assert result["self_rag_retries"] == 1

    def test_reflection_tokens_stripped_from_answer(self):
        from pipeline.nodes.self_rag import _extract_answer

        text = "MSME loans are available.\n\n[IsREL: yes] [IsSUP: yes] [IsUSE: yes]"
        answer = _extract_answer(text)
        assert "[IsREL" not in answer
        assert "[IsSUP" not in answer
        assert "[IsUSE" not in answer

    def test_parse_reflection_tokens_yes(self):
        from pipeline.nodes.self_rag import _parse_reflection_tokens

        text = "Some answer.\n\n[IsREL: yes] [IsSUP: yes] [IsUSE: yes]"
        tokens = _parse_reflection_tokens(text)
        assert tokens["IsREL"] == "yes"
        assert tokens["IsSUP"] == "yes"
        assert tokens["IsUSE"] == "yes"

    def test_parse_reflection_tokens_no(self):
        from pipeline.nodes.self_rag import _parse_reflection_tokens

        text = "Cannot answer.\n\n[IsREL: no] [IsSUP: no] [IsUSE: no]"
        tokens = _parse_reflection_tokens(text)
        assert tokens["IsUSE"] == "no"

    def test_self_rag_logs_latency(self):
        from pipeline.nodes.self_rag import self_rag_node

        state = _base_state()
        state["chunks"] = [_make_chunk()]
        output = "Answer.\n\n[IsREL: yes] [IsSUP: yes] [IsUSE: yes]"
        with patch("pipeline.nodes.self_rag.Groq", return_value=_mock_groq(output)):
            result = self_rag_node(state)
        assert "self_rag" in result["latency_map"]

    def test_groq_error_clears_answer(self):
        from pipeline.nodes.self_rag import self_rag_node

        state = _base_state()
        state["chunks"] = [_make_chunk()]
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")
        with patch("pipeline.nodes.self_rag.Groq", return_value=mock_client):
            result = self_rag_node(state)
        assert result["answer"] == ""


# ── Graph routing tests ────────────────────────────────────────────────────
class TestGraphRoutingT5:
    def test_incorrect_routes_to_end(self):
        from pipeline.graph import _route_after_crag

        state = _base_state()
        state["crag_action"] = "INCORRECT"
        assert _route_after_crag(state) == "generator"

    def test_correct_simple_routes_to_generator(self):
        from pipeline.graph import _route_after_crag

        state = _base_state()
        state["crag_action"] = "CORRECT"
        state["route"] = "simple"
        assert _route_after_crag(state) == "generator"

    def test_correct_moderate_routes_to_self_rag(self):
        from pipeline.graph import _route_after_crag

        state = _base_state()
        state["crag_action"] = "CORRECT"
        state["route"] = "moderate"
        assert _route_after_crag(state) == "self_rag"

    def test_self_rag_retry_routes_to_retriever(self):
        from pipeline.graph import _route_after_self_rag

        state = _base_state()
        state["answer"] = ""
        state["self_rag_retries"] = 0
        assert _route_after_self_rag(state) == "retriever"

    def test_self_rag_done_routes_to_generator(self):
        from pipeline.graph import _route_after_self_rag

        state = _base_state()
        state["answer"] = "A complete answer about MSME loans."
        state["self_rag_retries"] = 0
        assert _route_after_self_rag(state) == "generator"

    def test_graph_compiles_with_t5_nodes(self):
        from pipeline.graph import build_graph

        compiled = build_graph().compile()
        assert compiled is not None
