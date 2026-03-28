"""
Stub implementations — replaced task by task.
T6: generator and citation_builder are now real; stubs remain for asr only.
"""

from __future__ import annotations

from pipeline.state import PipelineState


def asr_node(state: PipelineState) -> PipelineState:
    """T7 — transcribe audio. Stub: pass query through as transcript."""
    state["transcript"] = state.get("query", "")
    state["lang"] = state.get("lang_hint") or "en"
    return state
