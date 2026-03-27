"""
Citation builder node — assembles the final response schema.

Adds query_id and computes total latency.
The full response dict is what gets serialised and sent to the client.
"""
from __future__ import annotations
import uuid
import structlog
from pipeline.state import PipelineState

logger = structlog.get_logger(__name__)


def citation_builder_node(state: PipelineState) -> PipelineState:
    # Ensure query_id exists
    if not state.get("query_id"):
        state["query_id"] = str(uuid.uuid4())

    # Compute total latency
    latency_map = dict(state.get("latency_map") or {})
    latency_map["total"] = round(sum(latency_map.values()), 1)
    state["latency_map"] = latency_map

    logger.info(
        "citation_builder.done",
        query_id=state["query_id"],
        total_ms=latency_map.get("total", 0),
        confidence=state.get("confidence", 0),
        sources=len(state.get("sources", [])),
    )
    return state
