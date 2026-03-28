from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    chroma_ok: bool
    corpus_chunks: int
    total_queries: int = Field(default=0)
    avg_latency_ms: float | None = Field(default=None)
    avg_confidence: float | None = Field(default=None)
    avg_faithfulness: float | None = Field(default=None)
    routing_distribution: dict[str, int] = Field(default_factory=dict)
    node_latency_avg_ms: dict[str, float] = Field(default_factory=dict)


class TextQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    lang: str | None = Field(default=None)


class SourceChunk(BaseModel):
    chunk_id: str
    doc_title: str
    page_num: int
    chunk_text: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    lang: str
    sources: list[SourceChunk]
    confidence: float = Field(ge=0.0, le=1.0)
    routing_path: str
    crag_action: str
    self_rag_retries: int = Field(default=0)
    latency_ms: dict[str, float]


class ExplainResponse(BaseModel):
    query_id: str
    transcript: str
    lang: str
    routing_path: str
    crag_action: str
    self_rag_retries: int
    confidence: float
    latency_ms: dict[str, float]
    sources: list[SourceChunk]
    ragas: dict[str, float | None] = Field(
        default_factory=lambda: {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
        }
    )
