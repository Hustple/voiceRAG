from __future__ import annotations
import time
from typing import Any
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer
from app.config import settings
from pipeline.state import PipelineState, SourceChunk
from storage.chroma_client import get_collection

logger = structlog.get_logger(__name__)
E5_QUERY_PREFIX = "query: "
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("retriever.loading_model", model=settings.EMBEDDING_MODEL)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norm == 0, 1, norm)


def _mmr(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    docs: list[Any],
    k: int,
    lambda_: float = 0.7,
) -> list[int]:
    query_emb = _normalize(query_emb.reshape(1, -1))[0]
    doc_embs = _normalize(doc_embs)

    selected: list[int] = []
    candidate_indices = list(range(len(docs)))

    while len(selected) < k and candidate_indices:
        relevance = np.array([
            float(np.dot(query_emb, doc_embs[i]))
            for i in candidate_indices
        ])

        if not selected:
            best_local = int(np.argmax(relevance))
        else:
            selected_embs = doc_embs[selected]
            redundancy = np.array([
                float(np.max(selected_embs @ doc_embs[i].reshape(-1, 1)))
                for i in candidate_indices
            ])
            # Subtract tiny redundancy penalty to break ties toward diversity
            mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy - 1e-6 * redundancy
            best_local = int(np.argmax(mmr_score))

        best_global = candidate_indices[best_local]
        selected.append(best_global)
        candidate_indices.pop(best_local)

    return selected


def retriever_node(state: PipelineState) -> PipelineState:
    t_start = time.perf_counter()
    query_text = state.get("hyde_query") or state.get("transcript") or state.get("query", "")
    query_to_embed = E5_QUERY_PREFIX + query_text

    model = _get_model()
    query_emb = model.encode(
        [query_to_embed], normalize_embeddings=True, show_progress_bar=False,
    )[0]

    collection = get_collection()
    top_k = settings.RETRIEVAL_TOP_K
    final_k = settings.RETRIEVAL_FINAL_K

    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=min(top_k, collection.count() or top_k),
        include=["documents", "metadatas", "embeddings", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        logger.warning("retriever.no_results", query=query_text[:80])
        state["chunks"] = []
        state["raw_scores"] = []
        latency_map = dict(state.get("latency_map") or {})
        latency_map["retrieval"] = round((time.perf_counter() - t_start) * 1000, 1)
        state["latency_map"] = latency_map
        return state

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    raw_scores = [round(1 - d, 4) for d in distances]
    doc_embs = np.array(results["embeddings"][0])

    selected_indices = _mmr(
        query_emb=query_emb,
        doc_embs=doc_embs,
        docs=docs,
        k=min(final_k, len(docs)),
    )

    chunks: list[SourceChunk] = []
    selected_scores: list[float] = []
    for idx in selected_indices:
        meta = metas[idx]
        chunks.append(SourceChunk(
            chunk_id=results["ids"][0][idx],
            doc_title=meta.get("doc_title", "Unknown"),
            page_num=int(meta.get("page_num", 0)),
            chunk_text=docs[idx],
            relevance_score=raw_scores[idx],
        ))
        selected_scores.append(raw_scores[idx])

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
    logger.info("retriever.done", chunks_returned=len(chunks), latency_ms=elapsed_ms)

    latency_map = dict(state.get("latency_map") or {})
    latency_map["retrieval"] = elapsed_ms
    state["chunks"] = chunks
    state["raw_scores"] = selected_scores
    state["latency_map"] = latency_map
    return state
