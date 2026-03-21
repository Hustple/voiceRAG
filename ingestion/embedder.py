"""
Embedder — wraps multilingual-E5-large and batch-upserts chunks into ChromaDB.

E5 requires "passage: " prefix at ingestion and "query: " prefix at retrieval.
"""
from __future__ import annotations
import structlog
from sentence_transformers import SentenceTransformer
from app.config import settings
from ingestion.chunker import Chunk
from storage.chroma_client import get_collection

logger = structlog.get_logger(__name__)

BATCH_SIZE = 64
E5_PASSAGE_PREFIX = "passage: "

def _load_model() -> SentenceTransformer:
    logger.info("embedder.loading_model", model=settings.EMBEDDING_MODEL)
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    logger.info("embedder.model_ready")
    return model

def embed_and_upsert(chunks: list[Chunk]) -> int:
    if not chunks:
        logger.warning("embedder.no_chunks")
        return 0
    model = _load_model()
    collection = get_collection()
    total_upserted = 0
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts_to_embed = [E5_PASSAGE_PREFIX + c.text for c in batch]
        embeddings = model.encode(
            texts_to_embed,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "doc_title": c.doc_title,
                    "source_path": c.source_path,
                    "page_num": c.page_num,
                    "lang_hint": c.lang_hint,
                    "token_count": c.token_count,
                    "chunk_index": c.chunk_index,
                }
                for c in batch
            ],
        )
        total_upserted += len(batch)
        logger.info("embedder.batch_upserted", upserted=total_upserted, total=len(chunks))
    logger.info("embedder.done", total_upserted=total_upserted)
    return total_upserted
