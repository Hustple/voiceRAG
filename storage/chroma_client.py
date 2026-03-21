from __future__ import annotations
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings

_client: chromadb.ClientAPI | None = None
COLLECTION_NAME = "voicerag_corpus"

def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client

def get_collection() -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
