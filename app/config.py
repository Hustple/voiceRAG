from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    GROQ_API_KEY: str = Field(...)
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    SARVAM_API_KEY: str = Field(default="")
    WHISPER_MODEL: str = Field(default="medium")
    CHROMA_PERSIST_PATH: str = Field(default="./data/chroma")
    QUERY_LOG_PATH: str = Field(default="./data/logs/queries.db")
    CORPUS_PATH: str = Field(default="./ingestion/corpus")
    EMBEDDING_MODEL: str = Field(default="intfloat/multilingual-e5-large")
    CRAG_CORRECT_THRESHOLD: float = Field(default=0.7)
    CRAG_INCORRECT_THRESHOLD: float = Field(default=0.4)
    RETRIEVAL_TOP_K: int = Field(default=8)
    RETRIEVAL_FINAL_K: int = Field(default=5)
    SELF_RAG_MAX_RETRIES: int = Field(default=1)
    LOG_LEVEL: str = Field(default="INFO")
    APP_ENV: str = Field(default="production")


settings = Settings()  # type: ignore[call-arg]
