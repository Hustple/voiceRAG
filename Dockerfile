FROM python:3.11-slim

LABEL maintainer="Utkarsh"
LABEL description="VoiceRAG — Hindi/English voice-driven Self-CRAG API"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 git curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml ./

RUN pip install --upgrade pip && pip install \
        fastapi==0.115.5 "uvicorn[standard]==0.32.1" python-multipart==0.0.12 \
        langgraph==0.2.55 langchain-core==0.3.25 langchain-groq==0.2.3 groq==0.13.0 \
        chromadb==0.5.23 sentence-transformers==3.3.1 openai-whisper==20240930 \
        langdetect==1.0.9 pymupdf==1.24.14 tiktoken==0.8.0 ragas==0.2.6 datasets==3.2.0 \
        pydantic==2.10.3 pydantic-settings==2.6.1 python-dotenv==1.0.1 httpx==0.28.0 \
        sse-starlette==2.1.3 structlog==24.4.0 rich==13.9.4

COPY app/ ./app/
COPY pipeline/ ./pipeline/
COPY ingestion/ ./ingestion/
COPY storage/ ./storage/
COPY evaluation/ ./evaluation/

RUN mkdir -p /app/data/chroma /app/data/logs /app/data/corpus
RUN addgroup --system voicerag && adduser --system --ingroup voicerag voicerag \
    && chown -R voicerag:voicerag /app
USER voicerag

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
