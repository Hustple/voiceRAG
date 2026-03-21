# VoiceRAG

> Bilingual Hindi/English voice-driven RAG API — Self-CRAG over Indian government financial documents.

**Demo:** [Loom link — added after T10]

## Quick start

```bash
# 1. Add your Groq API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=...

# 2. Start the API
docker compose up -d

# 3. Index the corpus
python -m ingestion.ingest

# 4. Query
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query/text \
     -H "Content-Type: application/json" \
     -d '{"query": "MSME loan eligibility kya hai?", "lang": "hi"}'
```

## Architecture

```
Voice/Text → ASR (Whisper/Saaras) → Query Classifier → [HyDE] → ChromaDB retrieval
→ CRAG evaluator → [Self-RAG reflection] → Groq generator → Citation builder → SSE stream
```

## Results

| Metric | Naive RAG | VoiceRAG (Self-CRAG) |
|--------|-----------|----------------------|
| Faithfulness | — | — |
| Answer relevancy | — | — |
| Context precision | — | — |

*(populated after T9 eval run)*
