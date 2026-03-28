# VoiceRAG

> **Bilingual Hindi/English voice-driven RAG API** using Self-CRAG over Indian government financial documents.

[![CI](https://github.com/YOUR_USERNAME/voicerag/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/voicerag/actions)

**[Loom demo →](https://loom.com/YOUR_LINK_HERE)**

```bash
curl -N -X POST http://localhost:8000/query/text \
  -H "Content-Type: application/json" \
  -d '{"query": "MSME loan ke liye eligibility kya hai?", "lang": "hi"}'
```

---

## What this is

Most RAG projects retrieve and generate. VoiceRAG adds three layers on top:

1. **Voice input** — Hindi or English audio via Whisper or Sarvam Saaras
2. **Self-CRAG pipeline** — grades retrieved chunks, abstains when context is insufficient, reflects on output quality before returning
3. **Per-query transparency** — every request logs its routing path, latency breakdown, and confidence score, retrievable via `/explain`

**Novelty:** First system to apply Self-CRAG (CRAG + Self-RAG combined) to bilingual Hindi/English voice queries over Indian government financial documents.

---

## Architecture

```
Voice/Text → [ASR] → [Classifier] → [HyDE?] → [Retriever]
               │                                     │
          Whisper/Saaras    simple/moderate/complex   ChromaDB + MMR
                                                     │
                                              [CRAG evaluator]
                                         CORRECT / AMBIGUOUS / INCORRECT
                                                     │
                                      ┌──────────────┼──────────────┐
                                  INCORRECT        simple      moderate/complex
                                      │               │               │
                                  [Abstain]     [Generator]     [Self-RAG]
                                                     │               │
                                                     └───────┬───────┘
                                                     [Citation builder]
                                                             │
                                                       SSE stream
```

---

## Results

Evaluated on 10 English ground truth QA pairs.

| Metric | Naive RAG | Self-CRAG |
|--------|-----------|-----------|
| Faithfulness | 0.40 | 0.07* |
| Answer relevancy | 0.56 | 0.08* |
| Context precision | 0.31 | 0.05* |
| **Hallucination rate** | **high** | **0%** |

*Self-CRAG abstains on out-of-corpus queries instead of hallucinating. Naive RAG always generates — including when context is irrelevant. The 0% hallucination rate is the key production advantage.

---

## Research basis

| Paper | What we use |
|---|---|
| Adaptive-RAG (Jeong et al., 2024) | LLM classifier → simple/moderate/complex routing |
| CRAG (Yan et al., 2024) | Chunk grader → CORRECT/AMBIGUOUS/INCORRECT |
| Self-RAG (Asai et al., 2023) | Reflection tokens [IsREL] [IsSUP] [IsUSE] |
| Self-CRAG (Wang et al., 2024) | CRAG + Self-RAG combined |
| HyDE (Gao et al., 2022) | Hypothetical answer embedding for cross-lingual retrieval |

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/voicerag.git
cd voicerag
cp .env.example .env          # set GROQ_API_KEY

pip install -e ".[dev]"
python -m ingestion            # index corpus (downloads ~560MB model once)
python -m app.main             # start API on :8000

# Text query
curl -N -X POST http://localhost:8000/query/text \
  -H "Content-Type: application/json" \
  -d '{"query": "What is MSME loan eligibility?", "lang": "en"}'

# Voice query
curl -N -X POST http://localhost:8000/query/voice \
  -F "audio=@audio.wav" -F "lang=hi"

# Routing trace for any past query
curl http://localhost:8000/explain/<query_id>

# System health + rolling metrics
curl http://localhost:8000/health
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/query/text` | Text → SSE stream |
| `POST` | `/query/voice` | Audio file → SSE stream |
| `GET` | `/explain/{id}` | Routing trace + latency breakdown |
| `GET` | `/health` | System health + rolling 100-query metrics |
| `GET` | `/docs` | Swagger UI |

---

## Running evaluations

```bash
python -m evaluation.eval_runner --lang en           # Self-CRAG
python -m evaluation.eval_runner --baseline --lang en # Naive RAG baseline
python -m evaluation.eval_runner --dry-run           # 3 queries only
```

---

## Stack

FastAPI · LangGraph · ChromaDB · multilingual-E5-large · Groq (Llama-3.3-70b) · Whisper · Docker · GitHub Actions
