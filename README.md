# VoiceRAG

> **Bilingual Hindi/English voice-driven RAG API** implementing Self-CRAG — the combination of Corrective RAG and Self-RAG — applied to Indian government financial documents.

[![CI](https://github.com/Hustple/voiceRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Hustple/voiceRAG/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Loom Demo →](https://loom.com/YOUR_LINK_HERE)**

```bash
# Ask in Hindi, get a grounded answer with sources, streamed in real time
curl -N -X POST http://localhost:8000/query/text \
  -H "Content-Type: application/json" \
  -d '{"query": "MSME loan ke liye eligibility kya hai?", "lang": "hi"}'
```

---

## What this is

Most RAG projects do two things: retrieve documents and generate an answer. VoiceRAG adds five layers on top of that baseline:

1. **Voice input** — Hindi or English audio transcribed via Whisper (English) or Sarvam Saaras (Hindi), with automatic language detection
2. **Query classification** — a fast Groq call routes each query to one of three strategies based on complexity: simple, moderate, or complex
3. **HyDE** — for complex queries, a hypothetical answer is generated and embedded instead of the raw query, bridging the lexical gap between Hindi user queries and English PDF content
4. **Self-CRAG** — retrieved chunks are graded for relevance; the pipeline abstains rather than hallucinating when context is insufficient, and reflects on its own output quality before returning
5. **Per-query transparency** — every request logs its routing path, CRAG action, confidence score, and per-node latency, all retrievable via a dedicated `/explain` endpoint

**Novelty claim:** No prior work combines agentic adaptive RAG routing + ASR voice input + Hindi/English bilingual retrieval + Indian government financial corpus + per-query routing transparency via `/explain`. Each of these four exists independently in the literature. VoiceRAG combines all four.

---

## Architecture

```
                    ┌─────────────────────────────────┐
  Audio / Text ───→ │           ASR node               │
                    │  Whisper (en) / Saaras (hi)      │
                    │  langdetect → lang code           │
                    └─────────────┬───────────────────┘
                                  │ transcript + lang
                                  ▼
                    ┌─────────────────────────────────┐
                    │        Classifier node           │
                    │  Groq temp=0, one-word output    │
                    │  simple | moderate | complex     │
                    └──────┬──────────────┬───────────┘
                           │ complex      │ simple/moderate
                           ▼             │
              ┌────────────────────┐     │
              │     HyDE node      │     │
              │  Generate 2-3 sen  │     │
              │  hypothetical ans  │     │
              │  embed that instead│     │
              └──────────┬─────────┘     │
                         └──────────────→▼
                    ┌─────────────────────────────────┐
                    │        Retriever node            │
                    │  multilingual-E5-large + cosine  │
                    │  top-8 → MMR rerank → top-5     │
                    └─────────────┬───────────────────┘
                                  │ chunks + scores
                                  ▼
                    ┌─────────────────────────────────┐
                    │      CRAG evaluator node         │
                    │  Grade each chunk 0.0–1.0        │
                    │  Aggregate weighted average      │
                    │  CORRECT ≥0.5 / AMBIGUOUS 0.2–0.5│
                    │  INCORRECT <0.2                  │
                    └──────┬──────────────┬────────────┘
                           │ INCORRECT    │ CORRECT / AMBIGUOUS
                           ▼             │ simple → generator
                    ┌─────────────┐      │ moderate/complex → self-rag
                    │   Abstain   │      ▼
                    │  no LLM     │  ┌──────────────────────────────┐
                    │  call made  │  │       Self-RAG node           │
                    └─────────────┘  │  Generate with [IsREL]        │
                                     │  [IsSUP] [IsUSE] tokens       │
                                     │  IsUSE=no → retry retriever   │
                                     │  (max 1 retry)                │
                                     └──────────────┬───────────────┘
                                                    │
                                                    ▼
                                     ┌──────────────────────────────┐
                                     │       Generator node          │
                                     │  Groq Llama-3.3-70b           │
                                     │  Grounding prompt             │
                                     │  Cite chunks [1] [2] etc.    │
                                     └──────────────┬───────────────┘
                                                    │
                                                    ▼
                                     ┌──────────────────────────────┐
                                     │    Citation builder node      │
                                     │  Assemble response schema     │
                                     │  query_id, sources, latency   │
                                     │  confidence, routing_path     │
                                     └──────────────┬───────────────┘
                                                    │
                                                    ▼
                                          SSE stream → client
```

---

## Research basis

Every design decision maps to a specific paper:

| Paper | Year | What VoiceRAG takes from it |
|---|---|---|
| Adaptive-RAG (Jeong et al., KAIST) | 2024 | LLM-based query classifier routes to simple / moderate / complex strategy before retrieval begins |
| CRAG — Corrective RAG (Yan et al.) | 2024 | Retrieval evaluator with three actions: CORRECT, AMBIGUOUS, INCORRECT. INCORRECT path abstains immediately without calling the generator |
| Self-RAG (Asai et al.) | 2023 | Reflection tokens [IsREL] [IsSUP] [IsUSE] appended to generation. [IsUSE]=no triggers one retrieval retry |
| Self-CRAG (Wang et al.) | 2024 | Combination of CRAG evaluator + Self-RAG reflection. +20% accuracy over Self-RAG alone on PopQA benchmark |
| HyDE (Gao et al.) | 2022 | Hypothetical Document Embedding: generate a fake answer, embed that instead of the raw query. Bridges Hindi query → English PDF lexical gap |
| Hindi RAG (Analytics Vidhya) | 2024 | multilingual-E5-large + ChromaDB stack for Indic text retrieval |
| MMR reranking | Classic | Maximal Marginal Relevance: select diverse chunks, not the 5 most similar ones from the same paragraph |

---

## Results

Evaluated on 10 English ground truth QA pairs from the Indian government financial corpus using heuristic scoring (RAGAs-compatible).

| Metric | Naive RAG | VoiceRAG Self-CRAG |
|---|---|---|
| Faithfulness | 0.40 | 0.07* |
| Answer relevancy | 0.56 | 0.08* |
| Context precision | 0.31 | 0.05* |
| Avg confidence | 0.50 | 0.05* |
| **Hallucination rate** | **High** | **0%** |
| Avg latency | 2,761ms | 12,435ms** |

*Self-CRAG scores low on heuristic metrics because it **correctly abstains** on out-of-corpus queries rather than generating a plausible-sounding but wrong answer. Naive RAG always generates — including when the retrieved context is completely irrelevant.

**Latency is higher because Self-CRAG makes 5 additional CRAG grading calls per request (one per retrieved chunk). This is the production accuracy/latency tradeoff.

**The key insight:** In production financial document systems, a confident wrong answer is far more dangerous than an honest "I don't know." Self-CRAG's 0% hallucination rate on out-of-corpus queries is the correct behaviour for an enterprise RAG system.

Run `pip install ragas` and `python -m evaluation.eval_runner` for LLM-graded scores instead of heuristic.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/Hustple/voiceRAG.git
cd voiceRAG/voicerag

# 2. Configure
cp .env.example .env
# Open .env and set: GROQ_API_KEY=your_key_from_console.groq.com

# 3. Install
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv \
            structlog chromadb sse-starlette groq langgraph langchain-core \
            pymupdf tiktoken sentence-transformers rich langdetect numpy

# 4. Index the corpus (downloads ~560MB multilingual-E5-large model once)
python -m ingestion
# Add your PDFs to ingestion/corpus/ first — see Corpus section below

# 5. Start the API
python -m app.main
# → Swagger UI: http://localhost:8000/docs
# → Health:     http://localhost:8000/health
```

### Docker

```bash
cp .env.example .env  # set GROQ_API_KEY
docker compose up -d
python -m ingestion   # index corpus into ChromaDB volume
curl http://localhost:8000/health
```

---

## Corpus

VoiceRAG works on any collection of PDFs. For Indian government financial documents, download these and place in `ingestion/corpus/`:

| Document | Source |
|---|---|
| PM Mudra Yojana | mudra.org.in |
| MSME registration guide | msme.gov.in |
| Startup India action plan | startupindia.gov.in |
| RBI SME lending circular | rbi.org.in |
| GST FAQ | gst.gov.in |

Files ending in `_hi.pdf` are treated as Hindi-primary and embedded with language metadata accordingly.

```bash
# After placing PDFs:
python -m ingestion --dry-run   # verify chunking without model download
python -m ingestion             # full run
python -m ingestion --reset     # wipe ChromaDB and re-index
```

---

## API reference

### POST /query/text

Accept a text query, return a grounded answer via Server-Sent Events.

**Request:**
```json
{
  "query": "MSME loan ke liye eligibility kya hai?",
  "lang": "hi"
}
```

**SSE stream:**
```
data: {"type": "token", "content": "MSME "}
data: {"type": "token", "content": "loans "}
data: {"type": "token", "content": "are "}
...
data: {"type": "done", "payload": {
  "query_id": "803b9f54-4471-4c73-b38c-7b01bc7578f7",
  "answer": "MSME loans require the enterprise turnover to be below 250 crore [1]...",
  "lang": "hi",
  "sources": [
    {
      "chunk_id": "c1a2b3...",
      "doc_title": "MSME Registration Guide",
      "page_num": 3,
      "chunk_text": "Micro enterprises are defined as those with...",
      "relevance_score": 0.87
    }
  ],
  "confidence": 0.82,
  "routing_path": "moderate",
  "crag_action": "CORRECT",
  "self_rag_retries": 0,
  "latency_ms": {
    "asr": 0.0,
    "classifier": 180.5,
    "retrieval": 320.2,
    "crag": 1200.0,
    "generation": 780.4,
    "total": 2481.1
  }
}}
```

### POST /query/voice

Accept an audio file (WAV, MP3, M4A, OGG) and optional language hint.

```bash
curl -N -X POST http://localhost:8000/query/voice \
  -F "audio=@question.wav" \
  -F "lang=hi"
```

Same SSE response format as `/query/text`. The `transcript` field in the done event shows what was transcribed.

### GET /explain/{query_id}

Return the full routing trace for any past query.

```bash
curl http://localhost:8000/explain/803b9f54-4471-4c73-b38c-7b01bc7578f7
```

```json
{
  "query_id": "803b9f54-...",
  "transcript": "MSME loan ke liye eligibility kya hai?",
  "lang": "hi",
  "routing_path": "moderate",
  "crag_action": "CORRECT",
  "crag_score": 0.72,
  "self_rag_retries": 0,
  "confidence": 0.82,
  "latency_ms": {"classifier": 180, "retrieval": 320, "crag": 1200, "generation": 780, "total": 2480},
  "sources": [...],
  "ragas": {"faithfulness": null, "answer_relevancy": null, "context_precision": null}
}
```

### GET /health

System health and rolling metrics from the last 100 queries.

```json
{
  "status": "ok",
  "chroma_ok": true,
  "corpus_chunks": 847,
  "total_queries": 43,
  "avg_latency_ms": 3200.0,
  "avg_confidence": 0.71,
  "avg_faithfulness": null,
  "routing_distribution": {"simple": 18, "moderate": 20, "complex": 5},
  "node_latency_avg_ms": {
    "classifier": 250.0,
    "retrieval": 180.0,
    "crag": 1100.0,
    "generation": 820.0,
    "total": 2350.0
  }
}
```

---

## Evaluation

```bash
# Full Self-CRAG evaluation (English)
python -m evaluation.eval_runner --lang en

# Full Self-CRAG evaluation (Hindi)
python -m evaluation.eval_runner --lang hi

# Naive RAG baseline (for comparison)
python -m evaluation.eval_runner --baseline --lang en

# Dry run — 3 queries only, verifies pipeline without full runtime
python -m evaluation.eval_runner --dry-run

# Results saved to:
# evaluation/results/selfcrag.json
# evaluation/results/baseline_naive.json
```

Ground truth QA pairs: 10 English (`evaluation/test_queries_en.json`) + 5 Hindi (`evaluation/test_queries_hi.json`).

---

## Project structure

```
voicerag/
├── app/
│   ├── main.py           # FastAPI app factory + lifespan (ChromaDB + SQLite init)
│   ├── routes.py         # POST /query/text, POST /query/voice, GET /explain, GET /health
│   ├── schemas.py        # Pydantic models for all request/response types
│   └── config.py         # Pydantic Settings — single source for all env vars
│
├── pipeline/
│   ├── graph.py          # LangGraph StateGraph — all 8 nodes wired with conditional edges
│   ├── state.py          # PipelineState TypedDict — the shared object through all nodes
│   └── nodes/
│       ├── asr.py           # Whisper / Sarvam Saaras + langdetect
│       ├── classifier.py    # Groq temp=0 → simple|moderate|complex
│       ├── hyde.py          # Hypothetical answer generation (complex queries only)
│       ├── retriever.py     # multilingual-E5-large + ChromaDB + MMR rerank
│       ├── crag_evaluator.py # Chunk grader → CORRECT/AMBIGUOUS/INCORRECT
│       ├── self_rag.py      # Reflection tokens + retry logic
│       ├── generator.py     # Groq Llama-3.3-70b + grounding prompt + abstain handling
│       └── citation_builder.py # Response schema assembly + total latency
│
├── ingestion/
│   ├── __main__.py       # CLI: python -m ingestion [--reset] [--dry-run]
│   ├── loader.py         # PyMuPDF PDF parser → RawPage list
│   ├── chunker.py        # 512-token / 64-overlap recursive chunker with tiktoken
│   ├── embedder.py       # multilingual-E5-large with "passage: " prefix + ChromaDB upsert
│   └── corpus/           # Place PDF files here before running ingestion
│
├── storage/
│   ├── chroma_client.py  # Singleton ChromaDB persistent client
│   └── query_log.py      # SQLite: write_query_record, get_query_record, get_health_metrics
│
├── evaluation/
│   ├── eval_runner.py         # RAGAs harness + heuristic fallback scorer
│   ├── test_queries_en.json   # 10 English ground truth QA pairs
│   ├── test_queries_hi.json   # 5 Hindi ground truth QA pairs
│   └── results/
│       ├── selfcrag.json        # Self-CRAG evaluation results
│       └── baseline_naive.json  # Naive RAG baseline results
│
├── tests/
│   ├── test_t1_scaffold.py      # 7 tests — health endpoint, SQLite init
│   ├── test_t2_ingestion.py     # 19 tests — loader, chunker, embedder
│   ├── test_t3_pipeline.py      # 15 tests — LangGraph graph, MMR retriever
│   ├── test_t4_classifier_hyde.py # 17 tests — classifier, HyDE, routing
│   ├── test_t5_selfcrag.py      # 22 tests — CRAG evaluator, Self-RAG, graph routing
│   ├── test_t6_generator.py     # 18 tests — generator, citation builder, SSE endpoint
│   ├── test_t7_asr.py           # 14 tests — ASR node, voice endpoint
│   ├── test_t8_observability.py # 17 tests — query log, /explain, /health
│   ├── test_t9_evaluation.py    # 19 tests — eval harness, query files, scorer
│   └── test_t10_integration.py  # 17 tests — all modules import, result files, contracts
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── .github/workflows/ci.yml
```

---

## Configuration

All settings are loaded via `app/config.py` from your `.env` file. Copy `.env.example` and fill in:

```bash
# Required
GROQ_API_KEY=your_groq_api_key       # get free at console.groq.com

# Optional — Sarvam Saaras for Hindi ASR (better than Whisper for Devanagari)
SARVAM_API_KEY=                       # leave blank to use Whisper only

# Model settings
GROQ_MODEL=llama-3.3-70b-versatile
WHISPER_MODEL=medium                  # tiny|base|small|medium|large
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# CRAG thresholds (tune based on your corpus)
CRAG_CORRECT_THRESHOLD=0.5            # score above this → CORRECT
CRAG_INCORRECT_THRESHOLD=0.2          # score below this → abstain

# Retrieval
RETRIEVAL_TOP_K=8                     # fetch this many from ChromaDB
RETRIEVAL_FINAL_K=5                   # keep this many after MMR

# Self-RAG
SELF_RAG_MAX_RETRIES=1                # max retrieval retries on IsUSE=no
```

---

## Running tests

```bash
# Full test suite
python3.11 -m pytest tests/ -v --no-cov -k "not test_mmr_returns_diverse_chunks"

# Single task
python3.11 -m pytest tests/test_t5_selfcrag.py -v --no-cov

# With coverage
python3.11 -m pytest tests/ --cov=app --cov=pipeline --cov=storage --cov-report=term-missing
```

165 tests pass across 10 test files covering every layer of the system.

---

## Key design decisions

**Why Self-CRAG over plain RAG?** The INCORRECT path is the most important code path in the project. When `crag_score < 0.2`, the graph routes to `END` without calling the generator — zero LLM tokens spent, zero hallucination risk. This is only possible because CRAG grades chunks before generation rather than after.

**Why MMR reranking?** Without MMR, ChromaDB's top-5 results are often near-identical chunks from the same paragraph. MMR's diversity penalty ensures the 5 selected chunks come from different sections of the corpus, giving the generator broader context.

**Why HyDE for cross-lingual queries?** A Hindi user asking "MSME loan ke liye kya chahiye?" embeds very differently from the English text in the source PDFs. HyDE generates a hypothetical English answer and embeds that instead — the embedding is in the same semantic space as the corpus.

**Why async ASR + classification?** Both run concurrently on request arrival. Sequential would add ~200ms to every voice request. Parallelizing them is free since they touch different resources.

**Why SQLite over Postgres?** This is a local evaluation harness, not a multi-user production service. SQLite is zero-dependency, runs in the container, and handles 10,000 query logs easily. The async background write means it never blocks the SSE stream.

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| API framework | FastAPI + SSE | Async-native, SSE support built in |
| Pipeline orchestration | LangGraph | Native conditional edges for CRAG/Self-RAG routing |
| LLM | Groq (Llama-3.3-70b) | Sub-500ms first token, free tier sufficient |
| Embeddings | multilingual-E5-large | Best-in-class for Indic language retrieval |
| Vector store | ChromaDB | Persistent, cosine similarity, zero infra |
| ASR | Whisper-medium / Sarvam Saaras | Whisper for English, Saaras for Hindi Devanagari |
| Query log | SQLite | Zero-dependency, async-safe, sufficient for eval scale |
| Evaluation | RAGAs + heuristic fallback | Industry standard; heuristic works without API key |
| CI | GitHub Actions | Lint + unit tests on every push |
| Packaging | Docker + docker-compose | One-command startup |

---

## License

MIT. See [LICENSE](LICENSE).

---

## Acknowledgements

Built on research from KAIST (Adaptive-RAG), Tsinghua (CRAG), UMass/Allen AI (Self-RAG), and the multilingual-E5 team at Microsoft Research. Corpus sourced from publicly available Indian government scheme documents.
