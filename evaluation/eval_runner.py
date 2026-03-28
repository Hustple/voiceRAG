"""
RAGAs evaluation harness for VoiceRAG.

Runs the pipeline against ground truth QA pairs and computes:
  - faithfulness       : is the answer supported by retrieved context?
  - answer_relevancy   : does the answer address the query?
  - context_precision  : are retrieved chunks relevant to the query?

Usage:
  python -m evaluation.eval_runner                  # full run (all queries)
  python -m evaluation.eval_runner --lang en        # English only
  python -m evaluation.eval_runner --lang hi        # Hindi only
  python -m evaluation.eval_runner --dry-run        # first 3 queries only
  python -m evaluation.eval_runner --baseline       # run naive RAG baseline

Results saved to evaluation/results/selfcrag.json (or baseline_naive.json).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import structlog
from rich.console import Console
from rich.table import Table

logger = structlog.get_logger(__name__)
console = Console()

RESULTS_DIR = Path("evaluation/results")
QUERIES_EN = Path("evaluation/test_queries_en.json")
QUERIES_HI = Path("evaluation/test_queries_hi.json")


def _load_queries(lang: Optional[str]) -> list[dict]:
    queries = []
    if lang in (None, "en") and QUERIES_EN.exists():
        queries.extend(json.loads(QUERIES_EN.read_text()))
    if lang in (None, "hi") and QUERIES_HI.exists():
        queries.extend(json.loads(QUERIES_HI.read_text()))
    return queries


def _run_pipeline(query: str, lang: str) -> dict:
    """Run the full Self-CRAG pipeline and return result dict."""
    from pipeline.graph import pipeline
    from pipeline.state import PipelineState

    state = PipelineState(
        query=query,
        lang_hint=lang,
        latency_map={},
        self_rag_retries=0,
    )
    result = pipeline.invoke(state)
    return result


def _run_naive_rag(query: str, lang: str) -> dict:
    """
    Naive RAG baseline: retrieve → generate directly, no CRAG/Self-RAG.
    Used to demonstrate Self-CRAG improvement.
    """
    from pipeline.nodes.asr import asr_node
    from pipeline.nodes.citation_builder import citation_builder_node
    from pipeline.nodes.generator import generator_node
    from pipeline.nodes.retriever import retriever_node
    from pipeline.state import PipelineState

    state = PipelineState(
        query=query,
        lang_hint=lang,
        route="simple",  # force simple path
        crag_action="CORRECT",  # skip CRAG
        crag_score=0.5,
        latency_map={},
        self_rag_retries=0,
    )
    state = asr_node(state)
    state = retriever_node(state)
    state["crag_action"] = "CORRECT"
    state["crag_score"] = 0.5
    state = generator_node(state)
    state = citation_builder_node(state)
    return state


def _compute_ragas_scores(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, Optional[float]]:
    """
    Compute RAGAs metrics.
    Falls back to simple heuristics if ragas library has issues.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        data = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            }
        )
        scores = evaluate(data, metrics=[faithfulness, answer_relevancy, context_precision])
        return {
            "faithfulness": round(float(scores["faithfulness"]), 4),
            "answer_relevancy": round(float(scores["answer_relevancy"]), 4),
            "context_precision": round(float(scores["context_precision"]), 4),
        }
    except Exception as exc:
        logger.warning("ragas.eval_failed", error=str(exc), fallback="heuristic")
        return _heuristic_scores(query, answer, contexts, ground_truth)


def _heuristic_scores(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, Optional[float]]:
    """
    Lightweight heuristic scores when RAGAs library is unavailable.
    Used in CI and dry-run mode.
    """
    # Faithfulness: what fraction of answer words appear in contexts?
    context_text = " ".join(contexts).lower()
    answer_words = set(answer.lower().split())
    ctx_words = set(context_text.split())
    faithfulness = len(answer_words & ctx_words) / max(len(answer_words), 1)

    # Answer relevancy: query word overlap with answer
    query_words = set(query.lower().split())
    relevancy = len(query_words & answer_words) / max(len(query_words), 1)

    # Context precision: ground truth word overlap with contexts
    gt_words = set(ground_truth.lower().split())
    precision = len(gt_words & ctx_words) / max(len(gt_words), 1)

    return {
        "faithfulness": round(min(faithfulness, 1.0), 4),
        "answer_relevancy": round(min(relevancy, 1.0), 4),
        "context_precision": round(min(precision, 1.0), 4),
    }


def run_evaluation(
    lang: Optional[str] = None,
    dry_run: bool = False,
    baseline: bool = False,
) -> dict:
    queries = _load_queries(lang)
    if not queries:
        console.print("[red]No queries found. Check evaluation/*.json files.[/red]")
        return {}

    if dry_run:
        queries = queries[:3]
        console.print(f"[yellow]Dry run — using first {len(queries)} queries only[/yellow]\n")

    mode = "Naive RAG baseline" if baseline else "VoiceRAG Self-CRAG"
    console.print(f"\n[bold]Evaluating:[/bold] {mode}")
    console.print(f"[bold]Queries:[/bold] {len(queries)} ({lang or 'en+hi'})\n")

    results = []
    t_total = time.perf_counter()

    for i, item in enumerate(queries, 1):
        q_id = item["id"]
        query = item["query"]
        ground_truth = item["ground_truth"]
        q_lang = item.get("lang", "en")

        console.print(f"[{i}/{len(queries)}] {q_id}: {query[:60]}...")

        try:
            if baseline:
                result = _run_naive_rag(query, q_lang)
            else:
                result = _run_pipeline(query, q_lang)

            answer = result.get("answer", "")
            sources = result.get("sources", result.get("chunks", []))
            contexts = [s.get("chunk_text", s.get("text", "")) for s in sources]

            ragas = _compute_ragas_scores(query, answer, contexts, ground_truth)
            latency_ms = result.get("latency_map", {}).get("total", 0)

            entry = {
                "id": q_id,
                "query": query,
                "lang": q_lang,
                "answer": answer,
                "ground_truth": ground_truth,
                "routing_path": result.get("route", "unknown"),
                "crag_action": result.get("crag_action", "UNKNOWN"),
                "confidence": result.get("confidence", 0.0),
                "self_rag_retries": result.get("self_rag_retries", 0),
                "latency_ms": latency_ms,
                "ragas": ragas,
            }
            results.append(entry)

            console.print(
                f"    faith={ragas['faithfulness']:.2f} "
                f"rel={ragas['answer_relevancy']:.2f} "
                f"prec={ragas['context_precision']:.2f} "
                f"latency={latency_ms:.0f}ms"
            )

        except Exception as exc:
            logger.error("eval.query_failed", q_id=q_id, error=str(exc))
            console.print(f"    [red]FAILED: {exc}[/red]")

    elapsed = time.perf_counter() - t_total

    # Aggregate
    def avg(key: str) -> Optional[float]:
        vals = [r["ragas"].get(key) for r in results if r["ragas"].get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "mode": mode,
        "lang_filter": lang,
        "total_queries": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "aggregate": {
            "faithfulness": avg("faithfulness"),
            "answer_relevancy": avg("answer_relevancy"),
            "context_precision": avg("context_precision"),
            "avg_latency_ms": round(
                sum(r["latency_ms"] for r in results) / max(len(results), 1), 1
            ),
            "avg_confidence": round(
                sum(r["confidence"] for r in results) / max(len(results), 1), 3
            ),
        },
        "queries": results,
    }

    # Print summary table
    table = Table(title=f"Results — {mode}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    agg = summary["aggregate"]
    table.add_row("Faithfulness", f"{agg['faithfulness']:.4f}" if agg["faithfulness"] else "N/A")
    table.add_row(
        "Answer relevancy", f"{agg['answer_relevancy']:.4f}" if agg["answer_relevancy"] else "N/A"
    )
    table.add_row(
        "Context precision",
        f"{agg['context_precision']:.4f}" if agg["context_precision"] else "N/A",
    )
    table.add_row("Avg latency (ms)", f"{agg['avg_latency_ms']:.0f}")
    table.add_row("Avg confidence", f"{agg['avg_confidence']:.3f}")
    console.print(table)
    console.print(f"\nCompleted {len(results)} queries in {elapsed:.1f}s\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceRAG RAGAs evaluation")
    parser.add_argument("--lang", choices=["en", "hi"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--baseline", action="store_true", help="Run naive RAG baseline instead of Self-CRAG"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = run_evaluation(
        lang=args.lang,
        dry_run=args.dry_run,
        baseline=args.baseline,
    )

    if summary:
        fname = "baseline_naive.json" if args.baseline else "selfcrag.json"
        out_path = RESULTS_DIR / fname
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        console.print(f"[green]Results saved:[/green] {out_path}\n")


if __name__ == "__main__":
    main()
