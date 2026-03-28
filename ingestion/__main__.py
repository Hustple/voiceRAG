"""
Corpus ingestion CLI.

Usage:
    python -m ingestion                      # uses CORPUS_PATH from .env
    python -m ingestion --corpus ./my/pdfs   # override directory
    python -m ingestion --reset              # wipe collection + re-ingest
    python -m ingestion --dry-run            # parse + chunk only, no upsert
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import structlog
from rich.console import Console
from rich.table import Table

from app.config import settings
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_and_upsert
from ingestion.loader import load_corpus
from storage.chroma_client import COLLECTION_NAME, get_chroma_client, get_collection
from storage.query_log import init_query_log

logger = structlog.get_logger(__name__)
console = Console()


def _reset_collection() -> None:
    try:
        get_chroma_client().delete_collection(COLLECTION_NAME)
        logger.info("ingestion.collection_deleted")
    except Exception:
        pass


def _print_summary(chunks: list, elapsed: float) -> None:
    doc_counts: dict[str, dict] = {}
    for c in chunks:
        if c.doc_title not in doc_counts:
            doc_counts[c.doc_title] = {"chunks": 0, "pages": set(), "lang": c.lang_hint}
        doc_counts[c.doc_title]["chunks"] += 1
        doc_counts[c.doc_title]["pages"].add(c.page_num)
    table = Table(title="Ingestion summary")
    table.add_column("Document", style="cyan")
    table.add_column("Lang", justify="center")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    for title, meta in sorted(doc_counts.items()):
        table.add_row(title, meta["lang"], str(len(meta["pages"])), str(meta["chunks"]))
    console.print(table)
    console.print(f"\n[green]Total:[/green] {len(chunks)} chunks indexed in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceRAG corpus ingestion")
    parser.add_argument("--corpus", type=Path, default=Path(settings.CORPUS_PATH))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.corpus.exists():
        console.print(f"[red]Error:[/red] corpus directory not found: {args.corpus}")
        sys.exit(1)

    console.print(f"\n[bold]VoiceRAG ingestion[/bold] — corpus: {args.corpus}\n")
    init_query_log()

    if args.reset:
        console.print("[yellow]Resetting ChromaDB collection...[/yellow]")
        _reset_collection()

    t_start = time.perf_counter()

    console.print("Step 1/3  Loading PDFs...")
    pages = load_corpus(args.corpus)
    if not pages:
        console.print("[red]No pages loaded — check corpus directory.[/red]")
        sys.exit(1)
    console.print(f"          {len(pages)} pages loaded\n")

    console.print("Step 2/3  Chunking pages...")
    chunks = chunk_pages(pages)
    console.print(f"          {len(chunks)} chunks created\n")

    if args.dry_run:
        console.print("[yellow]Dry run — skipping upsert.[/yellow]")
        _print_summary(chunks, time.perf_counter() - t_start)
        return

    console.print("Step 3/3  Embedding + upserting to ChromaDB...")
    console.print("          (First run downloads ~560MB model)\n")
    embed_and_upsert(chunks)

    elapsed = time.perf_counter() - t_start
    stored = get_collection().count()
    console.print(f"\n[green]Verified:[/green] {stored} chunks in ChromaDB\n")
    _print_summary(chunks, elapsed)

    console.print("\nSpot check — querying 'MSME loan'...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    q_emb = model.encode(["query: MSME loan"], normalize_embeddings=True).tolist()
    results = get_collection().query(query_embeddings=q_emb, n_results=1)
    if results["documents"] and results["documents"][0]:
        top = results["documents"][0][0]
        meta = results["metadatas"][0][0]
        console.print(
            f"  [cyan]{meta['doc_title']}[/cyan] p.{meta['page_num']} — {top[:120].strip()}..."
        )
    console.print("\n[bold green]T2 complete. ChromaDB ready.[/bold green]\n")


if __name__ == "__main__":
    main()
