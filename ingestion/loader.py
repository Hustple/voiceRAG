"""
PDF loader using PyMuPDF (fitz).
Returns one RawPage per PDF page with doc metadata attached.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import fitz
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class RawPage:
    doc_title: str
    source_path: str
    page_num: int
    text: str
    lang_hint: str = "en"

def _derive_title(path: Path) -> str:
    stem = path.stem.replace("-", "_")
    words = stem.split("_")
    return " ".join(w.upper() if len(w) <= 3 else w.title() for w in words)

def _clean_text(raw: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", raw)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()

def load_pdf(path: Path, lang_hint: str = "en") -> list[RawPage]:
    doc_title = _derive_title(path)
    pages: list[RawPage] = []
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.error("loader.open_failed", path=str(path), error=str(exc))
        return pages
    logger.info("loader.loading", title=doc_title, total_pages=len(doc))
    for i, page in enumerate(doc):
        cleaned = _clean_text(page.get_text("text"))
        if len(cleaned) < 50:
            continue
        pages.append(RawPage(
            doc_title=doc_title,
            source_path=str(path.resolve()),
            page_num=i + 1,
            text=cleaned,
            lang_hint=lang_hint,
        ))
    doc.close()
    logger.info("loader.done", title=doc_title, pages_loaded=len(pages))
    return pages

def load_corpus(corpus_dir: Path) -> list[RawPage]:
    pdf_paths = sorted(corpus_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("loader.no_pdfs_found", corpus_dir=str(corpus_dir))
        return []
    all_pages: list[RawPage] = []
    for path in pdf_paths:
        lang_hint = "hi" if path.stem.endswith("_hi") else "en"
        all_pages.extend(load_pdf(path, lang_hint=lang_hint))
    logger.info("loader.corpus_loaded", total_pages=len(all_pages), docs=len(pdf_paths))
    return all_pages
