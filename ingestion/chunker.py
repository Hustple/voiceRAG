"""
Chunker — splits RawPage text into overlapping token-window chunks.

Uses tiktoken cl100k_base for token counting.
Splits on paragraph → sentence boundaries before falling back to sliding window.
E5 "passage:" prefix is applied at embed time, not here.
"""
from __future__ import annotations
import re
import uuid
from dataclasses import dataclass
import tiktoken
from ingestion.loader import RawPage

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

_enc = tiktoken.get_encoding("cl100k_base")

@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    source_path: str
    page_num: int
    lang_hint: str
    text: str
    token_count: int
    chunk_index: int

def _token_count(text: str) -> int:
    return len(_enc.encode(text))

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[।.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def _chunk_text(text: str) -> list[str]:
    if _token_count(text) <= CHUNK_SIZE:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_tokens: list[int] = []
    for para in paragraphs:
        para_tokens = _enc.encode(para)
        if len(para_tokens) > CHUNK_SIZE:
            for sent in _split_sentences(para):
                sent_tokens = _enc.encode(sent)
                if len(current_tokens) + len(sent_tokens) > CHUNK_SIZE:
                    if current_tokens:
                        chunks.append(_enc.decode(current_tokens))
                    current_tokens = current_tokens[-CHUNK_OVERLAP:] + sent_tokens
                else:
                    current_tokens.extend(sent_tokens)
        else:
            if len(current_tokens) + len(para_tokens) > CHUNK_SIZE:
                if current_tokens:
                    chunks.append(_enc.decode(current_tokens))
                current_tokens = current_tokens[-CHUNK_OVERLAP:] + para_tokens
            else:
                current_tokens.extend(para_tokens)
    if current_tokens:
        chunks.append(_enc.decode(current_tokens))
    return chunks if chunks else [text]

def chunk_pages(pages: list[RawPage]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    global_index = 0
    for page in pages:
        for local_idx, chunk_text in enumerate(_chunk_text(page.text)):
            chunk_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{page.source_path}::{page.page_num}::{local_idx}",
            ))
            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_title=page.doc_title,
                source_path=page.source_path,
                page_num=page.page_num,
                lang_hint=page.lang_hint,
                text=chunk_text,
                token_count=_token_count(chunk_text),
                chunk_index=global_index,
            ))
            global_index += 1
    return all_chunks
