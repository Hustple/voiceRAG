"""T2 acceptance tests — loader, chunker, embedder (mocked)."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.chunker import CHUNK_SIZE, Chunk, chunk_pages
from ingestion.loader import RawPage, _clean_text, _derive_title


def _make_page(text: str, page_num: int = 1, lang: str = "en") -> RawPage:
    return RawPage(
        doc_title="Test Doc",
        source_path="/tmp/test.pdf",
        page_num=page_num,
        text=text,
        lang_hint=lang,
    )


def _long_text(n: int = 600) -> str:
    s = "The MSME sector is eligible for priority sector lending under RBI guidelines. "
    return "\n\n".join((s * 50).split(". ")[: n // 14])


class TestLoader:
    def test_derive_title_snake_case(self):
        assert _derive_title(Path("pm_mudra_yojana.pdf")) == "PM Mudra Yojana"

    def test_derive_title_short_words_uppercased(self):
        title = _derive_title(Path("rbi_sme_circular.pdf"))
        assert "RBI" in title and "SME" in title

    def test_clean_text_collapses_blank_lines(self):
        assert _clean_text("Hello\n\n\n\nWorld") == "Hello\n\nWorld"

    def test_clean_text_collapses_spaces(self):
        result = _clean_text("Hello   world\t\there")
        assert "   " not in result and "\t" not in result

    def test_clean_text_preserves_devanagari(self):
        assert "परीक्षण" in _clean_text("यह एक परीक्षण है।")

    def test_load_pdf_missing_file_returns_empty(self):
        from ingestion.loader import load_pdf

        assert load_pdf(Path("/nonexistent/file.pdf")) == []

    def test_load_corpus_empty_dir_returns_empty(self, tmp_path):
        from ingestion.loader import load_corpus

        assert load_corpus(tmp_path) == []


class TestChunker:
    def test_short_page_single_chunk(self):
        assert len(chunk_pages([_make_page("Short MSME text.")])) == 1

    def test_long_page_multiple_chunks(self):
        assert len(chunk_pages([_make_page(_long_text(600))])) > 1

    def test_chunks_within_size_limit(self):
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        for c in chunk_pages([_make_page(_long_text(600))]):
            assert len(enc.encode(c.text)) <= CHUNK_SIZE + 10

    def test_chunk_ids_unique(self):
        chunks = chunk_pages([_make_page(_long_text(600))])
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_deterministic(self):
        p = _make_page("Determinism check text.")
        assert chunk_pages([p])[0].chunk_id == chunk_pages([p])[0].chunk_id

    def test_chunk_carries_metadata(self):
        c = chunk_pages([_make_page("Text.", page_num=3, lang="hi")])[0]
        assert c.page_num == 3 and c.lang_hint == "hi"

    def test_chunk_indices_sequential(self):
        pages = [_make_page(f"Page {i} financial content.", i) for i in range(3)]
        chunks = chunk_pages(pages)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_empty_input_returns_empty(self):
        assert chunk_pages([]) == []


class TestEmbedder:
    @pytest.fixture
    def mock_model(self):
        import numpy as np

        m = MagicMock()
        m.encode.return_value = np.random.rand(1, 1024).astype("float32")
        return m

    @pytest.fixture
    def mock_col(self):
        c = MagicMock()
        c.upsert.return_value = None
        return c

    def _chunk(self) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_title="Test",
            source_path="/tmp/t.pdf",
            page_num=1,
            lang_hint="en",
            text="MSME loan eligibility.",
            token_count=5,
            chunk_index=0,
        )

    def test_upsert_called(self, mock_model, mock_col):
        from ingestion.embedder import embed_and_upsert

        with (
            patch("ingestion.embedder._load_model", return_value=mock_model),
            patch("ingestion.embedder.get_collection", return_value=mock_col),
        ):
            assert embed_and_upsert([self._chunk()]) == 1
        mock_col.upsert.assert_called_once()

    def test_passage_prefix_applied(self, mock_model, mock_col):
        from ingestion.embedder import E5_PASSAGE_PREFIX, embed_and_upsert

        with (
            patch("ingestion.embedder._load_model", return_value=mock_model),
            patch("ingestion.embedder.get_collection", return_value=mock_col),
        ):
            embed_and_upsert([self._chunk()])
        texts = mock_model.encode.call_args[0][0]
        assert all(t.startswith(E5_PASSAGE_PREFIX) for t in texts)

    def test_empty_returns_zero(self):
        from ingestion.embedder import embed_and_upsert

        assert embed_and_upsert([]) == 0

    def test_metadata_fields_present(self, mock_model, mock_col):
        from ingestion.embedder import embed_and_upsert

        with (
            patch("ingestion.embedder._load_model", return_value=mock_model),
            patch("ingestion.embedder.get_collection", return_value=mock_col),
        ):
            embed_and_upsert([self._chunk()])
        meta = mock_col.upsert.call_args[1]["metadatas"][0]
        for key in ("doc_title", "page_num", "lang_hint", "token_count", "chunk_index"):
            assert key in meta
