"""
CI-safe unit tests for chunking.py.

Covers:
  - chunk_document: single-chunk short text, multi-chunk long text, chunk_size /
    overlap respected, returns List[Document], empty content
  - documents_to_records: record schema, _id format, chunk_id sequence, text_field
    key, missing-metadata skip, safety truncation, multiple documents

All tests mock app.services.chunking.get_settings to avoid requiring environment
variables (PINECONE_API_KEY etc.).  Zero network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from langchain_core.documents import Document  # noqa: E402

from app.services.chunking import (  # noqa: E402
    MAX_CHARS_PER_CHUNK,
    chunk_document,
    documents_to_records,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(text_field: str = "chunk_text") -> MagicMock:
    s = MagicMock()
    s.PINECONE_TEXT_FIELD = text_field
    return s


# ~1 100 chars per paragraph; five paragraphs → ~5 500 chars total
_LONG_TEXT = ("The quick brown fox jumps over the lazy dog. " * 25 + "\n\n") * 5


def _doc(
    content: str,
    doc_id: str = "test-doc-id",
    source: str = "wiki",
    title: str = "Test Title",
    url: str = "https://example.com",
) -> Document:
    return Document(
        page_content=content,
        metadata={"doc_id": doc_id, "source": source, "title": title, "url": url},
    )


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------

class TestChunkDocument:
    def test_short_text_produces_single_chunk(self):
        doc = Document(page_content="A short abstract.", metadata={})
        assert len(chunk_document(doc)) == 1

    def test_long_text_produces_multiple_chunks(self):
        doc = Document(page_content=_LONG_TEXT, metadata={})
        chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

    def test_returns_list_of_documents(self):
        doc = Document(page_content="Hello world.", metadata={})
        chunks = chunk_document(doc)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunk_size_respected_approximately(self):
        doc = Document(page_content=_LONG_TEXT, metadata={})
        chunk_size = 300
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=0)
        # RecursiveCharacterTextSplitter may slightly exceed on word boundaries;
        # allow a small margin (one extra word)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 80

    def test_overlap_increases_chunk_count(self):
        doc = Document(page_content=_LONG_TEXT, metadata={})
        no_overlap = chunk_document(doc, chunk_size=500, chunk_overlap=0)
        with_overlap = chunk_document(doc, chunk_size=500, chunk_overlap=200)
        assert len(with_overlap) >= len(no_overlap)

    def test_empty_content_returns_list(self):
        doc = Document(page_content="", metadata={})
        chunks = chunk_document(doc)
        assert isinstance(chunks, list)

    def test_metadata_not_required_in_document(self):
        doc = Document(page_content="Some content here.", metadata={})
        chunks = chunk_document(doc)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# documents_to_records
# ---------------------------------------------------------------------------

class TestDocumentsToRecords:
    # ------ basic schema ------

    def test_empty_document_list(self):
        with patch("app.services.chunking.get_settings", return_value=_settings()):
            records = documents_to_records([])
        assert records == []

    def test_short_document_produces_one_record(self):
        with patch("app.services.chunking.get_settings", return_value=_settings()):
            records = documents_to_records([_doc("Short content.")])
        assert len(records) == 1

    def test_record_id_format_is_doc_id_colon_index(self):
        with patch("app.services.chunking.get_settings", return_value=_settings()):
            records = documents_to_records([_doc("Short.", doc_id="abc123")])
        assert records[0]["_id"] == "abc123:0"

    def test_default_text_field_key_present(self):
        with patch("app.services.chunking.get_settings", return_value=_settings("chunk_text")):
            records = documents_to_records([_doc("Content.")])
        assert "chunk_text" in records[0]

    def test_custom_text_field_name_used(self):
        with patch("app.services.chunking.get_settings", return_value=_settings("content")):
            records = documents_to_records([_doc("Content.")])
        assert "content" in records[0]
        assert "chunk_text" not in records[0]

    def test_metadata_fields_copied_to_record(self):
        with patch("app.services.chunking.get_settings", return_value=_settings()):
            records = documents_to_records([
                _doc("Text.", doc_id="d1", source="arxiv", title="My Paper", url="https://arxiv.org/1")
            ])
        rec = records[0]
        assert rec["doc_id"] == "d1"
        assert rec["source"] == "arxiv"
        assert rec["title"] == "My Paper"
        assert rec["url"] == "https://arxiv.org/1"

    def test_chunk_id_is_int(self):
        with patch("app.services.chunking.get_settings", return_value=_settings()):
            records = documents_to_records([_doc("Text.")])
        assert isinstance(records[0]["chunk_id"], int)

    # ------ skip logic ------

    def test_document_missing_doc_id_is_skipped(self):
        doc = Document(page_content="Content.", metadata={"source": "wiki", "title": "T"})
        with patch("app.services.chunking.get_settings", return_value=_settings()):
            records = documents_to_records([doc])
        assert records == []

    def test_document_missing_source_is_skipped(self):
        doc = Document(page_content="Content.", metadata={"doc_id": "x", "title": "T"})
        with patch("app.services.chunking.get_settings", return_value=_settings()):
            records = documents_to_records([doc])
        assert records == []

    # ------ multi-chunk behaviour ------

    def test_long_document_produces_multiple_records(self):
        with patch("app.services.chunking.get_settings", return_value=_settings()):
            records = documents_to_records([_doc(_LONG_TEXT)])
        assert len(records) > 1

    def test_chunk_ids_are_sequential(self):
        with patch("app.services.chunking.get_settings", return_value=_settings()):
            records = documents_to_records([_doc(_LONG_TEXT)])
        assert [r["chunk_id"] for r in records] == list(range(len(records)))

    def test_record_ids_embed_chunk_index(self):
        with patch("app.services.chunking.get_settings", return_value=_settings()):
            records = documents_to_records([_doc(_LONG_TEXT, doc_id="parent")])
        for i, rec in enumerate(records):
            assert rec["_id"] == f"parent:{i}"

    # ------ multiple documents ------

    def test_multiple_documents_all_produce_records(self):
        doc1 = _doc("Text A.", doc_id="d1")
        doc2 = _doc("Text B.", doc_id="d2")
        with patch("app.services.chunking.get_settings", return_value=_settings()):
            records = documents_to_records([doc1, doc2])
        doc_ids = {r["doc_id"] for r in records}
        assert "d1" in doc_ids
        assert "d2" in doc_ids

    # ------ safety truncation ------

    def test_safety_truncation_applied_to_oversized_chunk(self):
        oversized_text = "z" * (MAX_CHARS_PER_CHUNK + 100)
        oversized_chunk = Document(page_content=oversized_text, metadata={})

        with patch("app.services.chunking.get_settings", return_value=_settings()):
            with patch("app.services.chunking.chunk_document", return_value=[oversized_chunk]):
                records = documents_to_records([_doc("Normal input.")])

        assert len(records) == 1
        assert len(records[0]["chunk_text"]) == MAX_CHARS_PER_CHUNK

    def test_chunk_text_within_limit_is_not_truncated(self):
        normal_text = "y" * (MAX_CHARS_PER_CHUNK - 100)
        normal_chunk = Document(page_content=normal_text, metadata={})

        with patch("app.services.chunking.get_settings", return_value=_settings()):
            with patch("app.services.chunking.chunk_document", return_value=[normal_chunk]):
                records = documents_to_records([_doc("Normal input.")])

        assert records[0]["chunk_text"] == normal_text
