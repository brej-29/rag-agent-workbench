"""
CI-safe unit tests for normalize.py pure functions.

Covers:
  - normalize_text: whitespace collapsing, edge stripping, None safety
  - is_valid_document: minimum-length gate
  - make_doc_id: SHA-256 determinism and collision-distinctness

Zero network calls.  make_doc_id is the eval reproducibility anchor:
same (source, title, url) must always produce the same id so that
golden.jsonl relevant_doc_ids stay valid across reruns.
"""

from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.services.normalize import is_valid_document, make_doc_id, normalize_text  # noqa: E402


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_collapses_multiple_spaces(self):
        assert normalize_text("foo   bar") == "foo bar"

    def test_collapses_newlines(self):
        assert normalize_text("foo\nbar\nbaz") == "foo bar baz"

    def test_collapses_tabs(self):
        assert normalize_text("foo\t\tbar") == "foo bar"

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize_text("  hello world  ") == "hello world"

    def test_mixed_whitespace_types(self):
        assert normalize_text("  foo\n\n   bar\t baz  ") == "foo bar baz"

    def test_empty_string_returns_empty(self):
        assert normalize_text("") == ""

    def test_none_input_treated_as_empty(self):
        # normalize_text uses `text or ""` so None is safe at runtime
        assert normalize_text(None) == ""  # type: ignore[arg-type]

    def test_already_normalized_string_unchanged(self):
        assert normalize_text("hello world") == "hello world"

    def test_only_whitespace_becomes_empty(self):
        assert normalize_text("   \n\t   ") == ""

    def test_single_newline_becomes_space(self):
        assert normalize_text("line1\nline2") == "line1 line2"


# ---------------------------------------------------------------------------
# is_valid_document
# ---------------------------------------------------------------------------

class TestIsValidDocument:
    def test_empty_string_is_invalid(self):
        assert is_valid_document("") is False

    def test_short_text_below_threshold_is_invalid(self):
        assert is_valid_document("too short") is False

    def test_exactly_default_min_chars_is_valid(self):
        assert is_valid_document("x" * 200) is True

    def test_one_below_default_min_chars_is_invalid(self):
        assert is_valid_document("x" * 199) is False

    def test_above_default_threshold_is_valid(self):
        assert is_valid_document("x" * 500) is True

    def test_custom_min_chars_met(self):
        assert is_valid_document("hello", min_chars=5) is True

    def test_custom_min_chars_not_met(self):
        assert is_valid_document("hell", min_chars=5) is False


# ---------------------------------------------------------------------------
# make_doc_id — determinism + collision-distinctness
# ---------------------------------------------------------------------------

class TestMakeDocId:
    def test_returns_64_char_hex_string(self):
        doc_id = make_doc_id("wiki", "Quantum computing")
        assert len(doc_id) == 64
        assert all(c in "0123456789abcdef" for c in doc_id)

    def test_determinism_same_inputs_same_id(self):
        id1 = make_doc_id("wiki", "Quantum computing", "https://en.wikipedia.org/wiki/Q")
        id2 = make_doc_id("wiki", "Quantum computing", "https://en.wikipedia.org/wiki/Q")
        assert id1 == id2

    def test_determinism_no_url(self):
        id1 = make_doc_id("arxiv", "Attention Is All You Need")
        id2 = make_doc_id("arxiv", "Attention Is All You Need")
        assert id1 == id2

    def test_collision_distinctness_different_source(self):
        id1 = make_doc_id("wiki", "Neural network", "https://example.com")
        id2 = make_doc_id("arxiv", "Neural network", "https://example.com")
        assert id1 != id2

    def test_collision_distinctness_different_title(self):
        id1 = make_doc_id("wiki", "Title A", "https://example.com")
        id2 = make_doc_id("wiki", "Title B", "https://example.com")
        assert id1 != id2

    def test_collision_distinctness_different_url(self):
        id1 = make_doc_id("wiki", "Same Title", "https://url-a.com")
        id2 = make_doc_id("wiki", "Same Title", "https://url-b.com")
        assert id1 != id2

    def test_url_none_and_url_empty_are_equivalent(self):
        # make_doc_id uses `url or ''`, so None and "" produce identical base strings
        assert make_doc_id("wiki", "Title", None) == make_doc_id("wiki", "Title", "")

    def test_matches_manual_sha256_with_url(self):
        source, title, url = "wiki", "Test Article", "https://example.com"
        expected = sha256(f"{source}|{title}|{url}".encode("utf-8")).hexdigest()
        assert make_doc_id(source, title, url) == expected

    def test_matches_manual_sha256_no_url(self):
        source, title = "arxiv", "Some Paper"
        expected = sha256(f"{source}|{title}|".encode("utf-8")).hexdigest()
        assert make_doc_id(source, title) == expected

    def test_different_source_url_combo_all_distinct(self):
        ids = {
            make_doc_id("wiki", "T", "https://a.com"),
            make_doc_id("wiki", "T", "https://b.com"),
            make_doc_id("arxiv", "T", "https://a.com"),
            make_doc_id("openalex", "T", "https://a.com"),
        }
        assert len(ids) == 4  # all four must be distinct
