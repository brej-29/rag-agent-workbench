"""
CI-safe unit tests for eval/corpus_manifest.py (T3-B / P1).

Only tests the pure-function helpers — no Pinecone calls, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR = _REPO_ROOT / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from corpus_manifest import compute_drift, parse_doc_id  # noqa: E402


class TestParseDocId:
    def test_strips_chunk_index(self):
        assert parse_doc_id("abc123:0") == "abc123"

    def test_strips_non_zero_chunk_index(self):
        assert parse_doc_id("abc123:17") == "abc123"

    def test_sha256_id_with_chunk(self):
        sha = "eaf004fec892968d5561c536c50eb1b0c12fd5fb08e18607ef9ed0666bb4a975"
        assert parse_doc_id(f"{sha}:0") == sha
        assert parse_doc_id(f"{sha}:99") == sha

    def test_id_without_colon_returns_whole_string(self):
        assert parse_doc_id("nodcolon") == "nodcolon"

    def test_only_last_colon_is_stripped(self):
        # doc_id could theoretically contain colons (though ours don't)
        assert parse_doc_id("a:b:3") == "a:b"


class TestComputeDrift:
    def test_no_drift(self):
        ids = {"a", "b", "c"}
        drift = compute_drift(ids, ids)
        assert drift["missing"] == []
        assert drift["extra"] == []

    def test_missing_from_live(self):
        manifest = {"a", "b", "c"}
        live = {"a", "b"}
        drift = compute_drift(manifest, live)
        assert drift["missing"] == ["c"]
        assert drift["extra"] == []

    def test_extra_in_live(self):
        manifest = {"a"}
        live = {"a", "b", "c"}
        drift = compute_drift(manifest, live)
        assert drift["missing"] == []
        assert sorted(drift["extra"]) == ["b", "c"]

    def test_both_missing_and_extra(self):
        manifest = {"a", "b"}
        live = {"b", "c"}
        drift = compute_drift(manifest, live)
        assert drift["missing"] == ["a"]
        assert drift["extra"] == ["c"]

    def test_both_empty(self):
        drift = compute_drift(set(), set())
        assert drift["missing"] == []
        assert drift["extra"] == []

    def test_output_is_sorted(self):
        manifest = {"z", "m", "a"}
        live = {"z", "q", "b"}
        drift = compute_drift(manifest, live)
        assert drift["missing"] == sorted(drift["missing"])
        assert drift["extra"] == sorted(drift["extra"])


class TestCollectAllVectorIds:
    def test_pinecone_v10_list_items_become_id_strings(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from eval.corpus_manifest import _collect_all_vector_ids

        index = MagicMock()
        # pinecone>=10: list() yields pages of ListItem objects with an .id attribute
        index.list.return_value = iter([[SimpleNamespace(id="a:0"), SimpleNamespace(id="a:1")],
                                        [SimpleNamespace(id="b:0")]])
        assert _collect_all_vector_ids(index, "eval") == ["a:0", "a:1", "b:0"]

    def test_plain_string_pages_still_work(self):
        from unittest.mock import MagicMock

        from eval.corpus_manifest import _collect_all_vector_ids

        index = MagicMock()
        index.list.return_value = iter([["a:0", "b:0"]])
        assert _collect_all_vector_ids(index, "eval") == ["a:0", "b:0"]
