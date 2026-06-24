"""
CI-safe unit tests for dedupe.py pure function.

Covers:
  - empty input
  - no-duplicate passthrough
  - duplicate _id removal (first occurrence wins)
  - insertion-order preservation
  - records missing _id are always retained (even multiple)
  - mixed id / no-id records

Zero network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.services.dedupe import dedupe_records  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(id_: "str | None", **kwargs: Any) -> Dict[str, Any]:
    """Build a minimal record dict. id_=None omits the '_id' key entirely."""
    rec: Dict[str, Any] = {}
    if id_ is not None:
        rec["_id"] = id_
    rec.update(kwargs)
    return rec


# ---------------------------------------------------------------------------
# TestDedupeRecords
# ---------------------------------------------------------------------------

class TestDedupeRecords:
    def test_empty_input_returns_empty(self):
        assert dedupe_records([]) == []

    def test_no_duplicates_returns_all_records(self):
        records = [_r("a"), _r("b"), _r("c")]
        assert dedupe_records(records) == records

    def test_single_record_returned_unchanged(self):
        record = _r("only", content="hello")
        assert dedupe_records([record]) == [record]

    def test_duplicate_id_keeps_first_occurrence(self):
        records = [_r("a", val=1), _r("a", val=2), _r("b", val=3)]
        result = dedupe_records(records)
        assert len(result) == 2
        assert result[0]["val"] == 1  # first occurrence kept
        assert result[1]["_id"] == "b"

    def test_three_dupes_of_same_id_keeps_one(self):
        records = [_r("x", n=1), _r("x", n=2), _r("x", n=3)]
        result = dedupe_records(records)
        assert len(result) == 1
        assert result[0]["n"] == 1

    def test_preserves_insertion_order(self):
        records = [_r("c"), _r("a"), _r("b")]
        result = dedupe_records(records)
        assert [r["_id"] for r in result] == ["c", "a", "b"]

    def test_records_without_id_are_always_retained(self):
        records = [_r(None, val="no-id-1"), _r(None, val="no-id-2")]
        result = dedupe_records(records)
        assert len(result) == 2

    def test_records_without_id_not_treated_as_duplicates_of_each_other(self):
        records = [_r(None, x=1), _r(None, x=2), _r(None, x=3)]
        assert len(dedupe_records(records)) == 3

    def test_mixed_id_and_no_id_records(self):
        records = [_r("a"), _r(None, x=1), _r("a"), _r(None, x=2)]
        result = dedupe_records(records)
        # "a" deduped to 1; both no-id records kept
        assert len(result) == 3
        id_records = [r for r in result if "_id" in r]
        no_id_records = [r for r in result if "_id" not in r]
        assert len(id_records) == 1
        assert len(no_id_records) == 2

    def test_does_not_mutate_input_list(self):
        records = [_r("a")]
        result = dedupe_records(records)
        result.append(_r("b"))
        assert len(records) == 1  # original list untouched
