"""
CI-safe unit tests for eval/metrics.py.

ALL tests run with zero network calls — no Pinecone, no Groq, no embedder.

Two test groups:
  1. Unit tests of the three metric functions against hand-computed expected
     values on small synthetic inputs.
  2. Fixture-based tests that load eval/fixtures/retrieval_fixture.json
     (committed synthetic data) and assert the metric functions produce the
     expected values for each recorded case.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

# Make sure repo root is on sys.path so `from eval.metrics import ...` works
# regardless of how pytest was invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.metrics import ndcg_at_k, mrr, recall_at_k  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture() -> list[dict]:
    fixture_path = _REPO_ROOT / "eval" / "fixtures" / "retrieval_fixture.json"
    with fixture_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["queries"]


# ---------------------------------------------------------------------------
# 1. Unit tests — recall@k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_all_relevant_found(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "c"}
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_partial_relevant_found(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "z"}  # "z" not retrieved
        assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(0.5)

    def test_cutoff_limits_consideration(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"e"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.0  # "e" is at rank 5
        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_none_found(self):
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), k=5) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], {"a"}, k=5) == 0.0

    def test_k_larger_than_retrieved(self):
        retrieved = ["a", "b"]
        relevant = {"a"}
        # k=10 but only 2 retrieved; "a" is found → recall = 1/1 = 1.0
        assert recall_at_k(retrieved, relevant, k=10) == 1.0


# ---------------------------------------------------------------------------
# 2. Unit tests — MRR
# ---------------------------------------------------------------------------

class TestMRR:
    def test_first_hit_at_rank_1(self):
        assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_first_hit_at_rank_2(self):
        assert mrr(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_first_hit_at_rank_3(self):
        assert mrr(["x", "y", "z"], {"z"}) == pytest.approx(1.0 / 3.0)

    def test_no_hit(self):
        assert mrr(["a", "b", "c"], {"missing"}) == 0.0

    def test_empty_retrieved(self):
        assert mrr([], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert mrr(["a", "b"], set()) == 0.0

    def test_multiple_relevant_uses_first(self):
        # "b" at rank 2, "c" at rank 3 — should use rank 2
        assert mrr(["a", "b", "c"], {"b", "c"}) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3. Unit tests — nDCG@k
# ---------------------------------------------------------------------------

class TestNDCGAtK:
    def test_perfect_ranking(self):
        # All relevant docs at top positions → nDCG = 1.0
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        result = ndcg_at_k(retrieved, relevant, k=4)
        assert result == pytest.approx(1.0)

    def test_single_relevant_at_rank_1(self):
        assert ndcg_at_k(["a", "b", "c"], {"a"}, k=3) == pytest.approx(1.0)

    def test_single_relevant_at_rank_3(self):
        # DCG = 1/log2(4) = 0.5; IDCG = 1/log2(2) = 1.0 → nDCG = 0.5
        result = ndcg_at_k(["x", "y", "z"], {"z"}, k=3)
        assert result == pytest.approx(0.5)

    def test_relevant_at_positions_1_and_3(self):
        # DCG@5 = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG@5 = 1/log2(2) + 1/log2(3) = 1.0 + 1/log2(3)
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c"}
        expected = 1.5 / (1.0 + 1.0 / math.log2(3))
        assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(expected, rel=1e-9)

    def test_no_relevant_found(self):
        assert ndcg_at_k(["a", "b"], {"missing"}, k=5) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), k=5) == 0.0

    def test_empty_retrieved(self):
        assert ndcg_at_k([], {"a"}, k=5) == 0.0

    def test_k_larger_than_retrieved(self):
        # Only 2 docs retrieved; k=10. Relevant "a" at rank 1 → perfect score.
        assert ndcg_at_k(["a", "b"], {"a"}, k=10) == pytest.approx(1.0)

    def test_value_bounded_at_one(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        result = ndcg_at_k(retrieved, relevant, k=3)
        assert 0.0 <= result <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 4. Fixture-based tests (zero network calls)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_queries():
    return _load_fixture()


class TestFixtureMetrics:
    """
    Loads eval/fixtures/retrieval_fixture.json (committed synthetic data)
    and asserts the metric functions produce the correct values.

    Hand-computed expected values are documented inline.
    """

    def test_fixture_q1_recall(self, fixture_queries):
        """Strong retrieval: doc_rag_001 (rank 1) and doc_rag_003 (rank 3) both relevant."""
        q = next(e for e in fixture_queries if e["id"] == "fixture_q1")
        result = recall_at_k(q["retrieved_doc_ids"], set(q["relevant_doc_ids"]), k=q["k"])
        # Both relevant docs found in top 5 → 2/2 = 1.0
        assert result == pytest.approx(1.0)

    def test_fixture_q1_mrr(self, fixture_queries):
        q = next(e for e in fixture_queries if e["id"] == "fixture_q1")
        result = mrr(q["retrieved_doc_ids"], set(q["relevant_doc_ids"]))
        # doc_rag_001 is at rank 1 → RR = 1/1 = 1.0
        assert result == pytest.approx(1.0)

    def test_fixture_q1_ndcg(self, fixture_queries):
        q = next(e for e in fixture_queries if e["id"] == "fixture_q1")
        result = ndcg_at_k(q["retrieved_doc_ids"], set(q["relevant_doc_ids"]), k=q["k"])
        # DCG@5  = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG@5 = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309... = 1.6309...
        # nDCG@5 = 1.5 / 1.6309... ≈ 0.91972
        expected = 1.5 / (1.0 + 1.0 / math.log2(3))
        assert result == pytest.approx(expected, rel=1e-9)

    def test_fixture_q2_recall(self, fixture_queries):
        """Late hit: single relevant doc at rank 3."""
        q = next(e for e in fixture_queries if e["id"] == "fixture_q2")
        result = recall_at_k(q["retrieved_doc_ids"], set(q["relevant_doc_ids"]), k=q["k"])
        # doc_ann_012 found at rank 3; 1/1 = 1.0
        assert result == pytest.approx(1.0)

    def test_fixture_q2_mrr(self, fixture_queries):
        q = next(e for e in fixture_queries if e["id"] == "fixture_q2")
        result = mrr(q["retrieved_doc_ids"], set(q["relevant_doc_ids"]))
        # First relevant at rank 3 → RR = 1/3
        assert result == pytest.approx(1.0 / 3.0)

    def test_fixture_q2_ndcg(self, fixture_queries):
        q = next(e for e in fixture_queries if e["id"] == "fixture_q2")
        result = ndcg_at_k(q["retrieved_doc_ids"], set(q["relevant_doc_ids"]), k=q["k"])
        # DCG@3  = 1/log2(4) = 0.5  (doc_ann_012 at position 3)
        # IDCG@3 = 1/log2(2) = 1.0  (ideal: 1 relevant at position 1)
        # nDCG@3 = 0.5 / 1.0 = 0.5
        assert result == pytest.approx(0.5)

    def test_fixture_q3_all_zero(self, fixture_queries):
        """Complete miss: relevant doc not in retrieved list."""
        q = next(e for e in fixture_queries if e["id"] == "fixture_q3")
        relevant = set(q["relevant_doc_ids"])
        retrieved = q["retrieved_doc_ids"]
        assert recall_at_k(retrieved, relevant, k=q["k"]) == pytest.approx(0.0)
        assert mrr(retrieved, relevant) == pytest.approx(0.0)
        assert ndcg_at_k(retrieved, relevant, k=q["k"]) == pytest.approx(0.0)

    def test_fixture_q4_partial_recall(self, fixture_queries):
        """Partial hit: one of two relevant docs found at rank 2."""
        q = next(e for e in fixture_queries if e["id"] == "fixture_q4")
        result = recall_at_k(q["retrieved_doc_ids"], set(q["relevant_doc_ids"]), k=q["k"])
        # doc_tf_031 found at rank 2; doc_tf_missing not found → 1/2 = 0.5
        assert result == pytest.approx(0.5)

    def test_fixture_q4_mrr(self, fixture_queries):
        q = next(e for e in fixture_queries if e["id"] == "fixture_q4")
        result = mrr(q["retrieved_doc_ids"], set(q["relevant_doc_ids"]))
        # First relevant hit at rank 2 → RR = 0.5
        assert result == pytest.approx(0.5)

    def test_fixture_q4_ndcg(self, fixture_queries):
        q = next(e for e in fixture_queries if e["id"] == "fixture_q4")
        result = ndcg_at_k(q["retrieved_doc_ids"], set(q["relevant_doc_ids"]), k=q["k"])
        # retrieved = [doc_ann_030, doc_tf_031, doc_vec_032, doc_tf_033]; k=4
        # relevant  = {doc_tf_031, doc_tf_missing}
        # doc_tf_031 at position 2, doc_tf_missing not found
        # DCG@4  = 1/log2(3) ≈ 0.6309
        # IDCG@4 = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309  (2 relevant, ideal at pos 1+2)
        # nDCG@4 = 0.6309 / 1.6309 ≈ 0.3868
        dcg = 1.0 / math.log2(3)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = dcg / idcg
        assert result == pytest.approx(expected, rel=1e-9)
