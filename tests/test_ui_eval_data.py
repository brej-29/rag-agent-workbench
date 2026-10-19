"""Evaluation page data shaping (pure, no Streamlit)."""
from frontend.services.eval_data import abstention_counts, ab_deltas, load_results, sweep_rows


def test_load_results_reads_committed_snapshots():
    results = load_results()
    assert {"retrieval", "sweep", "rerank_ab", "answers", "adversarial"} <= set(results)


def test_sweep_rows_are_long_format_per_metric():
    sweep = {"aggregate": {"5": {"mean_recall_at_k": 0.91, "mean_ndcg_at_k": 0.90, "mean_precision_at_k": 0.36},
                           "10": {"mean_recall_at_k": 0.98, "mean_ndcg_at_k": 0.93, "mean_precision_at_k": 0.2}}}
    rows = sweep_rows(sweep)
    assert {"k": 5, "metric": "Recall@k", "value": 0.91} in rows
    assert [r["k"] for r in rows if r["metric"] == "Precision@k"] == [5, 10]


def test_ab_deltas_exclude_latency_which_lives_on_another_scale():
    ab = {"aggregate": {"baseline": {"mean_ndcg_at_3": 0.875, "mean_mrr": 0.98, "mean_latency_ms": 360},
                        "rerank": {"mean_ndcg_at_3": 0.818, "mean_mrr": 0.98, "mean_latency_ms": 795}}}
    deltas, latency = ab_deltas(ab)
    assert {"metric": "nDCG@3", "delta": -0.057} in deltas
    assert all(d["metric"] != "latency" for d in deltas) and latency == 435


def test_abstention_counts():
    rows = [{"answerable": True, "abstained": False}, {"answerable": True, "abstained": True},
            {"answerable": False, "abstained": True}, {"answerable": False, "abstained": False, "error": "429"}]
    assert abstention_counts(rows) == {"answered_ok": 1, "false_abstain": 1, "correct_abstain": 1,
                                       "answered_unanswerable": 0, "errors": 1}
