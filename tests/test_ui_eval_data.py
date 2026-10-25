"""Evaluation page data shaping (pure, no Streamlit)."""
from frontend.services.eval_data import (ab_deltas, abstention_counts, confusion_rows, coverage_rows, gap_range,
                                         disagreement_line, headline, job_name, load_results, mode_rows, span,
                                         sweep_rows, type_rows)


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



def test_committed_snapshots_include_the_compliance_evals():
    assert {"regs_retrieval", "regs_modes", "regs_abstain", "router", "gapcheck", "corpus"} <= set(load_results())


def test_mode_rows_keep_a_fixed_order_and_mark_production():
    rows = mode_rows({"diversified-inject": {"n": 50, "recall": 0.8713, "mrr": 0.8457},
                      "lexical": {"n": 50, "recall": 0.61, "mrr": 0.44},
                      "diversified": {"n": 50, "recall": 0.8713, "mrr": 0.8457}})
    assert [r["mode"] for r in rows] == ["Lexical (BM25)", "Diversified (production)", "+ crosswalk inject"]
    assert [r["production"] for r in rows] == [False, True, False] and rows[1]["recall"] == 0.871


def test_type_rows_name_question_types_with_counts():
    ret = {"k": 6, "aggregate": {"by_type": {"cross": {"n": 6, "recall": 0.4833, "mrr": 0.74, "ndcg": 0.46}}}}
    assert type_rows(ret) == [{"Question type": "Across two frameworks", "Questions": 6, "Recall@6": 0.483,
                               "MRR": 0.74, "nDCG@6": 0.46}]


def test_confusion_rows_are_a_request_by_job_table_in_plain_names():
    rows = confusion_rows({"ask->ask": 17, "gap_check->ask": 2, "gap_check->gap_check": 3})
    assert rows == [{"Request": "Ask", "Ask": 17, "Gap check": 0}, {"Request": "Gap check", "Ask": 2, "Gap check": 3}]
    assert job_name("clarify") == "Clarify" and job_name("other") == "other"


def test_disagreement_line_reads_as_a_sentence():
    assert disagreement_line("incident_injection:gdpr/art-34/par-1 expected partial, got unverified") == (
        "Incident injection · `gdpr/art-34/par-1`: labelled partial, judged unverified")
    assert disagreement_line("free text") == "free text"


def test_gap_range_spans_every_run_and_falls_back_to_the_latest():
    gap = {"accuracy": 0.765, "verified_rate": 0.824, "false_covered": [],
           "runs": [{"accuracy": 0.765, "verified_rate": 0.824, "false_covered": 0},
                    {"accuracy": 0.882, "verified_rate": 1.0, "false_covered": 0}]}
    assert gap_range(gap) == {"runs": 2, "accuracy": (0.765, 0.882), "verified_rate": (0.824, 1.0), "false_covered": 0}
    assert gap_range({k: v for k, v in gap.items() if k != "runs"})["runs"] == 1
    assert span((0.765, 0.882)) == "0.77–0.88" and span((1.0, 1.0), pct=True) == "100%"


def test_headline_skips_missing_snapshots():
    out = headline({"router": {"n": 36, "accuracy": 0.944, "confusion": {}, "misses": []}})
    assert out == [{"label": "Requests routed to the right job", "value": "94%", "note": "36 labelled requests"}]


def test_coverage_rows_say_binding_or_voluntary():
    rows = coverage_rows([{"key": "nis2", "name": "NIS2", "binding": "directive", "provisions": 349, "url": "u",
                           "retrieved": "2026-10-20"}])
    assert rows == [{"Instrument": "NIS2", "Type": "Binding directive (through national law)", "Provisions": 349,
                     "Source": "u", "Retrieved": "2026-10-20"}]
