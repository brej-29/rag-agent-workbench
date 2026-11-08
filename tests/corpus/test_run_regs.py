from eval.run_regs import label_hits

def test_article_level_labels_match_their_paragraphs_once():
    ranked = ["x/art-26/par-2", "x/art-26/par-1", "y/art-33/par-1", "z/art-1"]
    assert label_hits(ranked, ["x/art-26", "y/art-33/par-1"]) == ["x/art-26", "y/art-33/par-1", "z/art-1"]


def test_every_golden_label_exists_in_the_corpus():
    """Labels must point at real provisions (an article/annex label must have at least one paragraph)."""
    import json
    from pathlib import Path

    import pytest
    from app.services.regs import DATA_DIR, load_provisions

    if not (DATA_DIR / "regs-v1.jsonl").exists():
        pytest.skip("corpus not built")
    ids = [p["id"] for p in load_provisions()]
    rows = [json.loads(l) for l in Path("eval/regs_golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    missing = [lab for r in rows if not r.get("_meta") for lab in r["relevant"]
               if not any(i == lab or i.startswith(lab + "/") for i in ids)]
    assert missing == []


def test_floor_stats_finds_a_separating_floor_or_reports_overlap():
    from eval.run_regs import floor_stats

    sep = floor_stats(answerable_best=[0.42, 0.55, 0.61], unanswerable_top=[0.12, 0.18, 0.25])
    assert sep["min_answerable"] == 0.42 and sep["max_unanswerable"] == 0.25
    assert sep["overlap"] == 0 and 0.25 < sep["suggested_floor"] < 0.42
    over = floor_stats(answerable_best=[0.30, 0.50], unanswerable_top=[0.20, 0.35])
    assert over["overlap"] == 2  # 0.30 <= 0.35 and 0.35 >= 0.30: one each side


def test_summarize_reports_each_question_type():
    from eval.run_regs import summarize

    per = [{"type": "cross", "recall": 0.5, "mrr": 1.0, "ndcg": 0.6}, {"type": "concept", "recall": 1.0, "mrr": 0.5, "ndcg": 1.0}]
    out = summarize(per)
    assert out["recall"] == 0.75 and out["by_type"]["cross"] == {"n": 1, "recall": 0.5, "mrr": 1.0, "ndcg": 0.6}


def test_abstain_report_counts_both_error_kinds():
    from eval.run_regs import abstain_report

    decide = lambda q: (q.startswith("no"), "no_match" if q.startswith("no") else None)  # noqa: E731
    rep = abstain_report(["yes 1", "no 2"], ["no 3", "yes 4"], decide)
    assert rep["correct_abstain"] == 1 and rep["unanswerable"] == 2 and rep["false_abstain"] == 1
    assert rep["missed_unanswerable"] == ["yes 4"] and rep["false_abstentions"] == ["no 2"]
