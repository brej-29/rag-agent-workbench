from datetime import date

from frontend.services.home_html import (
    MILESTONES, basis_ref, proof_html, proof_panels, stats, timeline, timeline_html, timeline_summary, until,
)
from frontend.services.presenters import RULEBOOK_LINES, rulebook_stack_html

RESULTS = {
    "regs_retrieval": {"k": 6, "n": 50, "aggregate": {"recall": 0.8713}},
    "regs_abstain": {"correct_abstain": 13, "unanswerable": 14, "false_abstain": 0, "answerable": 50},
    "router": {"n": 36, "accuracy": 0.944},
}
SNAPSHOT = {"obligations": 112, "milestones": [
    {"id": "gdpr", "from": "2018-05-25", "condition": None, "basis": "gdpr/art-99/par-2", "quote": "It shall apply"},
    {"id": "ai-act-articles-102-110", "from": "2026-07-27", "condition": None, "basis": "eu-ai-act/art-113", "quote": ""},
    {"id": "ai-act-high-risk", "from": "2027-12-02", "condition": "annex_iii", "basis": "eu-ai-act/art-113",
     "quote": "(i) 2 December 2027"},
]}


def test_proof_panels_keep_the_published_values():
    panels = proof_panels(RESULTS)
    assert [(p["big"], p["filled"], p["total"]) for p in panels] == [
        ("87%", 87, 100), ("13 of 14", 13, 14), ("0 of 50", 50, 50), ("94%", 34, 36)]
    html = proof_html(panels)
    assert html.count('class="on"') == 87 + 13 + 50 + 34


def test_timeline_uses_snapshot_dates_and_skips_unlabelled_rules():
    items = timeline(SNAPSHOT, date(2026, 11, 8))
    assert [(i["framework"], i["from"], i["past"]) for i in items] == [
        ("GDPR", "2018-05-25", True), ("AI Act", "2027-12-02", False)]
    assert items[0]["basis"] == "GDPR Art. 99(2)" and items[1]["until"] == "In 13 months"
    html = timeline_html(items)
    assert html.index("cw-today") > html.index("2018-05-25") and html.index("cw-today") < html.index("2027-12-02")
    assert timeline_summary(items).startswith("1 of 2 start dates have passed. Next: AI Act, high-risk uses")


def test_every_milestone_label_matches_a_dated_rule():
    import json
    from pathlib import Path

    snap = json.loads((Path(__file__).resolve().parents[1] / "eval/results/timeline.json").read_text(encoding="utf-8"))
    dated = {(m["id"], m["condition"]) for m in snap["milestones"]}
    assert set(MILESTONES) <= dated


def test_small_helpers():
    assert until("2026-11-10", date(2026, 11, 8)) == "In 2 days"
    assert until("2026-11-08", date(2026, 11, 8)) == "In force"
    assert basis_ref("eu-ai-act/art-113") == "AI Act Art. 113"
    assert [x["n"] for x in stats([{"provisions": 3}, {"provisions": 4}], SNAPSHOT, 2)] == [2, 7, 112, 2]


def test_rulebook_stack_has_eight_described_books():
    html = rulebook_stack_html()
    assert html.count('class="cw-book"') == 8 and html.count("<small>") == 8
    assert len(RULEBOOK_LINES) == 6
