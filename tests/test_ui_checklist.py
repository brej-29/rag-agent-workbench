import csv
import io

from frontend.services.presenters import checklist_csv, checklist_markdown, status_label, step_line, trace_summary, verdict

ITEM = {"article": "eu-ai-act/art-26", "ref": "AI Act Art. 26", "title": "Obligations of deployers of high-risk AI systems",
        "framework": "EU AI Act", "kind": "obligation", "status": "not_yet", "applies_from": "2027-12-02",
        "why": {"ai_roles": ["deployer"]}, "depends_on": [], "note": None}
CL = {"as_of": "2026-10-20", "profile": {"ai_roles": ["deployer"]},
      "groups": {"now": [], "coming": [ITEM], "depends": [{**ITEM, "article": "eu-ai-act/art-50", "ref": "AI Act Art. 50",
                                                             "status": "applies", "depends_on": ["ai_uses"]}]},
      "not_applicable": [], "questions": [], "practices": [], "counts": {"now": 0, "coming": 1, "depends": 1, "not_applicable": 0}}


def test_markdown_export_lists_groups_with_status_and_disclaimer():
    md = checklist_markdown(CL)
    assert "# Compliance checklist (as of 2026-10-20)" in md and "not legal advice" in md
    assert "- [ ] **AI Act Art. 26** Obligations of deployers of high-risk AI systems — applies from 2027-12-02" in md
    assert "(depends on: ai uses)" in md


def test_csv_export_has_one_row_per_item():
    rows = list(csv.DictReader(io.StringIO(checklist_csv(CL))))
    assert [r["article"] for r in rows] == ["eu-ai-act/art-26", "eu-ai-act/art-50"]
    assert rows[0]["group"] == "Coming" and rows[1]["depends_on"] == "ai_uses"


def test_partly_status_and_checklist_wording():
    assert status_label({"status": "partly", "applies_from": "2026-12-02"}) == ("orange", "partly from 2026-12-02")
    assert "checklist" in step_line({"id": "route", "status": "done", "detail": {"intent": "checklist"}}).lower()
    assert trace_summary(False, 900, 0, None, "checklist") == "Checklist built in 0.9 s"
    assert verdict({"checklist": {"counts": {}}}) == ("blue", "From the rules table, no model")


def test_profile_words_never_borrows_another_facts_label():
    from frontend.services.presenters import profile_words

    words = profile_words({"nis2_sector": "none", "high_risk": "annex_iii", "size": "small", "eu_market": True})
    assert words == "nis2 sector: none, Annex III, size: small, eu market"
    facts = [{"id": "nis2_sector", "labels": {"none": "not in a NIS2 sector"}}, {"id": "eu_market", "short": "EU market"}]
    assert profile_words({"nis2_sector": "none", "eu_market": True}, facts) == "not in a NIS2 sector, EU market"
