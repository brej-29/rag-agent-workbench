from datetime import date

import pytest

from app.services.regs import ask
from app.services.regs.registry import Registry

RULES = [
    {"id": "general", "match": {"prefix": "eu-ai-act", "kinds": ["article_paragraph"]}, "dates": [{"from": "2026-08-02"}],
     "basis": "b", "quote": ""},
    {"id": "hr", "match": {"prefix": "eu-ai-act", "sections": ["III.3"]},
     "dates": [{"from": "2027-12-02", "condition": "annex_iii"}, {"from": "2028-08-02", "condition": "annex_i"}],
     "basis": "b", "quote": ""},
    {"id": "gdpr", "match": {"prefix": "gdpr"}, "dates": [{"from": "2018-05-25"}], "basis": "b", "quote": ""},
    {"id": "rct", "match": {"prefix": "eu-ai-act", "kinds": ["recital"]}, "dates": [{}], "basis": "b", "note": "recital"},
]
ART26 = {"id": "eu-ai-act/art-26/par-1", "kind": "article_paragraph", "chapter": "III", "section": "3", "article": "26",
         "paragraph": "1", "binding": "regulation"}
GDPR33 = {"id": "gdpr/art-33/par-1", "kind": "article_paragraph", "article": "33", "paragraph": "1", "binding": "regulation"}
REG = Registry(RULES, [])


@pytest.mark.parametrize("q,kind,names", [
    ("What penalties apply under the California Consumer Privacy Act?", "out_of_scope", ["CCPA/CPRA"]),
    ("What does HIPAA require?", "out_of_scope", ["HIPAA"]),
    ("What does ISO/IEC 27001 Annex A 5.1 say?", "licensed", ["ISO/IEC 27001"]),
    ("Which SOC 2 criteria cover availability?", "licensed", ["SOC 2"]),
])
def test_scope_guard_abstains_on_named_out_of_scope_instruments(q, kind, names):
    g = ask.scope_guard(q)
    assert g["abstain"] and g["kind"] == kind and g["names"] == names and g["message"]


def test_scope_guard_mixed_question_answers_with_a_warning():
    g = ask.scope_guard("How does GDPR breach notification compare with HIPAA?")
    assert not g["abstain"] and "HIPAA" in g["warning"]


@pytest.mark.parametrize("q", ["What does Article 26 of the AI Act require?", "Who won the 2018 FIFA World Cup?",
                               "Is a DORA register of information required?"])
def test_scope_guard_passes_other_questions(q):
    assert not ask.scope_guard(q)["abstain"]


def test_named_frameworks():
    assert ask.named_frameworks("Under the AI Act and GDPR Article 35...") == {"EU AI Act", "GDPR"}
    assert ask.named_frameworks("SP 800-53 AC-2") == {"NIST SP 800-53"}


def test_status_for_dates_conditions_and_voluntary():
    d = date(2026, 10, 20)
    assert ask.status_for(GDPR33, REG, d, None) == {"status": "applies", "applies_from": "2018-05-25", "note": None}
    st = ask.status_for(ART26, REG, d, None)
    assert st["status"] == "not_yet" and st["applies_from"] == "2027-12-02"
    assert ask.status_for(ART26, REG, date(2028, 1, 1), None)["status"] == "depends"
    assert ask.status_for(ART26, REG, date(2028, 1, 1), "annex_iii")["status"] == "applies"
    assert ask.status_for(ART26, REG, date(2028, 1, 1), "annex_i") == {
        "status": "not_yet", "applies_from": "2028-08-02", "note": "2027-12-02 if annex_iii; 2028-08-02 if annex_i"}
    assert ask.status_for({"id": "nist-csf/GV.OC-01", "binding": "voluntary_framework"}, REG, d, None)["status"] == "voluntary"
    assert ask.status_for({"id": "eu-ai-act/rct-1", "kind": "recital", "binding": "regulation"}, REG, d, None)["status"] == "no_date"


def test_status_text_is_plain():
    d = date(2026, 10, 20)
    assert ask.status_text({"status": "applies", "applies_from": "2018-05-25", "note": None}, d) == "in force since 2018-05-25"
    assert "not yet in force on 2026-10-20" in ask.status_text({"status": "not_yet", "applies_from": "2027-12-02", "note": None}, d)
    assert ask.status_text({"status": "voluntary", "applies_from": None, "note": None}, d) == "voluntary framework (no legal effective date)"


def test_diversify_caps_each_framework_then_fills():
    c = [{"framework": "A", "rank_score": s} for s in (0.9, 0.8, 0.7, 0.6)] + [{"framework": "B", "rank_score": 0.5}]
    out = ask.diversify(c, k=4, max_per=2)
    assert [x["framework"] for x in out] == ["A", "A", "B", "A"]


def test_mismatch_warning_when_named_framework_not_cited():
    assert ask.mismatch_warning("What does GDPR say about logs?", [{"framework": "EU AI Act"}])
    assert ask.mismatch_warning("What does GDPR say?", [{"framework": "GDPR"}]) is None
    assert ask.mismatch_warning("What must deployers log?", [{"framework": "EU AI Act"}]) is None


def test_retrieve_regs_maps_ids_to_provisions_with_status_and_filters():
    calls = {}

    def fake_search(**kw):
        calls.update(kw)
        return [{"_id": "gdpr/art-33/par-1", "_score": 0.61}, {"_id": "eu-ai-act/rct-12", "_score": 0.60},
                {"_id": "missing/x", "_score": 0.9}]

    class S:
        REGS_NAMESPACE, REGS_CANDIDATES, REGS_TOP_K, REGS_MAX_PER_FRAMEWORK, REGS_RECITAL_WEIGHT = "regs-v1", 30, 6, 3, 0.9

    corpus = {"gdpr/art-33/par-1": {**GDPR33, "ref": "GDPR Art. 33(1)", "title": "Breach", "text": "notify within 72 hours",
                                    "framework": "GDPR", "source_url": "u"},
              "eu-ai-act/rct-12": {"id": "eu-ai-act/rct-12", "kind": "recital", "ref": "AI Act Recital 12", "title": "",
                                   "text": "x", "framework": "EU AI Act", "binding": "regulation", "source_url": "u"}}
    sources, top = ask.retrieve_regs("breach", ["GDPR", "EU AI Act"], date(2026, 10, 20), None, S, fake_search, corpus, REG)
    assert calls["filters"] == {"framework": {"$in": ["GDPR", "EU AI Act"]}} and calls["namespace"] == "regs-v1"
    assert [s["ref"] for s in sources] == ["GDPR Art. 33(1)", "AI Act Recital 12"] and top == 0.61
    assert sources[0]["status"] == "applies" and sources[0]["score"] == 0.61
    assert sources[1]["rank_score"] == pytest.approx(0.54)
