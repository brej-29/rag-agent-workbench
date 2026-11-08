from datetime import date
from app.services.regs.registry import Registry

P = [
    {"id": "eu-ai-act/art-113", "kind": "article", "text": "It shall apply from 2 August 2026. However: (a) Chapters I and II shall apply from 2 February 2025; (c) Chapter III, Sections 1, 2, and 3, with the exception of Article 6(5), shall apply from: (i) 2 December 2027 as regards AI systems classified as high-risk pursuant to Article 6(2) and Annex III; and (ii) 2 August 2028 as regards AI systems classified as high-risk pursuant to Article 6(1) and Annex I;", "chapter": "XIII", "section": None, "article": "113", "paragraph": None},
    {"id": "eu-ai-act/art-5/par-1", "kind": "article_paragraph", "text": "x", "chapter": "II", "section": None, "article": "5", "paragraph": "1"},
    {"id": "eu-ai-act/art-26/par-1", "kind": "article_paragraph", "text": "x", "chapter": "III", "section": "3", "article": "26", "paragraph": "1"},
    {"id": "eu-ai-act/art-6/par-5", "kind": "article_paragraph", "text": "x", "chapter": "III", "section": "1", "article": "6", "paragraph": "5"},
]
RULES = [
    {"id": "general", "match": {"prefix": "eu-ai-act", "kinds": ["article", "article_paragraph"]},
     "dates": [{"from": "2026-08-02"}], "basis": "eu-ai-act/art-113", "quote": "It shall apply from 2 August 2026"},
    {"id": "ch1-2", "match": {"prefix": "eu-ai-act", "chapters": ["I", "II"]},
     "dates": [{"from": "2025-02-02"}], "basis": "eu-ai-act/art-113", "quote": "Chapters I and II shall apply from 2 February 2025"},
    {"id": "high-risk", "match": {"prefix": "eu-ai-act", "sections": ["III.1", "III.2", "III.3"], "exclude_paragraphs": ["6(5)"]},
     "dates": [{"from": "2027-12-02", "condition": "annex_iii"}, {"from": "2028-08-02", "condition": "annex_i"}],
     "basis": "eu-ai-act/art-113", "quote": "2 December 2027 as regards AI systems classified as high-risk pursuant to Article 6(2) and Annex III"},
]
REG = Registry(RULES, P)
by = {p["id"]: p for p in P}


def test_later_rules_override_and_conditions():
    assert REG.applies(by["eu-ai-act/art-5/par-1"], date(2026, 10, 20)) is True
    hr = by["eu-ai-act/art-26/par-1"]
    assert REG.applies(hr, date(2026, 10, 20)) is False
    assert REG.applies(hr, date(2027, 12, 2), "annex_iii") is True
    assert REG.applies(hr, date(2027, 12, 2), "annex_i") is False
    assert REG.fields(hr)["applies_from"] == "2027-12-02" and "annex_i" in REG.fields(hr)["applies_note"]


def test_exclusions_fall_back_to_earlier_rules():
    assert REG.fields(by["eu-ai-act/art-6/par-5"])["applies_from"] == "2026-08-02"


def test_unmatched_provisions_have_no_date():
    assert REG.applies({"id": "gdpr/art-1", "kind": "article"}, date(2026, 1, 1)) is None


def test_quotes_are_checked_against_basis_text():
    assert REG.unverified_quotes() == []
    bad = Registry(RULES + [{"id": "bad", "match": {"prefix": "eu-ai-act"}, "dates": [{"from": "2030-01-01"}],
                             "basis": "eu-ai-act/art-113", "quote": "shall apply from 1 January 2030"}], P)
    assert bad.unverified_quotes() == ["bad"]
