"""Applicability status of real provisions on fixed dates (registry + committed corpus)."""
from datetime import date

import pytest

from app.services.regs import DATA_DIR
from app.services.regs.ask import regs_corpus, status_for

pytestmark = pytest.mark.skipif(not (DATA_DIR / "regs-v1.jsonl").exists(), reason="corpus not built")
D = date(2026, 10, 20)


@pytest.mark.parametrize("pid,condition,status,applies_from", [
    ("gdpr/art-33/par-1", None, "applies", "2018-05-25"),
    ("eu-ai-act/art-26/par-1", None, "not_yet", "2027-12-02"),
    ("eu-ai-act/art-26/par-1", "annex_i", "not_yet", "2028-08-02"),
    ("eu-ai-act/art-5/par-1a", None, "not_yet", "2026-12-02"),
    ("eu-ai-act/art-5/par-1", None, "applies", "2025-02-02"),
    ("eu-ai-act/art-53/par-1", None, "applies", "2025-08-02"),
    ("eu-ai-act/art-50/par-1", None, "applies", "2026-08-02"),
    ("cra/art-14/par-1", None, "applies", "2026-09-11"),
    ("cra/art-13/par-1", None, "not_yet", "2027-12-11"),
    ("nis2/art-23/par-1", None, "applies", "2024-10-18"),
    ("dora/art-19/par-1", None, "applies", "2025-01-17"),
    ("nist-csf/GV.OC-01", None, "voluntary", None),
    ("eu-ai-act/rct-1", None, "no_date", None),
])
def test_status_on_2026_10_20(pid, condition, status, applies_from):
    corpus, registry = regs_corpus()
    st = status_for(corpus[pid], registry, D, condition)
    assert (st["status"], st["applies_from"]) == (status, applies_from)


def test_high_risk_deployer_duties_flip_on_the_annex_iii_date():
    corpus, registry = regs_corpus()
    p = corpus["eu-ai-act/art-26/par-1"]
    assert status_for(p, registry, date(2027, 12, 1), "annex_iii")["status"] == "not_yet"
    assert status_for(p, registry, date(2027, 12, 2), "annex_iii")["status"] == "applies"
    assert status_for(p, registry, date(2027, 12, 2), None)["status"] == "depends"
