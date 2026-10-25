import json
from datetime import date
from pathlib import Path

import pytest

from app.services.regs.checklist import article_index, build_checklist, load_catalog, summary_text
from app.services.regs.facts import load_facts
from app.services.regs.rules import evaluate

ROWS = [json.loads(l) for l in (Path(__file__).resolve().parents[2] / "eval/regs_profiles.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
PROFILES = {r["name"]: r for r in ROWS if not r.get("_meta")}


def _outcomes(cl):
    out = {i["article"]: "binds" for g in ("now", "coming") for i in cl["groups"][g]}
    out.update({i["article"]: "depends" for i in cl["groups"]["depends"]})
    out.update({i["article"]: "not" for i in cl["not_applicable"]})
    return out


@pytest.mark.parametrize("name", list(PROFILES))
def test_golden_profiles(name):
    r = PROFILES[name]
    got = _outcomes(build_checklist(r["profile"], date.fromisoformat(r["as_of"])))
    wrong = {a: (k, got.get(a)) for k in ("binds", "depends", "not") for a in r[k] if got.get(a) != k}
    assert wrong == {}


def test_nothing_known_binds_nothing_and_asks_the_deciding_questions():
    cl = build_checklist({}, date(2026, 10, 20))
    assert cl["counts"]["now"] == cl["counts"]["coming"] == 0 and cl["counts"]["depends"] == 112
    assert len(cl["questions"]) == 3 and all(q["decides"] > 0 and q["label"] for q in cl["questions"])


def _candidates(f):
    if f["type"] == "bool":
        return [True, False]
    if f["type"] == "choice":
        return list(f["values"])
    return [[]] + [[v] for v in f["values"]]


def test_answering_a_fact_never_changes_a_decided_outcome():
    cat = load_catalog()
    for r in PROFILES.values():
        base = {o["article"]: evaluate(o["when"], r["profile"], cat["defs"])[0] for o in cat["obligations"]}
        for f in load_facts():
            if f["id"] in r["profile"]:
                continue
            for v in _candidates(f):
                for o in cat["obligations"]:
                    before = base[o["article"]]
                    if before is not None:
                        after = evaluate(o["when"], {**r["profile"], f["id"]: v}, cat["defs"])[0]
                        assert after == before, (r["name"], f["id"], v, o["article"])


def test_every_depends_item_names_a_deciding_fact_and_why_explains_binding_items():
    cl = build_checklist(PROFILES["eu_cv_deployer"]["profile"], date(2026, 10, 20))
    assert all(i["depends_on"] for i in cl["groups"]["depends"])
    art26 = next(i for g in ("now", "coming") for i in cl["groups"][g] if i["article"] == "eu-ai-act/art-26")
    assert art26["why"]["ai_roles"] == ["deployer"] and art26["why"]["high_risk"] == "annex_iii"
    assert art26["ref"] == "AI Act Art. 26" and art26["title"] == "Obligations of deployers of high-risk AI systems"


def test_high_risk_deployer_items_move_to_applies_now_on_the_annex_iii_date():
    p = PROFILES["eu_cv_deployer"]["profile"]
    def group(d):
        cl = build_checklist(p, d)
        return next(g for g in ("now", "coming") for i in cl["groups"][g] if i["article"] == "eu-ai-act/art-26")
    assert group(date(2027, 12, 1)) == "coming" and group(date(2027, 12, 2)) == "now"


def test_statuses_partly_and_directive_note():
    cl = build_checklist(PROFILES["eu_cv_deployer"]["profile"], date(2026, 10, 20))
    art5 = next(i for i in cl["groups"]["now"] if i["article"] == "eu-ai-act/art-5")
    assert (art5["status"], art5["applies_from"]) == ("partly", "2026-12-02") and art5["kind"] == "prohibition"
    nis = build_checklist(PROFILES["dns_micro"]["profile"], date(2026, 10, 20))
    art21 = next(i for i in nis["groups"]["now"] if i["article"] == "nis2/art-21")
    assert art21["binding"] == "directive" and "transposing law" in art21["status_note"]


def test_camera_maker_reporting_now_and_product_duties_later():
    cl = build_checklist(PROFILES["eu_camera_maker"]["profile"], date(2026, 10, 20))
    assert "cra/art-14" in {i["article"] for i in cl["groups"]["now"]}
    coming = [i for i in cl["groups"]["coming"] if i["article"] == "cra/art-13"]
    assert coming and coming[0]["applies_from"] == "2027-12-11"


def test_practices_and_text_option():
    cl = build_checklist(PROFILES["eu_cv_deployer"]["profile"], date(2026, 10, 20), with_text=True)
    assert [p["framework"] for p in cl["practices"]] == ["NIST AI RMF"]
    assert sum(len(g["items"]) for g in cl["practices"][0]["groups"]) == 72
    assert cl["groups"]["now"][0]["paragraphs"][0]["text"]
    cloud = build_checklist(PROFILES["eu_cloud_large"]["profile"], date(2026, 10, 20))
    assert [p["framework"] for p in cloud["practices"]] == ["NIST CSF 2.0"]


def test_summary_text_is_deterministic():
    cl = build_checklist({}, date(2026, 10, 20))
    text = summary_text(cl)
    assert text.startswith("For this profile on 2026-10-20: 0 articles apply now, 0 apply later and 112 depend")
    assert "not legal advice" in text


def test_implementing_regulation_items_follow_the_entity_type():
    cl = build_checklist(PROFILES["eu_cloud_large"]["profile"], date(2026, 10, 24))
    now = {i["article"]: i for i in cl["groups"]["now"]}
    assert {"nis2-ir/art-2", "nis2-ir/art-3", "nis2-ir/art-7"} <= set(now)
    assert any(p["id"] == "nis2-ir/anx-1/pt-5.1" for p in now["nis2-ir/art-2"]["paragraphs"])  # include: annex
    assert now["nis2-ir/art-2"]["status"] == "applies" and now["nis2-ir/art-2"]["applies_from"] == "2024-11-07"


def test_items_carry_crosswalk_and_practices_narrow_to_linked_csf():
    cl = build_checklist(PROFILES["eu_cloud_large"]["profile"], date(2026, 10, 24))
    art21 = next(i for i in cl["groups"]["now"] if i["article"] == "nis2/art-21")
    groups = {m["group"] for m in art21["crosswalk"]["items"]}
    assert "NIS2 Implementing Regulation 2024/2690" in groups
    csf = next(p for p in cl["practices"] if p["framework"] == "NIST CSF 2.0")
    assert csf["narrowed"] and all(i["id"].startswith("nist-csf/") and "-" in i["id"] for g in csf["groups"] for i in g["items"])
    gdpr32 = next(i for g in ("now", "coming") for i in cl["groups"][g] if i["article"] == "gdpr/art-32")
    assert gdpr32["crosswalk"]["items"] == []
