"""Deterministic checklist: which articles of the EU laws bind a profile on a date (spec §11). No LLM.

Rules live in obligations.yaml (+ facts.yaml); dates come from the applicability registry.
"""
from collections import Counter
from datetime import date
from functools import lru_cache
import re
from typing import Any, Dict, List, Optional

import yaml

from app.services.regs import DATA_DIR
from app.services.regs.ask import regs_corpus, status_for
from app.services.regs.facts import load_facts, normalize_profile
from app.services.regs.registry import Registry, _norm
from app.services.regs.rules import evaluate


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    return yaml.safe_load((DATA_DIR / "obligations.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def article_index() -> Dict[str, List[dict]]:
    """'eu-ai-act/art-26' -> its provisions in corpus order; article-level records map to themselves."""
    corpus, _ = regs_corpus()
    index: Dict[str, List[dict]] = {}
    for pid, p in corpus.items():
        index.setdefault("/".join(pid.split("/")[:2]), []).append(p)
    return index


def catalog_problems(catalog: dict, facts: List[dict], corpus: Dict[str, dict],
                     index: Dict[str, List[dict]]) -> List[str]:
    """Everything CI must refuse: unknown articles/facts/values/defs, duplicates, quotes not in their basis."""
    problems: List[str] = []
    by_id = {f["id"]: f for f in facts}

    def check(cond: dict, where: str) -> None:
        if "use" in cond:
            if cond["use"] not in catalog["defs"]:
                problems.append(f"{where}: unknown def {cond['use']}")
            return
        if "not" in cond:
            return check(cond["not"], where)
        for op in ("all", "any"):
            if op in cond:
                for c in cond[op]:
                    check(c, where)
                return
        if len(cond) != 1:
            problems.append(f"{where}: a fact test has exactly one key: {cond}")
            return
        (fact, wanted), = cond.items()
        f = by_id.get(fact)
        if not f:
            problems.append(f"{where}: unknown fact {fact}")
            return
        allowed = [True, False] if f["type"] == "bool" else f["values"]
        for w in wanted if isinstance(wanted, list) else [wanted]:
            if w not in allowed or (f["type"] != "bool" and isinstance(w, bool)):
                problems.append(f"{where}: {fact} has no value {w!r}")

    for name, cond in catalog["defs"].items():
        check(cond, f"def {name}")
    seen = set()
    for ob in catalog["obligations"]:
        a = ob["article"]
        if a in seen:
            problems.append(f"{a}: listed twice")
        seen.add(a)
        if a not in index:
            problems.append(f"{a}: not in the corpus")
        check(ob["when"], a)
        basis = corpus.get(ob.get("basis", ""))
        if not basis:
            problems.append(f"{a}: basis {ob.get('basis')} not in the corpus")
        elif not ob.get("quote") or _norm(ob["quote"]) not in _norm(basis["text"]):
            problems.append(f"{a}: quote not found in {ob['basis']}")
    for pr in catalog.get("practices", []):
        check(pr["when"], pr["framework"])
    for f in facts:
        if f["basis"] not in index:
            problems.append(f"fact {f['id']}: basis {f['basis']} not in the corpus")
    return problems


def _article_ref(p: dict) -> str:
    return re.sub(r"\s*\([^)]*\)$", "", p["ref"])


def article_status(paras: List[dict], registry: Registry, as_of: date, condition: Optional[str]) -> dict:
    """One status for an article from its provisions' statuses ('partly' when some apply and some do not yet)."""
    sts = [status_for(p, registry, as_of, condition) for p in paras]
    dated = [s for s in sts if s["status"] != "no_date"] or sts
    kinds = {s["status"] for s in dated}
    note = next((s["note"] for s in dated if s["note"]), None)

    def dates(kind: str) -> List[str]:
        return sorted(s["applies_from"] for s in dated if s["status"] == kind and s["applies_from"])

    if kinds == {"applies"}:
        return {"status": "applies", "applies_from": dates("applies")[-1], "note": note}
    if "applies" in kinds and "not_yet" in kinds:
        return {"status": "partly", "applies_from": dates("not_yet")[0], "note": note}
    if "depends" in kinds:
        return {"status": "depends", "applies_from": (dates("depends") or [None])[0], "note": note}
    if "not_yet" in kinds:
        return {"status": "not_yet", "applies_from": dates("not_yet")[0], "note": note}
    return {"status": dated[0]["status"], "applies_from": None, "note": note}


def _practices(profile: dict, catalog: dict, index: Dict[str, List[dict]]) -> List[dict]:
    out = []
    for pr in catalog.get("practices", []):
        value, why = evaluate(pr["when"], profile, catalog["defs"])
        if value is not True:
            continue
        groups: Dict[str, List[dict]] = {}
        for key, paras in index.items():
            if key.startswith(pr["prefix"] + "/"):
                for p in paras:
                    if p["kind"] in pr["kinds"]:
                        groups.setdefault(p["path"][0], []).append(
                            {"id": p["id"], "ref": p["ref"], "title": p["title"] if p["kind"] == "category" else p["text"]})
        out.append({"framework": pr["framework"], "reason": pr["reason"], "why": why,
                    "groups": [{"name": n, "items": items} for n, items in groups.items()]})
    return out


def build_checklist(profile: Dict[str, Any], as_of: date, with_text: bool = False) -> Dict[str, Any]:
    """Pure function of (profile, as_of): grouped articles, deciding questions, voluntary practices."""
    catalog, facts = load_catalog(), {f["id"]: f for f in load_facts()}
    prof = normalize_profile(profile)
    _, registry = regs_corpus()
    index = article_index()
    condition = prof.get("high_risk") if prof.get("high_risk") in ("annex_iii", "annex_i") else None
    groups: Dict[str, List[dict]] = {"now": [], "coming": [], "depends": []}
    not_applicable: List[dict] = []
    waiting: Counter = Counter()
    for ob in catalog["obligations"]:
        paras = index[ob["article"]]
        head = paras[0]
        value, why = evaluate(ob["when"], prof, catalog["defs"])
        item = {"article": ob["article"], "ref": _article_ref(head), "title": head.get("title") or "",
                "framework": head["framework"], "binding": head["binding"], "kind": ob.get("kind", "obligation"),
                "note": ob.get("note"), "url": head.get("source_url", "")}
        if value is False:
            not_applicable.append({**item, "because": why})
            continue
        st = article_status(paras, registry, as_of, condition)
        item.update(status=st["status"], applies_from=st["applies_from"], status_note=st["note"],
                    why={k: v for k, v in why.items() if v is not None},
                    depends_on=sorted(k for k, v in why.items() if v is None),
                    paragraphs=[{"id": p["id"], "ref": p["ref"], **({"text": p["text"]} if with_text else {})}
                                for p in paras])
        if value is None:
            waiting.update(item["depends_on"])
            groups["depends"].append(item)
        elif st["status"] in ("applies", "partly"):
            groups["now"].append(item)
        else:
            groups["coming"].append(item)
    groups["coming"].sort(key=lambda i: i["applies_from"] or "9999-12-31")
    questions = [{"fact": f, "label": facts[f]["label"], "type": facts[f]["type"],
                  "values": facts[f].get("values", []), "decides": n} for f, n in waiting.most_common(3)]
    return {"as_of": as_of.isoformat(), "profile": prof, "groups": groups, "not_applicable": not_applicable,
            "questions": questions, "practices": _practices(prof, catalog, index),
            "counts": {**{k: len(v) for k, v in groups.items()}, "not_applicable": len(not_applicable)}}


def summary_text(checklist: Dict[str, Any]) -> str:
    c = checklist["counts"]
    parts = [f"For this profile on {checklist['as_of']}: {c['now']} articles apply now, {c['coming']} apply later "
             f"and {c['depends']} depend on facts not given yet ({c['not_applicable']} do not apply)."]
    if checklist["questions"]:
        parts.append("To narrow it down: " + " ".join(q["label"] for q in checklist["questions"]))
    parts.append("Information from official EU text, not legal advice.")
    return "\n\n".join(parts)
