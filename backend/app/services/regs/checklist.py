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
