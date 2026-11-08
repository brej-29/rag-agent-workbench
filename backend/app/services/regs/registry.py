"""Applicability dates for provisions. Each rule cites the provision (and verbatim quote) that sets it."""
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class Dated:
    date: Optional[date]
    condition: Optional[str]
    basis: str
    quote: str
    note: Optional[str]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _matches(m: dict, p: dict) -> bool:
    """`kinds` filters; chapters/sections/articles/paragraphs are alternatives (any may match); exclusions win."""
    if not p["id"].startswith(m["prefix"] + "/"):
        return False
    art, par = p.get("article"), p.get("paragraph")
    ref = f"{art}({par})" if par else art
    if art in [str(a) for a in m.get("exclude_articles", [])] or ref in m.get("exclude_paragraphs", []):
        return False
    if "kinds" in m and p.get("kind") not in m["kinds"]:
        return False
    scopes = {"chapters": p.get("chapter"), "sections": f"{p.get('chapter')}.{p.get('section')}",
              "articles": art, "paragraphs": ref}
    given = [k for k in scopes if k in m]
    return not given or any(scopes[k] in [str(x) for x in m[k]] for k in given)


class Registry:
    def __init__(self, rules: List[dict], provisions: List[dict]):
        self.rules = rules
        self.text = {p["id"]: _norm(p.get("text", "")) for p in provisions}

    @classmethod
    def load(cls, path: Path, provisions: List[dict]) -> "Registry":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or [], provisions)

    def _rule(self, p: dict) -> Optional[dict]:
        hit = None
        for rule in self.rules:
            if _matches(rule["match"], p):
                hit = rule
        return hit

    def dates(self, p: dict) -> List[Dated]:
        rule = self._rule(p)
        if not rule:
            return []
        entries = rule.get("dates") or [{}]
        return [Dated(date.fromisoformat(str(d["from"])) if d.get("from") else None, d.get("condition"),
                      rule.get("basis", ""), rule.get("quote", ""), rule.get("note")) for d in entries]

    def applies(self, p: dict, as_of: date, condition: Optional[str] = None) -> Optional[bool]:
        """True/False once known; None when undated, or when it depends on a condition the caller did not give."""
        ds = [d for d in self.dates(p) if d.date]
        if not ds:
            return None
        if condition is not None:
            chosen = [d for d in ds if d.condition in (None, condition)]
            return any(as_of >= d.date for d in chosen) if chosen else None
        unconditional = [d for d in ds if d.condition is None]
        if unconditional:
            return as_of >= unconditional[0].date
        if as_of < min(d.date for d in ds):
            return False
        return True if as_of >= max(d.date for d in ds) else None

    def fields(self, p: dict) -> dict:
        ds = self.dates(p)
        dated = [d for d in ds if d.date]
        if not ds:
            return {"applies_from": None, "applies_note": None, "date_basis": None}
        note = "; ".join(f"{d.date.isoformat()} if {d.condition}" for d in dated if d.condition) or None
        notes = "; ".join(x for x in (note, ds[0].note) if x) or None
        return {"applies_from": min(d.date for d in dated).isoformat() if dated else None,
                "applies_note": notes, "date_basis": ds[0].basis or None}

    def unverified_quotes(self) -> List[str]:
        return [r["id"] for r in self.rules if r.get("quote") and _norm(r["quote"]) not in self.text.get(r.get("basis", ""), "")]
