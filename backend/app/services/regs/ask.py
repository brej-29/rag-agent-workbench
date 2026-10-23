"""Compliance-path helpers: scope guard, applicability status, retrieval and diversification.

Pure functions (Pinecone search, corpus and registry are passed in) so each is unit-tested.
"""
import re
from datetime import date
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set, Tuple

from app.services.regs import DATA_DIR, load_provisions
from app.services.regs.registry import Registry

COVERED = ["EU AI Act", "GDPR", "NIS2", "DORA", "Cyber Resilience Act", "NIST SP 800-53", "NIST CSF 2.0", "NIST AI RMF"]

_FRAMEWORK_ALIASES: Dict[str, List[str]] = {
    "EU AI Act": [r"\bai act\b", r"artificial intelligence act", r"2024/1689"],
    "GDPR": [r"\bgdpr\b", r"general data protection regulation", r"2016/679"],
    "NIS2": [r"\bnis ?2\b", r"2022/2555", r"2024/2690"],
    "DORA": [r"\bdora\b", r"digital operational resilience", r"2022/2554"],
    "Cyber Resilience Act": [r"cyber resilience act", r"\bcra\b", r"2024/2847"],
    "NIST SP 800-53": [r"800-53"],
    "NIST CSF 2.0": [r"\bcsf\b", r"cybersecurity framework"],
    "NIST AI RMF": [r"\bai rmf\b", r"ai risk management framework"],
}
_OUT_OF_SCOPE: Dict[str, List[str]] = {
    "CCPA/CPRA": [r"\bccpa\b", r"\bcpra\b", r"california consumer privacy"],
    "HIPAA": [r"\bhipaa\b"],
    "India DPDP Act": [r"\bdpdp\b", r"digital personal data protection act"],
    "Sarbanes-Oxley": [r"sarbanes", r"\bsox\b"],
    "PCI DSS": [r"\bpci[ -]?dss\b", r"\bpci\b"],
    "UK Online Safety Act": [r"online safety act"],
    "LGPD": [r"\blgpd\b"],
    "PIPL": [r"\bpipl\b"],
    "PIPEDA": [r"\bpipeda\b"],
}
_LICENSED: Dict[str, List[str]] = {
    "ISO/IEC 27001": [r"iso\s*/?\s*(?:iec\s*)?27001"],
    "ISO/IEC 27002": [r"iso\s*/?\s*(?:iec\s*)?27002"],
    "ISO/IEC 42001": [r"iso\s*/?\s*(?:iec\s*)?42001"],
    "SOC 2": [r"\bsoc\s*2\b", r"trust services criteria"],
}


def _found(table: Dict[str, List[str]], text: str) -> List[str]:
    t = text.lower()
    return [name for name, pats in table.items() if any(re.search(p, t) for p in pats)]


def named_frameworks(query: str) -> Set[str]:
    return set(_found(_FRAMEWORK_ALIASES, query))


def _join(names: List[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def scope_guard(query: str) -> dict:
    """Deterministic: does the question name an instrument this corpus cannot answer from?"""
    out, lic, inside = _found(_OUT_OF_SCOPE, query), _found(_LICENSED, query), named_frameworks(query)
    names = out + lic
    if not names:
        return {"abstain": False, "kind": None, "names": [], "message": None, "warning": None}
    verb = "is" if len(names) == 1 else "are"
    if lic and not out:
        kind = "licensed"
        msg = (f"{_join(names)} {verb} a licensed standard, so its text is not in the public knowledge base. "
               "Ask about the frameworks covered here, or add your own licensed copy to a private workspace "
               "(coming soon).")
    else:
        kind = "out_of_scope"
        msg = (f"{_join(names)} {verb} not covered here, so I can't answer from official text. "
               f"This knowledge base covers {_join(COVERED)}.")
    if inside:
        return {"abstain": False, "kind": kind, "names": names, "message": None,
                "warning": f"{_join(names)} {verb} not covered; the answer uses {_join(sorted(inside))} only."}
    return {"abstain": True, "kind": kind, "names": names, "message": msg, "warning": None}


def status_for(p: dict, registry: Registry, as_of: date, condition: Optional[str]) -> dict:
    """Applicability on `as_of`, from the registry only (never from the model)."""
    if p.get("binding") == "voluntary_framework":
        return {"status": "voluntary", "applies_from": None, "note": None}
    dates = registry.dates(p)
    dated = [d for d in dates if d.date]
    note = registry.fields(p)["applies_note"] if dates else None
    if not dated:
        return {"status": "no_date", "applies_from": None, "note": note}
    relevant = ([d for d in dated if d.condition in (None, condition)] if condition else dated) or dated
    applies = registry.applies(p, as_of, condition)
    if applies is None:
        return {"status": "depends", "applies_from": min(d.date for d in relevant).isoformat(), "note": note}
    if applies:
        return {"status": "applies", "applies_from": min(d.date for d in relevant if d.date <= as_of).isoformat(),
                "note": note}
    return {"status": "not_yet", "applies_from": min(d.date for d in relevant if d.date > as_of).isoformat(),
            "note": note}


def status_text(st: dict, as_of: date) -> str:
    s, frm, note = st["status"], st.get("applies_from"), st.get("note")
    if s == "applies":
        return f"in force since {frm}"
    if s == "not_yet":
        return f"applies from {frm} (not yet in force on {as_of.isoformat()})" + (f"; {note}" if note else "")
    if s == "depends":
        return f"depends on classification: {note}"
    if s == "voluntary":
        return "voluntary framework (no legal effective date)"
    return f"no date of its own ({note})" if note else "no date of its own"


def diversify(cands: List[dict], k: int, max_per: int) -> List[dict]:
    """Best-first, at most `max_per` per framework, then fill any remaining slots best-first."""
    ranked = sorted(cands, key=lambda c: -c["rank_score"])
    chosen: List[dict] = []
    counts: Dict[str, int] = {}
    for c in ranked:
        if len(chosen) < k and counts.get(c["framework"], 0) < max_per:
            chosen.append(c)
            counts[c["framework"]] = counts.get(c["framework"], 0) + 1
    picked = {id(c) for c in chosen}
    chosen += [c for c in ranked if id(c) not in picked][: k - len(chosen)]
    return chosen


def mismatch_warning(query: str, sources: List[dict]) -> Optional[str]:
    named = named_frameworks(query)
    if not named or named & {s.get("framework") for s in sources}:
        return None
    return (f"The question names {_join(sorted(named))}, but no passage from it cleared the relevance floor; "
            "the cited provisions come from other frameworks.")


@lru_cache(maxsize=1)
def regs_corpus() -> Tuple[Dict[str, dict], Registry]:
    provisions = load_provisions()
    return {p["id"]: p for p in provisions}, Registry.load(DATA_DIR / "applicability.yaml", provisions)


def retrieve_regs(query: str, frameworks: Optional[List[str]], as_of: date, condition: Optional[str], settings,
                  search: Callable, corpus: Optional[Dict[str, dict]] = None,
                  registry: Optional[Registry] = None) -> Tuple[List[dict], float]:
    """Dense ranking from Pinecone; provision text and status from the committed corpus and registry."""
    if corpus is None or registry is None:
        corpus, registry = regs_corpus()
    hits = search(namespace=settings.REGS_NAMESPACE, query_text=query, top_k=settings.REGS_CANDIDATES,
                  filters={"framework": {"$in": frameworks}} if frameworks else None, fields=["ref"])
    cands, top = [], 0.0
    for h in hits:
        p = corpus.get(h.get("_id", ""))
        if not p:
            continue
        score = float(h.get("_score") or 0.0)
        top = max(top, score)
        st = status_for(p, registry, as_of, condition)
        cands.append({
            "source": p["framework"], "framework": p["framework"], "ref": p["ref"],
            "title": f"{p['ref']} — {p['title']}" if p.get("title") else p["ref"],
            "url": p.get("source_url", ""), "score": score, "chunk_text": p["text"], "doc_id": p["id"],
            "kind": p.get("kind"), "binding": p.get("binding"),
            "rank_score": score * (settings.REGS_RECITAL_WEIGHT if p.get("kind") == "recital" else 1.0),
            "status": st["status"], "applies_from": st["applies_from"], "status_note": st["note"],
        })
    return diversify(cands, settings.REGS_TOP_K, settings.REGS_MAX_PER_FRAMEWORK), top
