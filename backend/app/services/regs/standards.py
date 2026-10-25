"""Licensed standards a user brings (spec §15): parsed for the request that uses them; never stored."""
import re
from typing import Dict, List, Optional

from app.services.regs.lexical import BM25

STANDARDS = {"iso-27001": "ISO/IEC 27001:2022", "iso-42001": "ISO/IEC 42001"}
LICENSED_NAMES = {"ISO/IEC 27001": "iso-27001", "ISO/IEC 42001": "iso-42001"}  # scope-guard name -> copy key
_ANNEX_A_27001 = {5: 37, 6: 8, 7: 14, 8: 34}
_HEAD = re.compile(r"^(?:([AB])\.)?(\d{1,2}(?:\.\d{1,2}){0,2})\s+([A-Z].{1,150})$")
_TOC = re.compile(r"(?:\.{3,}|…)\s*\d*$|\s\d{1,3}$")
_ANNEX = re.compile(r"^Annex\s+([AB])\b")
_SCOPE = re.compile(r"^1\s+Scope$")


def detect_standard(text: str) -> Optional[str]:
    head = text[:20000]
    found = {key: m.start() for key, pat in (("iso-27001", r"27001"), ("iso-42001", r"42001"))
             if (m := re.search(pat, head))}
    return min(found, key=found.get) if found else None


def _valid(standard: str, letter: Optional[str], number: str) -> bool:
    parts = [int(x) for x in number.split(".")]
    if letter is None:
        return 4 <= parts[0] <= 10
    if standard == "iso-27001":
        return letter == "A" and len(parts) == 2 and 1 <= parts[1] <= _ANNEX_A_27001.get(parts[0], 0)
    return 2 <= parts[0] <= 10


def parse_clauses(text: str, standard: str) -> Dict[str, dict]:
    """Clause id -> {title, text}: numbered headings at line start, text up to the next heading.
    Table-of-contents entries (dot leaders, page numbers) are skipped; the annex state resets at the body's
    '1 Scope', so a contents line 'Annex A ...' cannot prefix the main clauses."""
    clauses: Dict[str, dict] = {}
    annex, current = None, None
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if _SCOPE.match(line):
            annex, current = None, None
            continue
        m = _ANNEX.match(line)
        if m:
            annex, current = m.group(1), None
            continue
        m = _HEAD.match(line)
        if m and _valid(standard, m.group(1) or annex, m.group(2)):
            if _TOC.search(m.group(3)):
                current = None
                continue
            letter = m.group(1) or annex
            cid = f"{letter}.{m.group(2)}" if letter else m.group(2)
            if cid not in clauses or not clauses[cid]["text"]:
                current = clauses[cid] = {"title": m.group(3).strip(), "text": ""}
                continue
        if current is not None:
            current["text"] = f"{current['text']} {line}".strip()
    return {cid: c for cid, c in clauses.items() if c["text"]}


def coverage(clauses: Dict[str, dict], standard: str) -> dict:
    out = {"clauses": len(clauses), "annex_a": sum(1 for c in clauses if c.startswith("A."))}
    if standard == "iso-27001":
        out["annex_a_expected"] = sum(_ANNEX_A_27001.values())
    return out


def copy_sources(query: str, standards: Optional[List[dict]], k: int = 3) -> List[dict]:
    """Clauses of the user's own copy relevant to the question: named ids first, then BM25."""
    named_q = re.sub(r"Annex\s+A\s*,?\s*(?:control\s*)?", "A.", query, flags=re.I)
    out = []
    for std in standards or []:
        clauses, name = std.get("clauses") or {}, STANDARDS.get(std["name"], std["name"])
        if not clauses:
            continue
        named = [cid for cid in clauses if re.search(rf"(?<![\w.]){re.escape(cid)}(?![\w.])", named_q)]
        bm = BM25((cid, f"{cid} {c['title']} {c['text']}") for cid, c in clauses.items())
        ranked = named + [cid for cid, _ in bm.search(query, k) if cid not in named]
        for cid in ranked[:k]:
            c = clauses[cid]
            out.append({"source": f"{name} (your copy)", "framework": f"{name} (your copy)", "ref": f"{name} {cid}",
                        "title": f"{name} {cid} — {c['title']}", "url": "", "score": 1.0, "chunk_text": c["text"][:4000],
                        "doc_id": f"{std['name']}/{cid}", "kind": "licensed_clause", "binding": "licensed_copy",
                        "rank_score": 1.0, "status": "licensed_copy", "applies_from": None, "status_note": None,
                        "private": True})
    return out
