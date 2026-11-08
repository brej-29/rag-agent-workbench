"""Crosswalk lookups (spec §13.4): matching requirements in other frameworks, with source and path."""
import json
from functools import lru_cache
from typing import Dict, List

import yaml

from app.services.regs import DATA_DIR
from app.services.regs.ask import regs_corpus

GROUPS = {"eu-ai-act": "EU AI Act", "gdpr": "GDPR", "nis2": "NIS2", "nis2-ir": "NIS2 Implementing Regulation 2024/2690",
          "dora": "DORA", "cra": "Cyber Resilience Act", "nist-800-53": "NIST SP 800-53", "nist-csf": "NIST CSF 2.0",
          "nist-ai-rmf": "NIST AI RMF", "iso-27001": "ISO/IEC 27001:2022", "iso-42001-fdis": "ISO/IEC FDIS 42001"}
CAVEAT = "Mappings show related requirements, not equivalence."


@lru_cache(maxsize=1)
def sources() -> Dict[str, dict]:
    return {s["id"]: s for s in yaml.safe_load((DATA_DIR / "crosswalk_sources.yaml").read_text(encoding="utf-8"))}


def _links() -> List[dict]:
    out = []
    for name in ("crosswalks.jsonl", "crosswalks.community.jsonl"):
        path = DATA_DIR / name
        if path.exists():
            out += [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


@lru_cache(maxsize=1)
def graph() -> Dict[str, List[dict]]:
    g: Dict[str, List[dict]] = {}
    for l in _links():
        g.setdefault(l["from"], []).append({**l, "other": l["to"]})
        g.setdefault(l["to"], []).append({**l, "other": l["from"]})
    for edges in g.values():
        edges.sort(key=lambda e: (e["kind"] != "official", e["other"]))
    return g


def _describe(node: str) -> dict:
    corpus, _ = regs_corpus()
    prefix = node.split("/", 1)[0]
    group = GROUPS.get(prefix, prefix)
    p = corpus.get(node)
    if p:
        return {"group": group, "ref": p["ref"], "title": p.get("title") or "", "licensed": False}
    return {"group": group, "ref": f"{group} {node.split('/', 1)[1]}", "title": "", "licensed": True}


def _group_of(node: str) -> str:
    return GROUPS.get(node.split("/", 1)[0], node.split("/", 1)[0])


def _walk(starts: List[str], hops: int, official_only: bool) -> List[dict]:
    """Breadth-first: each node keeps its first (shortest, official-first) path; `support` counts the paths
    reaching it at that depth. Nodes in a starting item's own instrument are walked through but not returned
    (siblings reached via a hub such as NIS2 Art. 21(2) are neighbours, not matches)."""
    g, seen, found = graph(), set(starts), []
    own = {_group_of(s) for s in starts}
    frontier = [(s, [], "official") for s in starts]
    for depth in range(1, hops + 1):
        nxt, this_depth = [], {}
        for node, via, kind in frontier:
            for e in g.get(node, []):
                if official_only and e["kind"] != "official":
                    continue
                if e["other"] in this_depth:
                    this_depth[e["other"]]["support"] += 1
                    continue
                if e["other"] in seen:
                    continue
                seen.add(e["other"])
                path_kind = "community" if "community" in (kind, e["kind"]) else "official"
                hit = {"id": e["other"], "kind": path_kind, "source": e["source"], "hops": depth, "via": via,
                       "note": e.get("note"), "support": 1}
                this_depth[e["other"]] = hit
                if _group_of(e["other"]) not in own:
                    found.append(hit)
                if not e["other"].startswith("iso-"):  # licensed ids are leaves
                    nxt.append((e["other"], via + [_describe(e["other"])["ref"]], path_kind))
        frontier = nxt
    return found


def _shape(found: List[dict], per_group: int) -> dict:
    items, more, counts = [], {}, {}
    for f in sorted(found, key=lambda f: (f["kind"] != "official", f["hops"], -f["support"], f["id"])):
        d = _describe(f["id"])
        n = counts[d["group"]] = counts.get(d["group"], 0) + 1
        if n > per_group:
            more[d["group"]] = more.get(d["group"], 0) + 1
            continue
        items.append({**f, **d, "source_title": sources().get(f["source"], {}).get("title", f["source"])})
    return {"items": items, "more": more}


def related(pid: str, hops: int = 2, official_only: bool = False, per_group: int = 4) -> dict:
    return _shape(_walk([pid], hops, official_only), per_group)


def related_many(ids: List[str], hops: int = 2, official_only: bool = False, per_group: int = 4) -> dict:
    return _shape(_walk(list(ids), hops, official_only), per_group)


def linked_ids(ids: List[str], prefix: str, hops: int = 2) -> List[str]:
    return sorted({f["id"] for f in _walk(list(ids), hops, False) if f["id"].startswith(prefix)})
