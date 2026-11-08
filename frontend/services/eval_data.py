"""Shape the committed eval snapshots (eval/results/*.json) for the Evaluation page.

Pure functions, no Streamlit: unit-tested in tests/test_ui_eval_data.py.
Refresh the snapshots with `python eval/publish_results.py` after an eval run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RESULTS_DIR = Path(__file__).resolve().parents[2] / "eval" / "results"
_SWEEP_METRICS = {"mean_recall_at_k": "Recall@k", "mean_ndcg_at_k": "nDCG@k", "mean_precision_at_k": "Precision@k"}


def load_results(directory: Optional[Path] = None) -> Dict[str, Any]:
    directory = directory or RESULTS_DIR  # read at call time, so the UI smoke test can point it elsewhere
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))}


def sweep_rows(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Long format for a line chart: one row per (k, metric)."""
    rows = []
    for k in sorted(sweep["aggregate"], key=int):
        for key, label in _SWEEP_METRICS.items():
            if key in sweep["aggregate"][k]:
                rows.append({"k": int(k), "metric": label, "value": round(sweep["aggregate"][k][key], 4)})
    return rows


def _label(key: str) -> str:
    m = re.match(r"mean_(\w+?)_at_(\d+)$", key)
    if m:
        name = {"recall": "Recall", "ndcg": "nDCG", "precision": "Precision"}.get(m.group(1), m.group(1))
        return f"{name}@{m.group(2)}"
    return {"mean_mrr": "MRR", "mean_recall_at_k": "Recall@10", "mean_ndcg_at_k": "nDCG@10"}.get(key, key)


def ab_deltas(ab: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """(rerank - baseline per quality metric, extra latency in ms).

    Latency is returned separately: it is not on the 0-1 quality scale, so it never
    shares an axis with the deltas.
    """
    base, rerank = ab["aggregate"]["baseline"], ab["aggregate"]["rerank"]
    deltas = [
        {"metric": _label(k), "delta": round(rerank[k] - base[k], 3)}
        for k in base if k != "mean_latency_ms" and k in rerank
    ]
    return deltas, round(rerank.get("mean_latency_ms", 0) - base.get("mean_latency_ms", 0))


def abstention_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    scored = [r for r in rows if "error" not in r]
    return {
        "answered_ok": sum(1 for r in scored if r["answerable"] and not r["abstained"]),
        "false_abstain": sum(1 for r in scored if r["answerable"] and r["abstained"]),
        "correct_abstain": sum(1 for r in scored if not r["answerable"] and r["abstained"]),
        "answered_unanswerable": sum(1 for r in scored if not r["answerable"] and not r["abstained"]),
        "errors": len(rows) - len(scored),
    }


# ---------------------------------------------------------------------------
# Compliance evals (sub-project 6)
# ---------------------------------------------------------------------------
MODE_LABELS = {"lexical": "Lexical (BM25)", "dense": "Dense", "fused": "Fused (RRF)",
               "diversified": "Diversified (production)", "diversified-boost": "+ crosswalk boost",
               "diversified-inject": "+ crosswalk inject"}
_TYPES = {"concept": "About a topic", "cross": "Across two frameworks", "reference": "Naming a provision"}
_JOBS = {"ask": "Ask", "checklist": "Checklist", "gap_check": "Gap check", "clarify": "Clarify"}  # route -> page name
_BINDING = {"regulation": "Binding regulation", "directive": "Binding directive (through national law)",
            "voluntary_framework": "Voluntary framework"}


def mode_rows(modes: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One bar per retrieval mode, in a fixed order (never sorted by value); production is marked."""
    return [{"mode": label, "recall": round(modes[k]["recall"], 3), "mrr": round(modes[k]["mrr"], 3),
             "n": modes[k]["n"], "production": k == "diversified"} for k, label in MODE_LABELS.items() if k in modes]


def type_rows(retrieval: Dict[str, Any]) -> List[Dict[str, Any]]:
    k = retrieval["k"]
    return [{"Question type": _TYPES.get(t, t), "Questions": v["n"], f"Recall@{k}": round(v["recall"], 3),
             "MRR": round(v["mrr"], 3), f"nDCG@{k}": round(v["ndcg"], 3)}
            for t, v in retrieval["aggregate"]["by_type"].items()]


def job_name(route: str) -> str:
    return _JOBS.get(route, route)


def confusion_rows(confusion: Dict[str, int]) -> List[Dict[str, Any]]:
    """{'gap_check->ask': 2} -> one row per request type, one column per job it was routed to, in page names."""
    seen = {part for key in confusion for part in key.split("->")}
    routes = [r for r in _JOBS if r in seen] + sorted(seen - set(_JOBS))
    table = {a: {b: 0 for b in routes} for a in routes}
    for key, n in confusion.items():
        a, b = key.split("->")
        table[a][b] = n
    return [{"Request": job_name(a), **{job_name(b): table[a][b] for b in routes}} for a in routes]


_DISAGREEMENT = re.compile(r"^(\w+):(\S+) expected (\w+), got (\w+)$")


def disagreement_line(text: str) -> str:
    """'incident:gdpr/art-34/par-1 expected partial, got unverified' as a readable line (Markdown)."""
    m = _DISAGREEMENT.match(text)
    if not m:
        return text
    name, pid, want, got = m.groups()
    return f"{name.replace('_', ' ').capitalize()} · `{pid}`: labelled {want}, judged {got}"


def _count(x: Any) -> int:
    return x if isinstance(x, int) else len(x)


def gap_range(gap: Dict[str, Any]) -> Dict[str, Any]:
    runs = gap.get("runs") or [gap]
    acc, ver = [r["accuracy"] for r in runs], [r["verified_rate"] for r in runs]
    return {"runs": len(runs), "accuracy": (min(acc), max(acc)), "verified_rate": (min(ver), max(ver)),
            "false_covered": max(_count(r["false_covered"]) for r in runs)}


def span(pair: Tuple[float, float], pct: bool = False) -> str:
    fmt = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.2f}")
    lo, hi = fmt(pair[0]), fmt(pair[1])
    return lo if lo == hi else f"{lo}–{hi}"


def headline(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Home page: the measured numbers that matter most, each with where it comes from."""
    out = []
    r = results.get("regs_retrieval")
    if r:
        out.append({"label": f"Right provision in the top {r['k']}", "value": f"{r['aggregate']['recall']:.0%}",
                    "note": f"Recall@{r['k']} on {r['n']} golden questions"})
    a = results.get("regs_abstain")
    if a:
        out.append({"label": "Out-of-scope questions declined", "value": f"{a['correct_abstain']} of {a['unanswerable']}",
                    "note": f"{a['false_abstain']} of {a['answerable']} in-scope questions wrongly declined"})
    rt = results.get("router")
    if rt:
        out.append({"label": "Requests routed to the right job", "value": f"{rt['accuracy']:.0%}",
                    "note": f"{rt['n']} labelled requests"})
    g = results.get("gapcheck")
    if g:
        rng = gap_range(g)
        out.append({"label": "Gap check: planted gaps called covered", "value": str(rng["false_covered"]),
                    "note": f"Across {rng['runs']} runs; quotes verified {span(rng['verified_rate'], pct=True)}"})
    return out


def coverage_rows(corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"Instrument": c["name"], "Type": _BINDING.get(c["binding"], c["binding"]), "Provisions": c["provisions"],
             "Source": c["url"], "Retrieved": c["retrieved"]} for c in corpus]
