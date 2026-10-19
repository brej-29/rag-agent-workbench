"""Shape the committed eval snapshots (eval/results/*.json) for the Evaluation page.

Pure functions, no Streamlit: unit-tested in tests/test_ui_eval_data.py.
Refresh the snapshots with `python eval/publish_results.py` after an eval run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

RESULTS_DIR = Path(__file__).resolve().parents[2] / "eval" / "results"
_SWEEP_METRICS = {"mean_recall_at_k": "Recall@k", "mean_ndcg_at_k": "nDCG@k", "mean_precision_at_k": "Precision@k"}


def load_results(directory: Path = RESULTS_DIR) -> Dict[str, Any]:
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
