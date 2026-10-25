#!/usr/bin/env python3
"""Snapshot the latest eval reports into eval/results/ (committed) for the UI's Evaluation and Home pages.

eval/reports/ is gitignored scratch; eval/results/ is what the deployed app reads.
Usage: python eval/publish_results.py
"""
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import List, Optional

REPORTS = Path(__file__).resolve().parent / "reports"
RESULTS = REPORTS.parent / "results"
REGS_DATA = REPORTS.parents[1] / "backend" / "app" / "data" / "regs"
STAMP = r"\d{8}T\d{6}Z\.json$"
KINDS = {
    "retrieval": re.compile(rf"^{STAMP}"),
    "sweep": re.compile(rf"^sweep_{STAMP}"),
    "rerank_ab": re.compile(rf"^ab_{STAMP}"),
    "answers": re.compile(rf"^answers_{STAMP}"),
    "adversarial": re.compile(rf"^adversarial_{STAMP}"),
    "regs_abstain": re.compile(rf"^regs_abstain_{STAMP}"),
    "router": re.compile(rf"^router_{STAMP}"),
}
MODES = ["lexical", "dense", "fused", "diversified", "diversified-boost", "diversified-inject"]
REGS_K = 6  # provisions sent to the model on the compliance path (REGS_TOP_K)
INSTRUMENTS = {"eu-ai-act": "EU AI Act", "gdpr": "GDPR", "nis2": "NIS2",
               "nis2-ir": "NIS2 Implementing Regulation 2024/2690", "dora": "DORA", "cra": "Cyber Resilience Act",
               "nist-800-53": "NIST SP 800-53", "nist-csf": "NIST CSF 2.0", "nist-ai-rmf": "NIST AI RMF"}


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _runs(reports: Path, prefix: str) -> List[Path]:
    pat = re.compile(rf"^{re.escape(prefix)}_{STAMP}")
    return sorted(p for p in reports.glob("*.json") if pat.match(p.name))


def _at_k(reports: Path, mode: str) -> List[dict]:
    return [r for r in map(_load, _runs(reports, f"regs_{mode}")) if r.get("k") == REGS_K]


def regs_retrieval(reports: Path) -> Optional[dict]:
    """The latest production-mode (diversified) retrieval run at k = 6, with per-type and per-question rows."""
    runs = _at_k(reports, "diversified")
    return runs[-1] if runs else None


def regs_modes(reports: Path) -> dict:
    """Latest run at k = 6 per retrieval mode, for the comparison chart."""
    out = {}
    for mode in MODES:
        runs = _at_k(reports, mode)
        if runs:
            agg = runs[-1]["aggregate"]
            out[mode] = {"n": runs[-1]["n"], **{m: agg[m] for m in ("recall", "mrr", "ndcg") if m in agg}}
    return out


def gapcheck_runs(reports: Path) -> Optional[dict]:
    """The latest golden gap check, plus the headline numbers of every run of the same prompt (the judge is a
    live model, so one run is not the whole story)."""
    paths = _runs(reports, "gapcheck")
    if not paths:
        return None
    latest, runs = _load(paths[-1]), []
    for p in paths:
        r = _load(p)
        if r.get("prompt_sha") == latest.get("prompt_sha"):
            runs.append({"run": p.stem.split("_", 1)[1],
                         **{k: r[k] for k in ("accuracy", "verified_rate", "missing_caught", "missing_planted", "tokens")},
                         "false_covered": len(r["false_covered"])})
    return {**latest, "runs": runs}


def corpus_summary(data_dir: Path) -> List[dict]:
    """Per instrument: binding, provision count, official source and retrieval date (Home page)."""
    manifest = {m["key"]: m for m in _load(data_dir / "manifest.json")}
    counts, binding = Counter(), {}
    for line in (data_dir / "regs-v1.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            key = p["id"].split("/", 1)[0]
            counts[key] += 1
            binding[key] = p["binding"]
    return [{"key": k, "name": name, "binding": binding[k], "provisions": counts[k], "url": manifest[k]["url"],
             "retrieved": manifest[k]["retrieved"]} for k, name in INSTRUMENTS.items() if k in counts]


def publish(reports: Path = REPORTS, results: Path = RESULTS, data_dir: Path = REGS_DATA) -> List[str]:
    results.mkdir(exist_ok=True)
    done = []
    for kind, pattern in KINDS.items():
        candidates = sorted(p for p in reports.glob("*.json") if pattern.match(p.name))
        if kind == "rerank_ab":  # the multi-k A/B is the one the design decision cites
            candidates = [p for p in candidates if _load(p).get("multi_k")]
        if candidates:
            shutil.copyfile(candidates[-1], results / f"{kind}.json")
            done.append(kind)
    built = {"regs_retrieval": regs_retrieval(reports), "regs_modes": regs_modes(reports) or None,
             "gapcheck": gapcheck_runs(reports), "corpus": corpus_summary(data_dir)}
    for kind, data in built.items():
        if data:
            (results / f"{kind}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            done.append(kind)
    return done


if __name__ == "__main__":
    print("published:", ", ".join(publish()))
