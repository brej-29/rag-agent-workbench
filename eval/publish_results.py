#!/usr/bin/env python3
"""Snapshot the latest eval reports into eval/results/ (committed) for the UI's Evaluation page.

eval/reports/ is gitignored scratch; eval/results/ is what the deployed app reads.
Usage: python eval/publish_results.py
"""
import json
import re
import shutil
from pathlib import Path

REPORTS = Path(__file__).resolve().parent / "reports"
RESULTS = REPORTS.parent / "results"
KINDS = {
    "retrieval": re.compile(r"^\d{8}T\d{6}Z\.json$"),
    "sweep": re.compile(r"^sweep_\d{8}T\d{6}Z\.json$"),
    "rerank_ab": re.compile(r"^ab_\d{8}T\d{6}Z\.json$"),
    "answers": re.compile(r"^answers_\d{8}T\d{6}Z\.json$"),
    "adversarial": re.compile(r"^adversarial_\d{8}T\d{6}Z\.json$"),
}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for kind, pattern in KINDS.items():
        candidates = sorted(p for p in REPORTS.glob("*.json") if pattern.match(p.name))
        if kind == "rerank_ab":  # the multi-k A/B is the one the design decision cites
            candidates = [p for p in candidates if json.loads(p.read_text(encoding="utf-8")).get("multi_k")]
        if candidates:
            shutil.copyfile(candidates[-1], RESULTS / f"{kind}.json")
            print(f"{kind:12s} <- {candidates[-1].name}")


if __name__ == "__main__":
    main()
