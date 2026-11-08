#!/usr/bin/env python3
"""
Answer-quality and abstention evaluation (live: Pinecone + Groq).

Runs the real LangGraph pipeline (production defaults) on eval/answers_golden.jsonl
and scores every answer:

  answerable questions   claim recall (reference claims stated by the answer),
                         faithfulness (answer vs retrieved context), citation
                         precision (cited sources that are labelled evidence docs),
                         dangling citations, false-abstain rate
  unanswerable questions correct-abstain rate (deterministic abstention OR the
                         answer declines); anything else is an unsupported answer

Judging uses a DIFFERENT model from the generator (EVAL_JUDGE_MODEL, default
openai/gpt-oss-120b judging gpt-oss-20b) with strict JSON-schema output.
Generation fallbacks are disabled so every answer comes from one known model.

Usage:
    python eval/run_answers.py [--limit N] [--namespace eval]
    python eval/run_answers.py --calibrate eval/labels/judge_calibration_<ts>.csv

The run also writes eval/labels/judge_calibration_<ts>.csv: fill the
human_supported column (true/false) by reading each answer, then --calibrate
reports judge-vs-human agreement and Cohen's kappa.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from eval.answer_metrics import citation_precision, claim_recall, cohen_kappa, judge_answer, summarize  # noqa: E402

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-120b")


def _load(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if not r.get("_meta")]


def _evaluate(entry: Dict[str, Any], graph: Any, judge_llm: Any, namespace: str) -> Dict[str, Any]:
    from app.services.faithfulness import judge_faithfulness_with_usage, verify_citations  # noqa: PLC0415
    from app.services.prompts.rag_prompt import build_context_string  # noqa: PLC0415

    t0 = time.perf_counter()
    state = graph.invoke({
        "query": entry["query"], "namespace": namespace, "top_k": None, "use_web_fallback": False,
        "min_score": None, "max_web_results": 5, "chat_history": [],
    })
    sources = list(state.get("retrieved") or []) + list(state.get("web_results") or [])
    answer = str(state.get("answer") or "")
    row: Dict[str, Any] = {
        "query": entry["query"], "answerable": entry["answerable"], "kind": entry.get("kind"),
        "deterministic_abstain": bool(state.get("insufficient_context")),
        "latency_ms": round((time.perf_counter() - t0) * 1000), "answer": answer[:800],
        "cited_doc_ids": [s.get("doc_id", "")[:8] for s in sources],
    }
    if row["deterministic_abstain"]:
        row.update(abstained=True, abstain_type="deterministic")
        return row

    claims = entry.get("reference_claims") or []
    verdict = judge_answer(judge_llm, entry["query"], answer, claims)
    row["abstained"] = verdict["declined"]
    row["abstain_type"] = "declined" if verdict["declined"] else None
    if entry["answerable"] and not verdict["declined"]:
        faith, _ = judge_faithfulness_with_usage(answer, build_context_string(sources), judge_llm)
        row.update(
            claims=claims,
            claims_supported=verdict["claims_supported"],
            claim_recall=claim_recall(verdict["claims_supported"]),
            faithfulness=faith.faithfulness_score,
            grounded=faith.grounded,
            citation_precision=citation_precision(answer, sources, entry.get("evidence_doc_ids") or []),
            dangling_citations=verify_citations(answer, sources),
        )
    return row


def _write(report: Dict[str, Any], rows: List[Dict[str, Any]], ts: str) -> Path:
    reports = _REPO_ROOT / "eval" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / f"answers_{ts}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    s = report["summary"]
    lines = [
        f"# Answer-quality eval {ts}", "",
        f"Generator: `{report['config']['generator']}` | Judge: `{report['config']['judge']}` | "
        f"Labels: {report['config']['label_status']}", "",
        "| Metric | Value |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in s.items()], "",
        "| # | Answerable | Abstained | Claim recall | Faithfulness | Cite prec. | Query |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['answerable']} | {r.get('abstain_type') or '-'} | {r.get('claim_recall', '-')} | "
            f"{r.get('faithfulness', '-')} | {r.get('citation_precision', '-')} | {r['query'][:70]} |"
        )
    (reports / f"answers_{ts}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels = _REPO_ROOT / "eval" / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    with open(labels / f"judge_calibration_{ts}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "claim", "judge_supported", "human_supported", "answer"])
        for r in rows:
            for claim, sup in zip(r.get("claims") or [], r.get("claims_supported") or []):
                w.writerow([r["query"], claim, str(sup).lower(), "", r["answer"]])
    return json_path


def split_resume(entries: List[Dict[str, Any]], previous_rows: List[Dict[str, Any]]):
    """Return (scored rows to keep by query, entries still to run)."""
    kept = {r["query"]: r for r in previous_rows if "error" not in r}
    return kept, [e for e in entries if e["query"] not in kept]


def calibrate(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        pairs = [(r["judge_supported"] == "true", r["human_supported"].strip().lower() == "true")
                 for r in csv.DictReader(f) if r["human_supported"].strip()]
    if not pairs:
        print("No human labels yet: fill the human_supported column (true/false).")
        return 1
    judge, human = zip(*pairs)
    agree = sum(a == b for a, b in pairs) / len(pairs)
    print(f"labelled={len(pairs)} agreement={agree:.3f} cohen_kappa={cohen_kappa(judge, human):.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--golden", default=str(_REPO_ROOT / "eval" / "answers_golden.jsonl"))
    ap.add_argument("--namespace", default="eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calibrate", type=Path)
    ap.add_argument("--resume", type=Path, help="Previous report: keep its scored rows, re-run errored/missing ones "
                                                "(Groq free tier allows 200k tokens/day per model)")
    args = ap.parse_args()
    if args.calibrate:
        return calibrate(args.calibrate)

    # Eval fidelity: one generator model, patient retries on free-tier 429s.
    # Set before the first get_settings() call in this process.
    os.environ["LLM_FALLBACK_MODELS"] = ""
    os.environ.setdefault("HTTP_MAX_RETRIES", "8")

    from app.core.config import get_settings  # noqa: PLC0415
    from app.services.chat.graph import get_chat_graph  # noqa: PLC0415
    from app.services.llm.groq_llm import _chat_model, configured_models  # noqa: PLC0415
    from app.services.pinecone_store import init_pinecone  # noqa: PLC0415

    settings = get_settings()
    init_pinecone(settings)
    graph, judge_llm = get_chat_graph(), _chat_model(JUDGE_MODEL)
    entries = _load(Path(args.golden))[: args.limit or None]
    meta = json.loads(Path(args.golden).read_text(encoding="utf-8").splitlines()[0])

    kept: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        kept, _ = split_resume(entries, json.loads(args.resume.read_text(encoding="utf-8"))["rows"])
    rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries, 1):
        if entry["query"] in kept:
            rows.append(kept[entry["query"]])
            continue
        for attempt in (1, 2):
            try:
                rows.append(_evaluate(entry, graph, judge_llm, args.namespace))
                break
            except Exception as exc:  # noqa: BLE001 - rate limits etc.; one paced retry
                if attempt == 2:
                    rows.append({"query": entry["query"], "answerable": entry["answerable"],
                                 "abstained": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                else:
                    time.sleep(60)
        r = rows[-1]
        print(f"[{i}/{len(entries)}] {'A' if r['answerable'] else 'U'} "
              f"abstain={r.get('abstain_type') or '-'} recall={r.get('claim_recall', '-')} "
              f"faith={r.get('faithfulness', '-')} {r['query'][:60]}", flush=True)

    scored = [r for r in rows if "error" not in r]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "timestamp": ts,
        "config": {
            "generator": configured_models()[0], "judge": JUDGE_MODEL,
            "reasoning_effort": settings.LLM_REASONING_EFFORT, "namespace": args.namespace,
            "top_k": settings.RAG_DEFAULT_TOP_K, "min_chunk_score": settings.RAG_MIN_CHUNK_SCORE,
            "crag": settings.RAG_CRAG_ENABLED, "contextualize": settings.RAG_CONTEXTUALIZE_ENABLED,
            "label_status": meta.get("label_status", "unknown"),
        },
        "summary": {**summarize(scored), "errors": len(rows) - len(scored)},
        "rows": rows,
    }
    path = _write(report, rows, ts)
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
