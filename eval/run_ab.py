#!/usr/bin/env python3
"""
A/B retrieval evaluation — baseline (dense only) vs rerank (dense + Pinecone hosted reranker).

Runs each golden entry twice:
  Arm A — baseline : dense retrieve top_k, no reranking
  Arm B — rerank   : dense retrieve CANDIDATES, apply cosine floor, rerank, take top_k

Produces a side-by-side delta table (recall@k, MRR, nDCG@k, mean latency) written
to eval/reports/ab_{timestamp}.{json,md}.

Usage:
    python eval/run_ab.py --namespace eval --top-k 10 --candidates 20

    # Override the reranker model (must be available on your Pinecone plan):
    python eval/run_ab.py --namespace eval --top-k 10 --candidates 30 \\
        --rerank-model pinecone-rerank-v0

Requirements:
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_HOST must be set (or in backend/.env).
    RAG_MIN_CHUNK_SCORE is read from settings (defaults to 0.25) — the same cosine
    floor that the production pipeline applies before reranking.

IMPORTANT — live inference, not CI:
    This script issues LIVE Pinecone query + rerank inference calls (read-only, no
    upserts) and incurs cost per query × arm.  Invoke via `make eval-ab` when you
    need an empirical justification to flip RAG_RERANK_ENABLED=True.
    DO NOT add this to CI.  DO NOT call it from `make eval`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — must come before any backend or eval imports.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_BACKEND_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(_BACKEND_DIR / ".env", override=False)
except ImportError:
    pass

from eval.metrics import ndcg_at_k, mrr, precision_at_k, recall_at_k  # noqa: E402


# ---------------------------------------------------------------------------
# Hit / chunk helpers
# ---------------------------------------------------------------------------

def _hits_to_chunks(hits: List[Dict[str, Any]], text_field: str = "chunk_text") -> List[Dict[str, Any]]:
    """Convert raw Pinecone hits into the chunk-dict format expected by rerank_chunks."""
    chunks: List[Dict[str, Any]] = []
    for hit in hits:
        fields: Dict[str, Any] = hit.get("fields") or {}
        raw_id: str = hit.get("_id") or ""
        doc_id: str = fields.get("doc_id") or (
            raw_id.rsplit(":", 1)[0] if ":" in raw_id else raw_id
        )
        chunks.append(
            {
                "_id": raw_id,
                "doc_id": doc_id,
                "chunk_text": str(fields.get(text_field) or ""),
                "score": float(hit.get("_score") or hit.get("score") or 0.0),
                "title": str(fields.get("title") or ""),
                "source": str(fields.get("source") or ""),
                "url": str(fields.get("url") or ""),
            }
        )
    return chunks


def _extract_doc_ids_from_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
    """Unique doc_ids from chunk-dict list, preserving rank order."""
    seen: set[str] = set()
    result: List[str] = []
    for chunk in chunks:
        doc_id: str = chunk.get("doc_id") or ""
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
    return result


def _extract_doc_ids_from_hits(hits: List[Dict[str, Any]]) -> List[str]:
    """Unique doc_ids from raw Pinecone hits, preserving rank order."""
    seen: set[str] = set()
    result: List[str] = []
    for hit in hits:
        fields: Dict[str, Any] = hit.get("fields") or {}
        doc_id: Optional[str] = fields.get("doc_id")
        if not doc_id:
            raw_id: str = hit.get("_id") or ""
            doc_id = raw_id.rsplit(":", 1)[0] if ":" in raw_id else raw_id
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
    return result


# ---------------------------------------------------------------------------
# Arm runners
# ---------------------------------------------------------------------------

def _run_baseline_arm(
    namespace: str, query: str, top_k: int
) -> Tuple[List[str], float]:
    """Arm A: standard dense retrieval, no reranking.

    Returns (ranked_doc_ids[:top_k], elapsed_ms).
    """
    from app.services.pinecone_store import search as pinecone_search  # noqa: PLC0415

    start = perf_counter()
    hits = pinecone_search(namespace=namespace, query_text=query, top_k=top_k,
                           filters=None, fields=None)
    elapsed_ms = (perf_counter() - start) * 1000.0
    ranked = _extract_doc_ids_from_hits(hits)
    return ranked[:top_k], elapsed_ms


def _run_rerank_arm(
    namespace: str,
    query: str,
    top_k: int,
    candidates: int,
    model: str,
    min_chunk_score: float,
    text_field: str = "chunk_text",
) -> Tuple[List[str], float]:
    """Arm B: dense retrieve CANDIDATES, cosine floor, hosted rerank, take top_k.

    The cosine floor (min_chunk_score) mirrors the production pipeline's
    RAG_MIN_CHUNK_SCORE gate — it runs on COSINE scores, BEFORE reranking.
    Rerank scores are never compared to this threshold.

    Returns (ranked_doc_ids[:top_k], total_elapsed_ms).
    """
    from app.services.pinecone_store import search as pinecone_search  # noqa: PLC0415
    from app.services.rerank import rerank_chunks  # noqa: PLC0415

    start = perf_counter()

    # Stage 1: dense retrieval with wider pool
    hits = pinecone_search(namespace=namespace, query_text=query,
                           top_k=candidates, filters=None, fields=None)
    retrieve_ms = (perf_counter() - start) * 1000.0

    # Convert to chunk dicts
    chunks = _hits_to_chunks(hits, text_field=text_field)

    # Cosine floor — same threshold as production RAG_MIN_CHUNK_SCORE
    # Applied on cosine scores before reranking (not on rerank scores)
    survivors = [c for c in chunks if c["score"] >= min_chunk_score]

    if not survivors:
        total_ms = (perf_counter() - start) * 1000.0
        return [], total_ms

    # Stage 2: hosted rerank
    reranked = rerank_chunks(query=query, chunks=survivors, top_n=top_k, model=model)
    total_ms = (perf_counter() - start) * 1000.0

    ranked = _extract_doc_ids_from_chunks(reranked)
    return ranked[:top_k], total_ms


# ---------------------------------------------------------------------------
# Golden loader (mirrors run.py)
# ---------------------------------------------------------------------------

def _load_golden(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_meta" in obj or "query" not in obj:
                continue
            entries.append(obj)
    return entries


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_ab_report(report: Dict[str, Any], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = report["timestamp"].replace(":", "-").replace(".", "-")
    json_path = reports_dir / f"ab_{ts}.json"
    md_path = reports_dir / f"ab_{ts}.md"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    k = report["top_k"]
    cands = report["candidates"]
    a = report["aggregate"]["baseline"]
    b = report["aggregate"]["rerank"]

    def _delta(b_val: float, a_val: float) -> str:
        d = b_val - a_val
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    multi_k_flag: bool = bool(report.get("multi_k"))

    lines = [
        f"# A/B Retrieval Eval - {report['timestamp']}",
        f"",
        f"Namespace: `{report['namespace']}` | top_k: {k} | candidates: {cands}",
        f"Reranker: `{report['rerank_model']}` | Queries evaluated: {report['total_queries']}",
        f"Cosine floor (RAG_MIN_CHUNK_SCORE): {report['min_chunk_score']}",
        f"",
        f"## Aggregate Delta",
        f"",
        f"| Metric | Baseline (A) | Rerank (B) | D (B-A) |",
        f"|--------|-------------|-----------|---------|",
    ]
    if multi_k_flag:
        lines += [
            f"| Mean Precision@1 | {a['mean_precision_at_1']:.4f} | {b['mean_precision_at_1']:.4f} |"
            f" {_delta(b['mean_precision_at_1'], a['mean_precision_at_1'])} |",
            f"| Mean Recall@3 | {a['mean_recall_at_3']:.4f} | {b['mean_recall_at_3']:.4f} |"
            f" {_delta(b['mean_recall_at_3'], a['mean_recall_at_3'])} |",
            f"| Mean nDCG@3 | {a['mean_ndcg_at_3']:.4f} | {b['mean_ndcg_at_3']:.4f} |"
            f" {_delta(b['mean_ndcg_at_3'], a['mean_ndcg_at_3'])} |",
            f"| Mean Recall@5 | {a['mean_recall_at_5']:.4f} | {b['mean_recall_at_5']:.4f} |"
            f" {_delta(b['mean_recall_at_5'], a['mean_recall_at_5'])} |",
            f"| Mean nDCG@5 | {a['mean_ndcg_at_5']:.4f} | {b['mean_ndcg_at_5']:.4f} |"
            f" {_delta(b['mean_ndcg_at_5'], a['mean_ndcg_at_5'])} |",
        ]
    lines += [
        f"| Mean Recall@{k} | {a['mean_recall_at_k']:.4f} | {b['mean_recall_at_k']:.4f} |"
        f" {_delta(b['mean_recall_at_k'], a['mean_recall_at_k'])} |",
        f"| Mean MRR | {a['mean_mrr']:.4f} | {b['mean_mrr']:.4f} |"
        f" {_delta(b['mean_mrr'], a['mean_mrr'])} |",
        f"| Mean nDCG@{k} | {a['mean_ndcg_at_k']:.4f} | {b['mean_ndcg_at_k']:.4f} |"
        f" {_delta(b['mean_ndcg_at_k'], a['mean_ndcg_at_k'])} |",
        f"| Mean latency (ms) | {a['mean_latency_ms']:.1f} | {b['mean_latency_ms']:.1f} |"
        f" {_delta(b['mean_latency_ms'], a['mean_latency_ms'])} |",
        f"",
        f"> Enable `RAG_RERANK_ENABLED=true` only when D nDCG@{k} > 0 at acceptable latency cost.",
        f"",
        f"## Per-Query Breakdown",
        f"",
        f"| # | Query | RecA | RecB | MRRA | MRRB | nDCGA | nDCGB |",
        f"|---|-------|------|------|------|------|-------|-------|",
    ]
    for i, pq in enumerate(report["per_query"], start=1):
        q = pq["query"][:50] + ("…" if len(pq["query"]) > 50 else "")
        ba = pq["baseline"]
        br = pq["rerank"]
        lines.append(
            f"| {i} | {q} "
            f"| {ba['recall_at_k']:.3f} | {br['recall_at_k']:.3f} "
            f"| {ba['mrr']:.3f} | {br['mrr']:.3f} "
            f"| {ba['ndcg_at_k']:.3f} | {br['ndcg_at_k']:.3f} |"
        )

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A/B retrieval eval: baseline dense vs Pinecone hosted reranker. "
            "Issues LIVE Pinecone calls — NOT for CI."
        )
    )
    parser.add_argument("--namespace", default="eval", help="Pinecone namespace")
    parser.add_argument("--top-k", type=int, default=10, help="Final chunks delivered to LLM")
    parser.add_argument("--candidates", type=int, default=20,
                        help="First-stage pool for rerank arm (>= top_k)")
    parser.add_argument("--golden", default=str(_REPO_ROOT / "eval" / "golden.jsonl"))
    parser.add_argument("--rerank-model", default="bge-reranker-v2-m3",
                        help="Pinecone inference reranker model")
    parser.add_argument(
        "--multi-k", action="store_true", default=False,
        help=(
            "Compute top-heavy metrics at multiple k values: "
            "precision@1, recall@3/5, nDCG@3/5 (in addition to the standard @top_k metrics). "
            "Requires top_k >= 5.  Used by the eval-ab-topk Makefile target."
        ),
    )
    args = parser.parse_args()

    if args.candidates < args.top_k:
        print(
            f"WARNING: --candidates ({args.candidates}) < --top-k ({args.top_k}); "
            f"clamping candidates to {args.top_k}.",
            file=sys.stderr,
        )
        args.candidates = args.top_k

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"ERROR: golden set not found at {golden_path}", file=sys.stderr)
        return 1

    entries = _load_golden(golden_path)
    if not entries:
        print("ERROR: no golden entries found.", file=sys.stderr)
        return 1

    print(
        f"Loaded {len(entries)} golden entries | "
        f"namespace={args.namespace} top_k={args.top_k} candidates={args.candidates} "
        f"model={args.rerank_model}"
    )

    from app.core.config import get_settings              # noqa: PLC0415
    from app.services.pinecone_store import init_pinecone  # noqa: PLC0415

    settings = get_settings()
    init_pinecone(settings)

    min_chunk_score = settings.RAG_MIN_CHUNK_SCORE
    text_field = settings.PINECONE_TEXT_FIELD

    k = args.top_k
    per_query_results: List[Dict[str, Any]] = []
    skipped = 0

    for entry in entries:
        query: str = entry["query"]
        relevant_ids: set[str] = set(entry.get("relevant_doc_ids") or [])

        if not relevant_ids or any("PLACEHOLDER" in rid for rid in relevant_ids):
            print(f"  [SKIP] {query[:60]}")
            skipped += 1
            continue

        print(f"  [QUERY] {query[:70]}")

        # Arm A — baseline
        ranked_a, lat_a = _run_baseline_arm(args.namespace, query, k)
        rec_a  = recall_at_k(ranked_a, relevant_ids, k)
        mrr_a  = mrr(ranked_a, relevant_ids)
        ndcg_a = ndcg_at_k(ranked_a, relevant_ids, k)

        # Arm B — rerank
        ranked_b, lat_b = _run_rerank_arm(
            args.namespace, query, k, args.candidates,
            args.rerank_model, min_chunk_score, text_field,
        )
        rec_b  = recall_at_k(ranked_b, relevant_ids, k)
        mrr_b  = mrr(ranked_b, relevant_ids)
        ndcg_b = ndcg_at_k(ranked_b, relevant_ids, k)

        baseline_entry: Dict[str, Any] = {
            "ranked_doc_ids": ranked_a,
            "recall_at_k": round(rec_a, 6),
            "mrr": round(mrr_a, 6),
            "ndcg_at_k": round(ndcg_a, 6),
            "latency_ms": round(lat_a, 2),
        }
        rerank_entry: Dict[str, Any] = {
            "ranked_doc_ids": ranked_b,
            "recall_at_k": round(rec_b, 6),
            "mrr": round(mrr_b, 6),
            "ndcg_at_k": round(ndcg_b, 6),
            "latency_ms": round(lat_b, 2),
        }

        if args.multi_k:
            # Top-heavy metrics — same ranked lists, no additional Pinecone calls.
            # These are the metrics reranking actually optimises for (top-of-list precision).
            p1_a   = precision_at_k(ranked_a, relevant_ids, 1)
            rec3_a = recall_at_k(ranked_a, relevant_ids, 3)
            rec5_a = recall_at_k(ranked_a, relevant_ids, 5)
            ndcg3_a = ndcg_at_k(ranked_a, relevant_ids, 3)
            ndcg5_a = ndcg_at_k(ranked_a, relevant_ids, 5)

            p1_b   = precision_at_k(ranked_b, relevant_ids, 1)
            rec3_b = recall_at_k(ranked_b, relevant_ids, 3)
            rec5_b = recall_at_k(ranked_b, relevant_ids, 5)
            ndcg3_b = ndcg_at_k(ranked_b, relevant_ids, 3)
            ndcg5_b = ndcg_at_k(ranked_b, relevant_ids, 5)

            baseline_entry.update({
                "precision_at_1": round(p1_a, 6),
                "recall_at_3": round(rec3_a, 6),
                "recall_at_5": round(rec5_a, 6),
                "ndcg_at_3": round(ndcg3_a, 6),
                "ndcg_at_5": round(ndcg5_a, 6),
            })
            rerank_entry.update({
                "precision_at_1": round(p1_b, 6),
                "recall_at_3": round(rec3_b, 6),
                "recall_at_5": round(rec5_b, 6),
                "ndcg_at_3": round(ndcg3_b, 6),
                "ndcg_at_5": round(ndcg5_b, 6),
            })

        per_query_results.append(
            {
                "query": query,
                "relevant_doc_ids": sorted(relevant_ids),
                "baseline": baseline_entry,
                "rerank": rerank_entry,
            }
        )
        print(
            f"         A: recall={rec_a:.4f} mrr={mrr_a:.4f} ndcg={ndcg_a:.4f} lat={lat_a:.0f}ms"
            f" | B: recall={rec_b:.4f} mrr={mrr_b:.4f} ndcg={ndcg_b:.4f} lat={lat_b:.0f}ms"
        )

    if not per_query_results:
        print(
            f"WARNING: all {skipped} entries were placeholder — label eval/golden.jsonl first.",
            file=sys.stderr,
        )
        return 0

    n = len(per_query_results)

    def _agg(arm: str) -> Dict[str, float]:
        base: Dict[str, float] = {
            "mean_recall_at_k": round(sum(r[arm]["recall_at_k"] for r in per_query_results) / n, 6),
            "mean_mrr": round(sum(r[arm]["mrr"] for r in per_query_results) / n, 6),
            "mean_ndcg_at_k": round(sum(r[arm]["ndcg_at_k"] for r in per_query_results) / n, 6),
            "mean_latency_ms": round(sum(r[arm]["latency_ms"] for r in per_query_results) / n, 2),
        }
        if args.multi_k:
            base.update({
                "mean_precision_at_1": round(
                    sum(r[arm]["precision_at_1"] for r in per_query_results) / n, 6),
                "mean_recall_at_3": round(
                    sum(r[arm]["recall_at_3"] for r in per_query_results) / n, 6),
                "mean_recall_at_5": round(
                    sum(r[arm]["recall_at_5"] for r in per_query_results) / n, 6),
                "mean_ndcg_at_3": round(
                    sum(r[arm]["ndcg_at_3"] for r in per_query_results) / n, 6),
                "mean_ndcg_at_5": round(
                    sum(r[arm]["ndcg_at_5"] for r in per_query_results) / n, 6),
            })
        return base

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: Dict[str, Any] = {
        "timestamp": ts,
        "namespace": args.namespace,
        "top_k": k,
        "candidates": args.candidates,
        "rerank_model": args.rerank_model,
        "min_chunk_score": min_chunk_score,
        "multi_k": args.multi_k,
        "total_queries": n,
        "skipped_entries": skipped,
        "aggregate": {
            "baseline": _agg("baseline"),
            "rerank": _agg("rerank"),
        },
        "per_query": per_query_results,
    }

    reports_dir = _REPO_ROOT / "eval" / "reports"
    json_path = _write_ab_report(report, reports_dir)

    a = report["aggregate"]["baseline"]
    b = report["aggregate"]["rerank"]

    print(f"\n--- A/B Aggregate (n={n} queries, top_k={k}, candidates={args.candidates}) ---")
    print(f"{'Metric':<22} {'Baseline (A)':>14} {'Rerank (B)':>12} {'D (B-A)':>10}")
    print("-" * 62)

    standard_rows = [
        (f"Mean Recall@{k}",   a["mean_recall_at_k"], b["mean_recall_at_k"]),
        ("Mean MRR",           a["mean_mrr"],          b["mean_mrr"]),
        (f"Mean nDCG@{k}",     a["mean_ndcg_at_k"],   b["mean_ndcg_at_k"]),
        ("Mean latency (ms)",  a["mean_latency_ms"],   b["mean_latency_ms"]),
    ]

    if args.multi_k:
        top_heavy_rows = [
            ("Mean Precision@1",  a["mean_precision_at_1"], b["mean_precision_at_1"]),
            ("Mean Recall@3",     a["mean_recall_at_3"],    b["mean_recall_at_3"]),
            ("Mean nDCG@3",       a["mean_ndcg_at_3"],      b["mean_ndcg_at_3"]),
            ("Mean Recall@5",     a["mean_recall_at_5"],    b["mean_recall_at_5"]),
            ("Mean nDCG@5",       a["mean_ndcg_at_5"],      b["mean_ndcg_at_5"]),
        ]
        print("  --- top-heavy (reranking target) ---")
        for label, ak, bk in top_heavy_rows:
            delta = bk - ak
            sign = "+" if delta >= 0 else ""
            print(f"  {label:<20} {ak:>14.4f} {bk:>12.4f} {sign}{delta:>+9.4f}")
        print("  --- standard continuity ---")

    for label, ak, bk in standard_rows:
        delta = b_val - a_val if (b_val := bk) is not None and (a_val := ak) is not None else 0  # noqa
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<20} {ak:>14.4f} {bk:>12.4f} {sign}{delta:>+9.4f}")

    print(f"\nReport written to: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
