#!/usr/bin/env python3
"""
Retrieval evaluation runner.

For each entry in the golden set, calls the existing pinecone_store.search()
(read-only; no upsert, no Groq LLM, no full graph), collects the ranked doc_ids,
and computes recall@k, MRR, and nDCG@k.  Aggregates per-query results and writes
a JSON report + markdown summary to eval/reports/.

Usage:
    python eval/run.py [--namespace eval] [--top-k 10] [--golden eval/golden.jsonl]

Requirements:
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_HOST must be set (or in
    backend/.env).  GROQ_API_KEY is NOT required — no LLM is called here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Path setup — must come before any backend or eval imports.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
sys.path.insert(0, str(_REPO_ROOT))       # for: from eval.metrics import ...
sys.path.insert(0, str(_BACKEND_DIR))     # for: from app... import ...

# Load backend .env so env vars are available before importing backend modules.
try:
    from dotenv import load_dotenv
    load_dotenv(_BACKEND_DIR / ".env", override=False)
except ImportError:
    pass

from eval.metrics import ndcg_at_k, mrr, recall_at_k  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_golden(path: Path) -> List[Dict[str, Any]]:
    """Load golden.jsonl, skipping the _meta header and any non-query entries."""
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


def _extract_ranked_doc_ids(hits: List[Dict[str, Any]]) -> List[str]:
    """Return unique doc_ids from Pinecone hits, preserving retrieval rank order.

    Each Pinecone record has _id = "{doc_id}:{chunk_idx}".  A single document
    may appear as multiple chunks; we keep only the first occurrence (the
    highest-scoring chunk) and work at the document level for metrics.
    The doc_id is read from hit["fields"]["doc_id"] (the authoritative field
    upserted by chunking.py) with a fallback to parsing the _id string.
    """
    seen: set[str] = set()
    ranked: List[str] = []
    for hit in hits:
        fields: Dict[str, Any] = hit.get("fields") or {}
        doc_id: Optional[str] = fields.get("doc_id")
        if not doc_id:
            raw_id: str = hit.get("_id") or ""
            doc_id = raw_id.rsplit(":", 1)[0] if ":" in raw_id else raw_id
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
    return ranked


def _run_retrieval(namespace: str, query: str, top_k: int) -> List[str]:
    """Call the existing pinecone_store.search() and return ranked doc_ids."""
    from app.services.pinecone_store import search as pinecone_search  # noqa: PLC0415

    hits = pinecone_search(
        namespace=namespace,
        query_text=query,
        top_k=top_k,
        filters=None,
        fields=None,
    )
    return _extract_ranked_doc_ids(hits)


def _write_report(report: Dict[str, Any], reports_dir: Path) -> Path:
    """Write JSON + markdown reports; return the JSON path."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = report["timestamp"].replace(":", "-").replace(".", "-")
    json_path = reports_dir / f"{ts}.json"
    md_path = reports_dir / f"{ts}.md"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    k = report["top_k"]
    agg = report["aggregate"]
    lines = [
        f"# Retrieval Eval Report — {report['timestamp']}",
        f"",
        f"Namespace: `{report['namespace']}` | top_k: {k} | Queries: {report['total_queries']}",
        f"",
        f"## Aggregate",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean Recall@{k} | {agg['mean_recall_at_k']:.4f} |",
        f"| Mean MRR | {agg['mean_mrr']:.4f} |",
        f"| Mean nDCG@{k} | {agg['mean_ndcg_at_k']:.4f} |",
        f"",
        f"## Per-Query Breakdown",
        f"",
        f"| # | Query | Recall@{k} | MRR | nDCG@{k} | Retrieved docs |",
        f"|---|-------|-----------|-----|---------|----------------|",
    ]
    for i, pq in enumerate(report["per_query"], start=1):
        q_short = pq["query"][:60] + ("…" if len(pq["query"]) > 60 else "")
        lines.append(
            f"| {i} | {q_short} | {pq['recall_at_k']:.4f} | "
            f"{pq['mrr']:.4f} | {pq['ndcg_at_k']:.4f} | {pq['retrieved_count']} |"
        )

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval evaluation runner — no LLM calls, read-only Pinecone queries."
    )
    parser.add_argument("--namespace", default="eval", help="Pinecone namespace to query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve per query")
    parser.add_argument(
        "--golden",
        default=str(_REPO_ROOT / "eval" / "golden.jsonl"),
        help="Path to golden.jsonl",
    )
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"ERROR: golden set not found at {golden_path}", file=sys.stderr)
        return 1

    entries = _load_golden(golden_path)
    if not entries:
        print("ERROR: no golden entries found (only the _meta header?)", file=sys.stderr)
        return 1

    print(f"Loaded {len(entries)} golden entries from {golden_path}")

    # Initialise Pinecone once.
    from app.core.config import get_settings          # noqa: PLC0415
    from app.services.pinecone_store import init_pinecone  # noqa: PLC0415

    settings = get_settings()
    init_pinecone(settings)

    k = args.top_k
    per_query_results = []
    skipped = 0

    for entry in entries:
        query = entry["query"]
        relevant_ids: set[str] = set(entry.get("relevant_doc_ids") or [])

        # Skip entries where relevant_doc_ids are still placeholders.
        if not relevant_ids or any("PLACEHOLDER" in rid for rid in relevant_ids):
            print(f"  [SKIP] Placeholder entry: {query[:60]}")
            skipped += 1
            continue

        print(f"  [QUERY] {query[:70]}")
        ranked = _run_retrieval(args.namespace, query, k)

        r_at_k = recall_at_k(ranked, relevant_ids, k)
        rr = mrr(ranked, relevant_ids)
        ndcg = ndcg_at_k(ranked, relevant_ids, k)

        per_query_results.append({
            "query": query,
            "retrieved_count": len(ranked),
            "retrieved_doc_ids": ranked[:k],
            "relevant_doc_ids": sorted(relevant_ids),
            "recall_at_k": round(r_at_k, 6),
            "mrr": round(rr, 6),
            "ndcg_at_k": round(ndcg, 6),
        })
        print(f"         recall@{k}={r_at_k:.4f}  MRR={rr:.4f}  nDCG@{k}={ndcg:.4f}")

    if not per_query_results:
        print(
            f"WARNING: all {skipped} entries were skipped (placeholder doc_ids). "
            "Run `make eval-corpus`, read the printed doc_ids, label relevant ones "
            "by document content, then fill in eval/golden.jsonl.",
            file=sys.stderr,
        )
        return 0

    n = len(per_query_results)
    aggregate = {
        "mean_recall_at_k": round(sum(r["recall_at_k"] for r in per_query_results) / n, 6),
        "mean_mrr": round(sum(r["mrr"] for r in per_query_results) / n, 6),
        "mean_ndcg_at_k": round(sum(r["ndcg_at_k"] for r in per_query_results) / n, 6),
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "timestamp": ts,
        "namespace": args.namespace,
        "top_k": k,
        "total_queries": n,
        "skipped_entries": skipped,
        "aggregate": aggregate,
        "per_query": per_query_results,
    }

    reports_dir = _REPO_ROOT / "eval" / "reports"
    json_path = _write_report(report, reports_dir)

    print(f"\n--- Aggregate (n={n} queries, top_k={k}) ---")
    print(f"  Mean Recall@{k} : {aggregate['mean_recall_at_k']:.4f}")
    print(f"  Mean MRR        : {aggregate['mean_mrr']:.4f}")
    print(f"  Mean nDCG@{k}   : {aggregate['mean_ndcg_at_k']:.4f}")
    print(f"\nReport written to: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
