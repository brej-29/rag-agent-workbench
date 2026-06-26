#!/usr/bin/env python3
"""
Top-k sweep evaluation.

Retrieves CHUNK_FETCH_K (default 20) chunks ONCE per query using the
dense-only path (rerank OFF), deduplicates to a doc-ranked list, then
computes recall@k, nDCG@k, and precision@k for each k in SWEEP_K_VALUES
= [1, 2, 3, 5, 8, 10].  One Pinecone call per golden query (not per k).

Purpose
-------
The corpus is ceiling-bound at the production setting: baseline recall@10=0.97,
nDCG@10=0.93.  Further maximising retrieval quality is uninformative here.
Instead this sweep finds the KNEE -- the smallest doc-level k at which quality
is within MARGIN (default 0.02) of the k=10 ceiling.  A smaller k means fewer
chunks in the LLM prompt, lower token cost, and less mid-context dilution.

Also records the minimum cosine score of any retrieved chunk whose doc_id is
in the relevant set.  This is a SAFETY BOUND for RAG_MIN_CHUNK_SCORE: a floor
at or below this value is guaranteed not to exclude any known-relevant chunk
from the eval set.  It is NOT a tuned optimum -- recall@k cannot validate a
precision floor on a saturated corpus.

Note on chunk vs doc level
--------------------------
CHUNK_FETCH_K=20 retrieves 20 chunks; for this corpus (34 chunks / 23 docs,
avg 1.48 chunks/doc) that deduplicates to ~13 unique docs, enough to evaluate
doc-level k up to 10.  The production RAG_DEFAULT_TOP_K is also a chunk-level
budget; for this corpus the chunk:doc ratio is close to 1:1 at the k values
swept, so the curve directly informs the RAG_DEFAULT_TOP_K recommendation.

Usage:
    python eval/run_sweep.py [--namespace eval] [--golden eval/golden.jsonl]

Requirements:
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_HOST must be set
    (or in the root .env / backend/.env).  No Groq / LLM calls are made.

IMPORTANT -- live inference, not CI:
    Issues LIVE Pinecone query calls (read-only, no upserts): one call per
    golden query.  DO NOT add to CI.  DO NOT call from `make eval`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path setup -- must come before any backend or eval imports.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_BACKEND_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env", override=False)
    load_dotenv(_BACKEND_DIR / ".env", override=False)
except ImportError:
    pass

from eval.metrics import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402

SWEEP_K_VALUES: List[int] = [1, 2, 3, 5, 8, 10]
CHUNK_FETCH_K: int = 20  # chunk-level retrieval depth; dedupes to ~13 unique docs
MARGIN: float = 0.02     # knee detection: within MARGIN of k=10 on both recall + nDCG


# ---------------------------------------------------------------------------
# Helpers
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


def _hits_to_doc_rank_list(hits: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    """Return [(doc_id, cosine_score), ...] deduped by doc_id, preserving rank order.

    Each hit may represent a chunk; multiple chunks from the same doc appear as
    separate hits.  We keep only the first occurrence per doc_id (highest-scoring
    chunk) to produce a document-level ranked list.
    """
    seen: set = set()
    result: List[Tuple[str, float]] = []
    for hit in hits:
        fields: Dict[str, Any] = hit.get("fields") or {}
        doc_id: str = fields.get("doc_id") or ""
        if not doc_id:
            raw_id: str = hit.get("_id") or ""
            doc_id = raw_id.rsplit(":", 1)[0] if ":" in raw_id else raw_id
        score: float = float(hit.get("_score") or hit.get("score") or 0.0)
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            result.append((doc_id, score))
    return result


def _run_retrieval(
    namespace: str, query: str, top_k: int
) -> List[Tuple[str, float]]:
    """Single Pinecone dense-only search; returns doc-ranked list with scores."""
    from app.services.pinecone_store import search as pinecone_search  # noqa: PLC0415

    hits = pinecone_search(
        namespace=namespace,
        query_text=query,
        top_k=top_k,
        filters=None,
        fields=None,
    )
    return _hits_to_doc_rank_list(hits)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_sweep_report(report: Dict[str, Any], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = report["timestamp"].replace(":", "-").replace(".", "-")
    json_path = reports_dir / f"sweep_{ts}.json"
    md_path = reports_dir / f"sweep_{ts}.md"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    k_vals: List[int] = report["sweep_k_values"]
    agg: Dict[str, Any] = report["aggregate"]
    max_k = max(k_vals)
    ceiling_rec: float = agg[str(max_k)]["mean_recall_at_k"]
    ceiling_ndcg: float = agg[str(max_k)]["mean_ndcg_at_k"]
    knee: int = report["knee_k"]
    margin: float = report["margin"]
    floor_sb: float = report["floor_safety_bound"]

    lines = [
        f"# Top-k Sweep Eval - {report['timestamp']}",
        f"",
        f"Namespace: `{report['namespace']}` | chunk_fetch_k: {report['chunk_fetch_k']}"
        f" | n: {report['total_queries']} queries",
        f"Path: dense-only (rerank OFF) -- reflects shipped production defaults.",
        f"",
        f"## Quality vs k",
        f"",
        f"| k  | Recall@k | nDCG@k  | P@k    | D-Recall | D-nDCG  |",
        f"|----|----------|---------|--------|----------|---------|",
    ]
    for k in k_vals:
        m = agg[str(k)]
        d_rec = m["mean_recall_at_k"] - ceiling_rec
        d_ndcg = m["mean_ndcg_at_k"] - ceiling_ndcg
        tag = ""
        if k == knee and k != max_k:
            tag = " <- knee"
        elif k == max_k:
            tag = " <- ceiling"
        lines.append(
            f"| {k:<2} | {m['mean_recall_at_k']:.4f}   | {m['mean_ndcg_at_k']:.4f}  |"
            f" {m['mean_precision_at_k']:.4f} | {d_rec:+.4f}  | {d_ndcg:+.4f} |{tag}"
        )
    lines += [
        f"",
        f"Margin for knee: within {margin} of k={max_k} on both recall and nDCG.",
        f"Knee: k={knee} (quality within {margin} of k={max_k} ceiling"
        + (" -- same as ceiling, no lower k qualifies" if knee == max_k else "") + ").",
        f"",
        f"## Cosine Floor Safety Bound",
        f"",
        f"Min cosine score of a retrieved chunk belonging to a golden-relevant doc: "
        f"{floor_sb:.4f}",
        f"",
        f"> A RAG_MIN_CHUNK_SCORE at or below {floor_sb:.4f} is safe for this eval set:",
        f"> no golden-relevant chunk scored below this threshold at chunk_fetch_k={report['chunk_fetch_k']}.",
        f"> This is a SAFETY BOUND, not a tuned optimum.  Recall@k does not measure",
        f"> precision; on this saturated corpus the floor cannot be validated tightly.",
        f"> To tune it sharply: use a precision-oriented eval, graded relevance labels,",
        f"> or a larger / noisier corpus where low-quality chunks are retrievable.",
    ]

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Top-k sweep evaluation -- no LLM calls, read-only Pinecone queries."
    )
    parser.add_argument("--namespace", default="eval", help="Pinecone namespace")
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
        print("ERROR: no golden entries found.", file=sys.stderr)
        return 1

    from app.core.config import get_settings           # noqa: PLC0415
    from app.services.pinecone_store import init_pinecone  # noqa: PLC0415

    settings = get_settings()
    init_pinecone(settings)

    print(
        f"Loaded {len(entries)} golden entries | "
        f"namespace={args.namespace} chunk_fetch_k={CHUNK_FETCH_K} "
        f"k_values={SWEEP_K_VALUES}"
    )

    # Per-k accumulators
    per_k_recalls: Dict[int, List[float]] = {k: [] for k in SWEEP_K_VALUES}
    per_k_ndcgs:   Dict[int, List[float]] = {k: [] for k in SWEEP_K_VALUES}
    per_k_precs:   Dict[int, List[float]] = {k: [] for k in SWEEP_K_VALUES}

    # Floor safety bound tracking
    floor_min_scores: List[float] = []
    skipped = 0

    for entry in entries:
        query: str = entry["query"]
        relevant_ids: set = set(entry.get("relevant_doc_ids") or [])

        if not relevant_ids or any("PLACEHOLDER" in r for r in relevant_ids):
            print(f"  [SKIP] {query[:60]}")
            skipped += 1
            continue

        # Single Pinecone call for this query
        doc_rank = _run_retrieval(args.namespace, query, CHUNK_FETCH_K)
        ranked_ids = [doc_id for doc_id, _ in doc_rank]

        for k in SWEEP_K_VALUES:
            per_k_recalls[k].append(recall_at_k(ranked_ids, relevant_ids, k))
            per_k_ndcgs[k].append(ndcg_at_k(ranked_ids, relevant_ids, k))
            per_k_precs[k].append(precision_at_k(ranked_ids, relevant_ids, k))

        # Track minimum cosine score of any relevant-doc chunk in the results
        rel_scores = [score for doc_id, score in doc_rank if doc_id in relevant_ids]
        if rel_scores:
            floor_min_scores.append(min(rel_scores))

        # Progress line (k=5 and k=10 representative values)
        r5  = recall_at_k(ranked_ids, relevant_ids, 5)
        r10 = recall_at_k(ranked_ids, relevant_ids, 10)
        n5  = ndcg_at_k(ranked_ids, relevant_ids, 5)
        n10 = ndcg_at_k(ranked_ids, relevant_ids, 10)
        print(
            f"  [QUERY] {query[:60]}"
            f"\n         recall@5={r5:.3f} recall@10={r10:.3f}"
            f"  nDCG@5={n5:.3f} nDCG@10={n10:.3f}"
        )

    if not per_k_recalls[max(SWEEP_K_VALUES)]:
        print("ERROR: no valid entries evaluated.", file=sys.stderr)
        return 1

    n = len(per_k_recalls[max(SWEEP_K_VALUES)])

    # Aggregate mean for each k
    aggregate: Dict[str, Dict[str, float]] = {}
    for k in SWEEP_K_VALUES:
        aggregate[str(k)] = {
            "mean_recall_at_k":    round(sum(per_k_recalls[k]) / n, 6),
            "mean_ndcg_at_k":      round(sum(per_k_ndcgs[k])   / n, 6),
            "mean_precision_at_k": round(sum(per_k_precs[k])    / n, 6),
        }

    max_k = max(SWEEP_K_VALUES)
    ceiling_rec  = aggregate[str(max_k)]["mean_recall_at_k"]
    ceiling_ndcg = aggregate[str(max_k)]["mean_ndcg_at_k"]

    # Knee: smallest k where recall AND nDCG are within MARGIN of ceiling
    knee_k = max_k
    for k in SWEEP_K_VALUES:
        if (
            ceiling_rec  - aggregate[str(k)]["mean_recall_at_k"] <= MARGIN
            and ceiling_ndcg - aggregate[str(k)]["mean_ndcg_at_k"] <= MARGIN
        ):
            knee_k = k
            break

    floor_safety = round(min(floor_min_scores), 4) if floor_min_scores else 0.0

    # ---- Terminal output ----
    print(f"\n--- Top-k Sweep (n={n}, chunk_fetch_k={CHUNK_FETCH_K}) ---")
    print(f"  {'k':<3} {'Recall@k':>10} {'nDCG@k':>10} {'P@k':>10} {'D-Recall':>10} {'D-nDCG':>10}")
    print("  " + "-" * 58)
    for k in SWEEP_K_VALUES:
        m = aggregate[str(k)]
        d_rec  = m["mean_recall_at_k"]    - ceiling_rec
        d_ndcg = m["mean_ndcg_at_k"] - ceiling_ndcg
        tag = ""
        if k == knee_k and k != max_k:
            tag = "  <- knee"
        elif k == max_k:
            tag = "  <- ceiling"
        print(
            f"  {k:<3} {m['mean_recall_at_k']:>10.4f} {m['mean_ndcg_at_k']:>10.4f}"
            f" {m['mean_precision_at_k']:>10.4f} {d_rec:>+10.4f} {d_ndcg:>+10.4f}{tag}"
        )
    print(f"\n  Margin used: {MARGIN}  (within {MARGIN} of k={max_k} ceiling on both recall + nDCG)")
    if knee_k == max_k:
        print(f"  Knee: k={knee_k} (no smaller k qualifies within margin={MARGIN})")
    else:
        print(f"  Knee: k={knee_k} -- smallest k where quality is within {MARGIN} of ceiling")
    print(
        f"\n  Floor safety bound: {floor_safety:.4f}"
        f"  (min cosine score of a relevant-doc chunk in top-{CHUNK_FETCH_K} results)"
    )
    print(
        f"  A RAG_MIN_CHUNK_SCORE <= {floor_safety:.4f} will not drop known-relevant chunks."
        f"  (SAFETY BOUND only, not a tuned optimum)"
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: Dict[str, Any] = {
        "timestamp": ts,
        "namespace": args.namespace,
        "chunk_fetch_k": CHUNK_FETCH_K,
        "sweep_k_values": SWEEP_K_VALUES,
        "total_queries": n,
        "skipped_entries": skipped,
        "margin": MARGIN,
        "knee_k": knee_k,
        "floor_safety_bound": floor_safety,
        "aggregate": aggregate,
    }

    reports_dir = _REPO_ROOT / "eval" / "reports"
    json_path = _write_sweep_report(report, reports_dir)
    print(f"\nReport written to: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
