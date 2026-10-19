"""Provision-level retrieval eval over the regulatory corpus.

python eval/run_regs.py --mode lexical          (offline)
python eval/run_regs.py --mode dense|fused      (needs namespace regs-v1 in Pinecone)
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from eval.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402


def label_hits(ranked_ids: List[str], labels: List[str]) -> List[str]:
    out, seen = [], set()
    for rid in ranked_ids:
        key = next((l for l in labels if rid == l or rid.startswith(l + "/")), rid)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _rank(mode: str, query: str, k: int, lexical, dense_search) -> List[str]:
    lex = [i for i, _ in lexical.search(query, k)] if mode in ("lexical", "fused") else []
    den = dense_search(query, k) if mode in ("dense", "fused") else []
    if mode != "fused":
        return lex or den
    scores = {}
    for ranks in (lex, den):
        for r, pid in enumerate(ranks):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (60 + r + 1)
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])][:k]


def main() -> int:
    from app.services.regs import load_provisions
    from app.services.regs.lexical import BM25

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lexical", "dense", "fused"], default="lexical")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    lexical = BM25.from_provisions(load_provisions())
    dense_search = None
    if args.mode != "lexical":
        from app.core.config import get_settings
        from app.services.pinecone_store import init_pinecone, search
        init_pinecone(get_settings())
        dense_search = lambda q, k: [h.get("_id", "") for h in search(namespace="regs-v1", query_text=q, top_k=k)]
    rows = [json.loads(l) for l in (ROOT / "eval" / "regs_golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("_meta")]
    per = []
    for r in rows:
        ranked = label_hits(_rank(args.mode, r["query"], args.k, lexical, dense_search), r["relevant"])
        rel = set(r["relevant"])
        per.append({"query": r["query"], "type": r.get("type"), "recall": recall_at_k(ranked, rel, args.k),
                    "mrr": mrr(ranked, rel), "ndcg": ndcg_at_k(ranked, rel, args.k)})
    agg = {m: round(sum(p[m] for p in per) / len(per), 4) for m in ("recall", "mrr", "ndcg")}
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "eval" / "reports" / f"regs_{args.mode}_{ts}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"mode": args.mode, "k": args.k, "n": len(per), "aggregate": agg, "per_query": per}, indent=2), encoding="utf-8")
    print(json.dumps({"mode": args.mode, "n": len(per), **agg}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
