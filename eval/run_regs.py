"""Provision-level retrieval eval over the regulatory corpus.

python eval/run_regs.py --mode lexical          (offline)
python eval/run_regs.py --mode dense|fused      (needs namespace regs-v1 in Pinecone)
python eval/run_regs.py --mode diversified      (production ranking: 30 candidates, per-framework cap)
python eval/run_regs.py --mode abstain          (scope guard + floor on answerable and unanswerable sets; no LLM)
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


def floor_stats(answerable_best: List[float], unanswerable_top: List[float]) -> dict:
    """Where should the relevance floor sit? Compare the cosine of the best *relevant* hit for answerable
    questions with the top cosine for questions the corpus cannot answer (same method as the original 0.20)."""
    a, u = sorted(answerable_best), sorted(unanswerable_top)
    min_a, max_u = a[0], u[-1]
    overlap = sum(1 for x in a if x <= max_u) + sum(1 for y in u if y >= min_a)
    return {"min_answerable": min_a, "p10_answerable": a[int(0.1 * (len(a) - 1))],
            "max_unanswerable": max_u, "p90_unanswerable": u[int(0.9 * (len(u) - 1))],
            "overlap": overlap, "suggested_floor": None if overlap else round((min_a + max_u) / 2, 3)}


def summarize(per: List[dict]) -> dict:
    def mean(rows, m):
        return round(sum(r[m] for r in rows) / len(rows), 4)

    by_type = {}
    for t in sorted({p.get("type") for p in per}):
        rows = [p for p in per if p.get("type") == t]
        by_type[t] = {"n": len(rows), **{m: mean(rows, m) for m in ("recall", "mrr", "ndcg")}}
    return {**{m: mean(per, m) for m in ("recall", "mrr", "ndcg")}, "by_type": by_type}


def abstain_report(answerable: List[str], unanswerable: List[str], decide) -> dict:
    """decide(query) -> (abstain, reason).  Errors both ways are listed so each can be read."""
    false = [q for q in answerable if decide(q)[0]]
    missed = [q for q in unanswerable if not decide(q)[0]]
    return {"unanswerable": len(unanswerable), "correct_abstain": len(unanswerable) - len(missed),
            "answerable": len(answerable), "false_abstain": len(false),
            "missed_unanswerable": missed, "false_abstentions": false}


def _regs_decider(settings, search):
    """Production abstention without the LLM: scope guard, then the floor over diversified retrieval."""
    from datetime import date

    from app.services.regs.ask import retrieve_regs, scope_guard

    def decide(q: str):
        guard = scope_guard(q)
        if guard["abstain"]:
            return True, guard["kind"]
        sources, _ = retrieve_regs(q, None, date.today(), None, settings, search)
        kept = [s for s in sources if s["score"] >= settings.REGS_MIN_SCORE]
        return (False, None) if kept else (True, "no_match")

    return decide


def _rank(mode: str, query: str, k: int, lexical, dense_search, lexical_weight: float = 1.0) -> List[str]:
    lex = [i for i, _ in lexical.search(query, k)] if mode in ("lexical", "fused") else []
    den = dense_search(query, k) if mode in ("dense", "fused") else []
    if mode != "fused":
        return lex or den
    scores = {}
    for weight, ranks in ((lexical_weight, lex), (1.0, den)):
        for r, pid in enumerate(ranks):
            scores[pid] = scores.get(pid, 0.0) + weight / (60 + r + 1)
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])][:k]


def _calibrate(rows: List[dict]) -> int:
    from app.services.pinecone_store import search

    def hits(q: str, k: int):
        return [(h.get("_id", ""), float(h.get("_score") or 0.0)) for h in search(namespace="regs-v1", query_text=q, top_k=k)]

    answerable, missed = [], []
    for r in rows:
        scored = [s for pid, s in hits(r["query"], 10) if any(pid == l or pid.startswith(l + "/") for l in r["relevant"])]
        (answerable.append(max(scored)) if scored else missed.append(r["query"]))
    unanswerable_rows = [json.loads(l) for l in (ROOT / "eval" / "regs_unanswerable.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    unanswerable = []
    for r in (x for x in unanswerable_rows if not x.get("_meta")):
        top = hits(r["query"], 1)
        unanswerable.append(top[0][1] if top else 0.0)
        print(f"  unanswerable {top[0][1] if top else 0:.3f} {top[0][0] if top else '-':32s} {r['query'][:60]}")
    stats = floor_stats(answerable, unanswerable)
    print(json.dumps({**stats, "answerable_scored": len(answerable), "relevant_not_in_top10": len(missed)}, indent=2))
    return 0


def main() -> int:
    from app.services.regs import load_provisions
    from app.services.regs.lexical import BM25

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lexical", "dense", "fused", "diversified", "calibrate", "abstain"],
                    default="lexical")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lexical-weight", type=float, default=1.0, help="RRF weight of the lexical list in fused mode")
    ap.add_argument("--crosswalk", choices=["off", "boost", "inject"], default="off",
                    help="diversified mode: crosswalk-aware retrieval variant")
    args = ap.parse_args()
    lexical = BM25.from_provisions(load_provisions())
    dense_search = None
    if args.mode != "lexical":
        from app.core.config import get_settings
        from app.services.pinecone_store import init_pinecone, search
        settings = get_settings()
        init_pinecone(settings)
        dense_search = lambda q, k: [h.get("_id", "") for h in search(namespace="regs-v1", query_text=q, top_k=k)]
    if args.mode == "diversified":
        from datetime import date

        from app.services.regs.ask import retrieve_regs
        ranked_settings = settings.model_copy(update={"REGS_TOP_K": args.k, "REGS_CROSSWALK_MODE": args.crosswalk})
        dense_search = lambda q, k: [s["doc_id"] for s in retrieve_regs(q, None, date.today(), None, ranked_settings, search)[0]]
        args.mode, mode_name = "dense", "diversified" + ("" if args.crosswalk == "off" else f"-{args.crosswalk}")
    else:
        mode_name = args.mode
    rows = [json.loads(l) for l in (ROOT / "eval" / "regs_golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("_meta")]
    if args.mode == "calibrate":
        return _calibrate(rows)
    if args.mode == "abstain":
        un = [json.loads(l) for l in (ROOT / "eval" / "regs_unanswerable.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        rep = abstain_report([r["query"] for r in rows], [r["query"] for r in un if not r.get("_meta")],
                             _regs_decider(settings, search))
        out = ROOT / "eval" / "reports" / f"regs_abstain_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({**rep, "floor": settings.REGS_MIN_SCORE}, indent=2), encoding="utf-8")
        print(json.dumps({k: rep[k] for k in ("correct_abstain", "unanswerable", "false_abstain", "answerable")}))
        for q in rep["missed_unanswerable"] + rep["false_abstentions"]:
            print("  error:", q)
        return 0
    per = []
    for r in rows:
        ranked = label_hits(_rank(args.mode, r["query"], args.k, lexical, dense_search, args.lexical_weight), r["relevant"])
        rel = set(r["relevant"])
        per.append({"query": r["query"], "type": r.get("type"), "recall": recall_at_k(ranked, rel, args.k),
                    "mrr": mrr(ranked, rel), "ndcg": ndcg_at_k(ranked, rel, args.k)})
    agg = summarize(per)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "eval" / "reports" / f"regs_{mode_name}_{ts}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"mode": mode_name, "lexical_weight": args.lexical_weight, "k": args.k, "n": len(per), "aggregate": agg, "per_query": per}, indent=2), encoding="utf-8")
    print(json.dumps({"mode": mode_name, "n": len(per), **agg}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
