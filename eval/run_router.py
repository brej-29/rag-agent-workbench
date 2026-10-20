"""Router intent accuracy over eval/regs_router.jsonl (uses the configured Groq model; ~30 calls).

python eval/run_router.py
"""
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]


def score(rows, predict):
    """predict(row) -> final intent (router plus the deterministic previous-intent rule)."""
    pairs = [(r["intent"], predict(r)) for r in rows]
    misses = [{"query": r["query"], "label": g, "predicted": p} for r, (g, p) in zip(rows, pairs) if g != p]
    return {"n": len(pairs), "accuracy": round(sum(g == p for g, p in pairs) / len(pairs), 3),
            "confusion": {f"{g}->{p}": c for (g, p), c in sorted(Counter(pairs).items())}, "misses": misses}


def main() -> int:
    from app.services.llm.groq_llm import get_llm
    from app.services.regs.router import apply_previous_intent, understand_request

    rows = [json.loads(l) for l in open(ROOT / "eval/regs_router.jsonl", encoding="utf-8") if l.strip()]
    rows = [r for r in rows if not r.get("_meta")]
    llm = get_llm()
    def predict(row):
        decision = understand_request(llm, row["query"], {}, row.get("previous_intent"))[0]
        return apply_previous_intent(decision["intent"], decision["profile_updates"], row.get("previous_intent"),
                                     row["query"])

    report = score(rows, predict)
    out = ROOT / "eval/reports" / f"router_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"router accuracy {report['accuracy']} on {report['n']}; misses {len(report['misses'])} -> {out.name}")
    for m in report["misses"]:
        print(f"  {m['label']} -> {m['predicted']}: {m['query']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
