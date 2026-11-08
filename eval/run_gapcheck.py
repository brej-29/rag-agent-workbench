"""Golden gap checks (spec §15.5): python eval/run_gapcheck.py  (live model; ~12k tokens)."""
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]


def score(rows: list) -> dict:
    labelled = [(r["name"], pid, want, r["verdicts"].get(pid, "absent")) for r in rows for pid, want in r["expected"].items()]
    exact = sum(1 for _, _, want, got in labelled if want == got)
    false_covered = [f"{r['name']}:{pid}" for r in rows for pid in r["planted_missing"] if r["verdicts"].get(pid) == "covered"]
    planted = [(r, pid) for r in rows for pid in r["planted_missing"]]
    return {"labelled": len(labelled), "exact": exact, "accuracy": round(exact / len(labelled), 3),
            "verified_rate": round(sum(1 for *_, got in labelled if got not in ("unverified", "absent")) / len(labelled), 3),
            "missing_caught": sum(1 for r, pid in planted if r["verdicts"].get(pid) == "missing"),
            "missing_planted": len(planted), "false_covered": false_covered, "gate_passed": not false_covered,
            "disagreements": [f"{n}:{pid} expected {want}, got {got}" for n, pid, want, got in labelled if want != got]}


def main() -> int:
    import hashlib

    from app.services.llm.groq_llm import get_llm
    from app.services.regs.gapcheck import _FORMAT, _SYSTEM, check_unit, plan

    # Runs are compared only within one prompt version (publish_results groups them by this).
    prompt_sha = hashlib.sha256((_SYSTEM + json.dumps(_FORMAT, sort_keys=True)).encode()).hexdigest()[:12]

    llm, rows = get_llm(), []
    for row in (json.loads(l) for l in (ROOT / "eval/gap_golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()):
        if row.get("_meta"):
            continue
        p = plan((ROOT / row["policy_file"]).read_text(encoding="utf-8"), row["selections"], {}, date.today())
        results = [check_unit(u, llm) for u in p["units"]]
        row["verdicts"] = {d["provision"]: d["verdict"] for r in results for d in r["duties"]}
        row["tokens"] = sum(r["tokens"] for r in results)
        rows.append(row)
    rep = score(rows) | {"tokens": sum(r["tokens"] for r in rows), "prompt_sha": prompt_sha}
    out = ROOT / "eval/reports" / f"gapcheck_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep | {"rows": [{k: r[k] for k in ("name", "verdicts", "tokens")} for r in rows]}, indent=2),
                   encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("accuracy", "verified_rate", "missing_caught", "missing_planted",
                                          "false_covered", "gate_passed", "tokens")}))
    for d in rep["disagreements"]:
        print("  ", d)
    return 0 if rep["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
