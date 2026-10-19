"""Size the regs-v1 upload, and upsert it only with --yes.

python -m corpus.upsert          dry run: counts, estimated embedding tokens and storage
python -m corpus.upsert --yes    write to Pinecone namespace regs-v1 (needs explicit approval)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
NAMESPACE = "regs-v1"
_META = ("ref", "title", "framework", "kind", "binding", "chapter", "section", "article", "paragraph", "date_basis")


def to_record(p: dict) -> dict:
    content = " ".join(x for x in (p["ref"], p.get("title"), p["text"], p.get("guidance")) if x)[:6000]
    rec = {"_id": p["id"], "content": content}
    rec.update({k: p[k] for k in _META if p.get(k)})
    if p.get("applies_from"):
        rec["applies_from"] = int(p["applies_from"].replace("-", ""))
    if p.get("path"):
        rec["path"] = " > ".join(p["path"])
    return rec


def estimate(provisions: list) -> dict:
    recs = [to_record(p) for p in provisions]
    chars = sum(len(r["content"]) for r in recs)
    meta = sum(len(json.dumps(r)) for r in recs)
    return {"records": len(recs), "content_chars": chars, "est_embed_tokens": chars // 4,
            "est_storage_mb": round((len(recs) * 1024 * 4 + meta) / 1e6, 1)}


def main() -> int:
    from app.services.regs import load_provisions

    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually upsert (requires approval)")
    args = ap.parse_args()
    provisions = load_provisions()
    print(json.dumps({"namespace": NAMESPACE, **estimate(provisions)}, indent=2))
    if not args.yes:
        print("Dry run only. Re-run with --yes after approval to upsert.")
        return 0
    from app.core.config import get_settings
    from app.services.pinecone_store import init_pinecone, upsert_records
    init_pinecone(get_settings())
    print("upserted:", upsert_records(NAMESPACE, [to_record(p) for p in provisions]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
