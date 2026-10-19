"""Build the provision corpus: python -m corpus.build [--offline]"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from corpus.fetch import RAW_DIR, fetch, raw_path
from corpus.parse_eu import parse_eu
from corpus.parse_nist import parse_oscal, parse_playbook
from corpus.sources import SOURCES

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "backend" / "app" / "data" / "regs"
sys.path.insert(0, str(ROOT / "backend"))


def split_long(rec: dict, max_chars: int = 6000) -> list:
    if len(rec["text"]) <= max_chars:
        return [rec]
    sentences, parts, cur = re.split(r"(?<=[.;:])\s+", rec["text"]), [], ""
    for s in sentences:
        if cur and len(cur) + 1 + len(s) > max_chars:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    parts.append(cur)
    return [{**rec, "id": f"{rec['id']}/part-{i}", "ref": f"{rec['ref']} (part {i})", "text": t}
            for i, t in enumerate(parts, 1)]


def parse_source(source, raw: bytes) -> list:
    if source.fmt == "eu-xhtml":
        return parse_eu(raw.decode("utf-8"), source)
    data = json.loads(raw)
    return parse_oscal(data["catalog"], source) if source.fmt == "oscal-catalog" else parse_playbook(data, source)


def build(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR, offline: bool = False) -> dict:
    from app.services.regs.registry import Registry  # backend package

    manifest, records = [], []
    for source in SOURCES:
        manifest.append(fetch(source, raw_dir) if not offline else {"key": source.key})
        records += parse_source(source, raw_path(source, raw_dir).read_bytes())
    records = [p for r in records if r["text"] for p in split_long(r)]
    ids = Counter(r["id"] for r in records)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate provision ids: {dupes[:10]}")
    registry = Registry.load(out_dir / "applicability.yaml", records)
    for r in records:
        r.update(registry.fields(r))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "regs-v1.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    if not offline:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stats = {"records": len(records), "chars": sum(len(r["text"]) for r in records),
             "by_framework": dict(Counter(r["framework"] for r in records)),
             "by_kind": dict(Counter(r["kind"] for r in records))}
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="reuse corpus/raw without fetching")
    build(offline=ap.parse_args().offline)
