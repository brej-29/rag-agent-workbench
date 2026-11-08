"""Crosswalks from published mappings (spec §13): only id-to-id links are kept, never mapping text.

python -m corpus.crosswalks fetch   download each manifest source into corpus/raw/crosswalks/<id>/ (gitignored)
python -m corpus.crosswalks build   official links -> backend/app/data/regs/crosswalks.jsonl (committed)
                                    community links -> backend/app/data/regs/crosswalks.community.jsonl (gitignored)
"""
import argparse
import hashlib
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET  # expat >= 2.4.1 rejects entity expansion; ElementTree loads no external entities
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
DATA = ROOT / "backend" / "app" / "data" / "regs"
RAW = Path(__file__).resolve().parent / "raw" / "crosswalks"
Pair = Tuple[str, str]
_M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def load_manifest() -> List[dict]:
    return yaml.safe_load((DATA / "crosswalk_sources.yaml").read_text(encoding="utf-8"))


def read_xlsx(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Sheet name -> rows (column letter -> cell text), stdlib only."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        shared = ["".join(t.text or "" for t in si.iter(_M + "t"))
                  for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(_M + "si")]
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    out = {}
    for sheet in ET.fromstring(z.read("xl/workbook.xml")).find(_M + "sheets"):
        target = rels[sheet.get(_R + "id")].lstrip("/")
        rows = []
        for row in ET.fromstring(z.read(target if target.startswith("xl/") else "xl/" + target)).iter(_M + "row"):
            cells = {}
            for c in row.findall(_M + "c"):
                v = c.find(_M + "v")
                if c.get("t") == "s" and v is not None:
                    text = shared[int(v.text)]
                elif c.get("t") == "inlineStr":
                    text = "".join(t.text or "" for t in c.iter(_M + "t"))
                else:
                    text = v.text if v is not None else ""
                cells[re.match(r"[A-Z]+", c.get("r")).group(0)] = (text or "").strip()
            rows.append(cells)
        out[sheet.get("name")] = rows
    return out


_CTRL = re.compile(r"\b([A-Z]{2})-0*(\d+)(?:\s*\(\s*0*(\d+)\s*\))?")
_CSF = re.compile(r"\b([A-Z]{2}\.[A-Z]{2}(?:-\d{2})?)(?![\w.-])")
_ANNEX_A = {5: 37, 6: 8, 7: 14, 8: 34}  # ISO/IEC 27001:2022 Annex A themes and their last control number
_RMF = re.compile(r"^(Govern|Map|Measure|Manage)\s+(\d+\.\d+)\b", re.IGNORECASE)
_ISO42001 = re.compile(r"^((?:[AB]\.)?\d{1,2}(?:\.\d{1,2}){0,3})\s+[A-Z]")


def nist_controls(cell: str) -> List[str]:
    return [f"nist-800-53/{f.lower()}-{n}" + (f".{e}" if e else "") for f, n, e in _CTRL.findall(cell or "")]


def csf_ids(cell: str) -> List[str]:
    return [f"nist-csf/{x}" for x in _CSF.findall(cell or "")]


def _iso27001(raw: str) -> str:
    raw = re.sub(r"^(?:mandatory clause|clause|annex a|control)\s*:?\s*", "", raw.strip(), flags=re.I)
    raw = raw.replace("А", "A")  # ENISA cells carry a Cyrillic A in some Annex A ids
    if re.fullmatch(r"\d{1,2}\.\d{5,}", raw):  # numeric cell: 4.0999999999999996 -> 4.1
        raw = f"{float(raw):.2f}".rstrip("0").rstrip(".")
    m = re.fullmatch(r"A\.?\s?(\d{1,2})\.(\d{1,2})", raw)
    if m:
        theme, n = int(m.group(1)), int(m.group(2))
        return f"A.{theme}.{n}" if 1 <= n <= _ANNEX_A.get(theme, 0) else ""
    m = re.fullmatch(r"(\d{1,2})(?:\.\d{1,2}){0,2}", raw)
    return raw if m and 4 <= int(m.group(1)) <= 10 else ""


def iso27001_ids(cell: str) -> List[str]:
    ids = [_iso27001(x) for x in re.split(r"[,;\n]", cell or "")]
    return [f"iso-27001/{x}" for x in dict.fromkeys(ids) if x]


def rmf_iso42001_pairs(lines: List[str]) -> List[Pair]:
    pairs, current = [], None
    for line in (l.strip() for l in lines):
        m = _RMF.match(line)
        if m:
            current = f"nist-ai-rmf/{m.group(1).lower()}-{m.group(2)}"
            continue
        m = _ISO42001.match(line)
        if current and m:
            pairs.append((current, f"iso-42001-fdis/{m.group(1)}"))
    return pairs


def nis2_ir_headings(provisions: List[dict]) -> List[Tuple[str, str, str]]:
    out = []
    for p in provisions:
        path = p.get("path") or []
        if p["id"].startswith("nis2-ir/anx-1/pt-") and len(path) > 1:
            m = re.search(r"Article\s*21\(2\),?\s*points?\s*(.*?)\s*of Directive", path[1])
            if m:
                letters = re.findall(r"\(([a-j])\)", m.group(1))
                out.append((p["id"], "nis2/art-21/par-2", "Art. 21(2) point" + ("s " if len(letters) > 1 else " ")
                            + ", ".join(f"({x})" for x in letters)))
    return out


def nist_csf_800_53(path: Path) -> List[Pair]:
    rows = read_xlsx(path)["Relationships"]
    return [(f"nist-csf/{r['A']}", c) for r in rows[1:] if _CSF.fullmatch(r.get("A", "")) for c in nist_controls(r.get("C", ""))]


def nist_800_53_iso(path: Path) -> List[Pair]:
    pairs = []
    for name, rows in read_xlsx(path).items():
        if name != "Definitions":
            for r in rows[1:]:
                for control in nist_controls(r.get("A", ""))[:1]:
                    pairs += [(control, i) for i in iso27001_ids(r.get("D", ""))]
    return pairs


def enisa_nis2(path: Path) -> List[Pair]:
    pairs = []
    for r in read_xlsx(path)["Mapping table"]:
        if re.fullmatch(r"\d{1,2}\.\d{1,2}", r.get("A", "")):
            point = f"nis2-ir/anx-1/pt-{r['A']}"
            pairs += [(point, i) for i in iso27001_ids(r.get("C", ""))] + [(point, c) for c in csf_ids(r.get("D", ""))]
    return pairs


def csf_to_iso(path: Path) -> List[Pair]:
    rows = read_xlsx(path)["Relationships"]
    return [(f"nist-csf/{r['A']}", i) for r in rows[1:] if _CSF.fullmatch(r.get("A", "")) for i in iso27001_ids(r.get("C", ""))]


def msft_airmf_iso42001(path: Path) -> List[Pair]:
    from pypdf import PdfReader  # dev dependency, build time only
    return rmf_iso42001_pairs([l for page in PdfReader(path).pages for l in (page.extract_text() or "").splitlines()])


PARSERS: Dict[str, Callable[[Path], List[Pair]]] = {
    "nist-csf-800-53": nist_csf_800_53, "nist-800-53-iso27001": nist_800_53_iso, "enisa-nis2": enisa_nis2,
    "razil-csf-iso27001": csf_to_iso, "pauloweit-iso27001-csf": csf_to_iso, "msft-airmf-iso42001": msft_airmf_iso42001,
}


def fetch(raw: Path = RAW) -> None:
    for src in load_manifest():
        if src["id"] not in PARSERS:
            continue
        path = raw / src["id"] / src["file"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            resp = httpx.get(src["url"], follow_redirects=True, timeout=120)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{src['id']:<24} sha256 {digest}" + ("" if src.get("sha256") in (None, digest) else "  CHANGED since manifest"))


def build(raw: Path = RAW, data: Path = DATA) -> dict:
    from app.services.regs import load_provisions

    provisions = load_provisions()
    ids = {p["id"] for p in provisions}
    official, community, report = [], [], {}
    for src in load_manifest():
        if src["id"] not in PARSERS:
            continue
        path = raw / src["id"] / src["file"]
        if not path.exists():
            report[src["id"]] = {"links": 0, "unresolved": [], "status": "not fetched"}
            continue
        kept, unresolved = [], set()
        for a, b in dict.fromkeys(PARSERS[src["id"]](path)):
            missing = [x for x in (a, b) if not x.startswith("iso-") and x not in ids]
            if missing:
                unresolved.update(missing)
                continue
            kept.append({"from": a, "to": b, "source": src["id"], "kind": src["kind"]})
        (official if src["kind"] == "official" else community).extend(kept)
        report[src["id"]] = {"links": len(kept), "unresolved": sorted(unresolved)}
    heads = [{"from": a, "to": b, "source": "nis2-ir-headings", "kind": "official", "note": n}
             for a, b, n in nis2_ir_headings(provisions)]
    official += heads
    report["nis2-ir-headings"] = {"links": len(heads), "unresolved": []}
    for name, links in (("crosswalks.jsonl", official), ("crosswalks.community.jsonl", community)):
        with open(data / name, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(l, sort_keys=True) + "\n" for l in links)
    (data / "crosswalk_report.json").write_text(json.dumps(
        {k: v for k, v in report.items() if not k.startswith(("razil", "pauloweit", "msft"))} | {
            k: {"links": v["links"], "status": v.get("status", "built locally")}
            for k, v in report.items() if k.startswith(("razil", "pauloweit", "msft"))}, indent=2), encoding="utf-8")
    print(json.dumps({k: (v["links"], len(v.get("unresolved", []))) for k, v in report.items()}))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fetch", "build"])
    args = ap.parse_args()
    fetch() if args.command == "fetch" else build()
