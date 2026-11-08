# Corpus foundation (sub-project 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn eight official regulatory and standards texts into a dated, citable provision corpus with a lexical index, a provision-level retrieval eval, and a sized (not yet executed) Pinecone upload.

**Architecture:** Offline build tooling in `corpus/` fetches official sources (EU CELLAR XHTML, NIST OSCAL/Playbook JSON), parses them into provision records, and writes `backend/app/data/regs/regs-v1.jsonl`. Runtime code in `backend/app/services/regs/` (registry resolver, BM25) reads that file. Every applicability date in the registry cites a provision and a verbatim quote, which a test checks against the built corpus.

**Tech Stack:** Python 3.11, httpx, BeautifulSoup (`html.parser`), PyYAML, pytest. No new dependencies.

## Global Constraints

- Sources only from official endpoints listed in Task 1; a source that cannot be fetched is left out, never scraped from a third party.
- EU texts via `http://publications.europa.eu/resource/celex/<CELEX>` with `Accept: application/xhtml+xml`, `Accept-Language: eng` (EUR-Lex HTML returns 202/empty to scripts).
- AI Act text = official consolidated version CELEX `02024R1689-20260727`; its recitals from the original `32024R1689` (consolidations omit recitals).
- Licences: EU `© European Union, https://eur-lex.europa.eu. Reuse authorised under Commission Decision 2011/833/EU`; NIST `U.S. Government work; not subject to copyright in the United States (17 U.S.C. 105)`.
- No ISO or SOC 2 text anywhere in the public corpus.
- Every registry date carries `basis` (provision id) and `quote`; a test asserts the quote occurs in that provision's text.
- No Pinecone write in this plan: Task 8 stops at a dry-run estimate; the upsert runs only after explicit user approval.
- Hermetic tests (no network); fixtures are small hand-written snippets of the real markup.
- Existing suites must stay green (`pytest tests/`), invariants untouched.

---

## File structure

| File | Responsibility |
|---|---|
| `corpus/__init__.py` | package marker |
| `corpus/sources.py` | the v1 source list (`Source` dataclass, `SOURCES`) |
| `corpus/fetch.py` | download + cache + manifest entry per source |
| `corpus/parse_eu.py` | EU XHTML (OJ and consolidated markup) → provision dicts |
| `corpus/parse_nist.py` | OSCAL catalogs and AI RMF Playbook → provision dicts |
| `corpus/build.py` | CLI: fetch → parse → split long records → dates → write JSONL, manifest, NOTICE, stats |
| `corpus/upsert.py` | dry-run size/token estimate; `--yes` upserts to namespace `regs-v1` |
| `corpus/raw/` | cached downloads (gitignored) |
| `backend/app/data/regs/regs-v1.jsonl` | built corpus (committed) |
| `backend/app/data/regs/applicability.yaml` | dated rules with basis + quote (committed, hand-reviewed) |
| `backend/app/data/regs/manifest.json`, `NOTICE.md` | provenance and attribution |
| `backend/app/services/regs/__init__.py` | `load_provisions()` |
| `backend/app/services/regs/registry.py` | `Registry`, `Dated`, `applies()` |
| `backend/app/services/regs/lexical.py` | `BM25` index |
| `eval/regs_golden.jsonl`, `eval/run_regs.py` | provision-level retrieval eval |
| `tests/corpus/test_*.py` | unit tests per module |

## Provision record (all tasks)

```
id          eu-ai-act/art-26/par-1 | eu-ai-act/rct-12 | eu-ai-act/anx-III/pt-1 | nist-800-53/ac-2.1 | nist-csf/GV.OC-01 | nist-ai-rmf/govern-1.1
ref         "AI Act Art. 26(1)" | "AI Act Recital 12" | "AI Act Annex III, point 1" | "SP 800-53 AC-2(1)" | "CSF GV.OC-01" | "AI RMF GOVERN 1.1"
framework, instrument, binding (regulation|directive|voluntary_framework), licence, source_url
kind        article | article_paragraph | recital | annex | annex_point | control | control_enhancement | category | subcategory
title, text, path (list[str]), chapter, section, article, paragraph (str or null)
amended_by  list[str]
guidance    str or null (AI RMF "about" text)
applies_from, applies_note, date_basis   (set by build from the registry)
```

---

### Task 1: Sources and fetcher

**Files:** Create `corpus/__init__.py`, `corpus/sources.py`, `corpus/fetch.py`, `tests/corpus/__init__.py`, `tests/corpus/test_fetch.py`; modify `.gitignore` (add `corpus/raw/`).

**Interfaces:** Produces `Source(key, prefix, short, framework, instrument, url, fmt, licence, binding, accept, role)`, `SOURCES: list[Source]`, `fetch(source, raw_dir, client=None) -> dict` (manifest entry `{key,url,file,sha256,bytes,retrieved,licence}`), `raw_path(source, raw_dir) -> Path`.

- [ ] Step 1: failing test

```python
# tests/corpus/test_fetch.py
import hashlib
import httpx
import pytest
from corpus.fetch import fetch
from corpus.sources import SOURCES, Source

SRC = Source("t", "t", "T", "Test", "Test act", "https://x.invalid/doc", "eu-xhtml", "lic", "regulation")

def _client(body: bytes, seen: list):
    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=body)
    return httpx.Client(transport=httpx.MockTransport(handler))

def test_fetch_sends_negotiation_headers_and_caches(tmp_path):
    seen = []
    entry = fetch(SRC, tmp_path, _client(b"<html/>", seen))
    assert seen[0].headers["accept"] == "application/xhtml+xml" and seen[0].headers["accept-language"] == "eng"
    assert entry["sha256"] == hashlib.sha256(b"<html/>").hexdigest() and entry["bytes"] == 7
    fetch(SRC, tmp_path, _client(b"other", seen))
    assert len(seen) == 1  # cached

def test_empty_body_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="empty"):
        fetch(SRC, tmp_path, _client(b"", []))

def test_sources_are_official_and_licensed():
    for s in SOURCES:
        assert s.url.startswith(("http://publications.europa.eu/resource/celex/",
                                 "https://raw.githubusercontent.com/usnistgov/oscal-content/",
                                 "https://airc.nist.gov/docs/"))
        assert s.licence
    assert {s.prefix for s in SOURCES} == {"eu-ai-act", "gdpr", "nis2", "dora", "cra", "nist-800-53", "nist-csf", "nist-ai-rmf"}
```

- [ ] Step 2: run `pytest tests/corpus/test_fetch.py -q` → FAIL (ModuleNotFoundError: corpus).
- [ ] Step 3: implement

```python
# corpus/sources.py
"""v1 corpus sources: official machine-readable endpoints only."""
from dataclasses import dataclass

CELLAR = "http://publications.europa.eu/resource/celex/{}"
OSCAL = "https://raw.githubusercontent.com/usnistgov/oscal-content/main/nist.gov/{}"
EU_LICENCE = "© European Union, https://eur-lex.europa.eu. Reuse authorised under Commission Decision 2011/833/EU"
NIST_LICENCE = "U.S. Government work; not subject to copyright in the United States (17 U.S.C. 105)"


@dataclass(frozen=True)
class Source:
    key: str          # raw file stem
    prefix: str       # provision id prefix
    short: str        # citation prefix, e.g. "AI Act"
    framework: str
    instrument: str
    url: str
    fmt: str          # eu-xhtml | oscal-catalog | airmf-playbook
    licence: str
    binding: str      # regulation | directive | voluntary_framework
    accept: str = "application/xhtml+xml"
    role: str = "text"  # text | recitals


def _eu(key, prefix, short, framework, instrument, celex, binding="regulation", role="text"):
    return Source(key, prefix, short, framework, instrument, CELLAR.format(celex), "eu-xhtml", EU_LICENCE, binding, role=role)


def _nist(key, short, framework, instrument, url, fmt):
    return Source(key, key, short, framework, instrument, url, fmt, NIST_LICENCE, "voluntary_framework", "application/json")


SOURCES = [
    _eu("eu-ai-act", "eu-ai-act", "AI Act", "EU AI Act", "Regulation (EU) 2024/1689 (consolidated 27 July 2026)", "02024R1689-20260727"),
    _eu("eu-ai-act-recitals", "eu-ai-act", "AI Act", "EU AI Act", "Regulation (EU) 2024/1689", "32024R1689", role="recitals"),
    _eu("gdpr", "gdpr", "GDPR", "GDPR", "Regulation (EU) 2016/679", "32016R0679"),
    _eu("nis2", "nis2", "NIS2", "NIS2", "Directive (EU) 2022/2555", "32022L2555", binding="directive"),
    _eu("dora", "dora", "DORA", "DORA", "Regulation (EU) 2022/2554", "32022R2554"),
    _eu("cra", "cra", "CRA", "Cyber Resilience Act", "Regulation (EU) 2024/2847", "32024R2847"),
    _nist("nist-800-53", "SP 800-53", "NIST SP 800-53", "NIST SP 800-53 Rev. 5",
          OSCAL.format("SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json"), "oscal-catalog"),
    _nist("nist-csf", "CSF", "NIST CSF 2.0", "NIST Cybersecurity Framework 2.0",
          OSCAL.format("CSF/v2.0/json/NIST_CSF_v2.0_catalog.json"), "oscal-catalog"),
    _nist("nist-ai-rmf", "AI RMF", "NIST AI RMF", "NIST AI 100-1 (AI RMF 1.0), Playbook",
          "https://airc.nist.gov/docs/playbook.json", "airmf-playbook"),
]
```

```python
# corpus/fetch.py
"""Download each source once into corpus/raw/ and describe it for the manifest."""
import datetime
import hashlib
from pathlib import Path
from typing import Optional

import httpx

from corpus.sources import Source

RAW_DIR = Path(__file__).resolve().parent / "raw"


def raw_path(source: Source, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{source.key}.{'json' if source.accept == 'application/json' else 'xhtml'}"


def fetch(source: Source, raw_dir: Path = RAW_DIR, client: Optional[httpx.Client] = None) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_path(source, raw_dir)
    if not path.exists():
        client = client or httpx.Client(timeout=120, follow_redirects=True)
        resp = client.get(source.url, headers={"Accept": source.accept, "Accept-Language": "eng",
                                               "User-Agent": "rag-agent-workbench corpus builder"})
        resp.raise_for_status()
        if not resp.content:
            raise RuntimeError(f"{source.key}: empty response from {source.url}")
        path.write_bytes(resp.content)
    data = path.read_bytes()
    return {"key": source.key, "url": source.url, "file": path.name, "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "retrieved": datetime.date.fromtimestamp(path.stat().st_mtime).isoformat(),
            "licence": source.licence}
```

- [ ] Step 4: run the test → PASS. Step 5: commit `feat(corpus): official source list and cached fetcher`.

### Task 2: EU parser

**Files:** Create `corpus/parse_eu.py`, `tests/corpus/test_parse_eu.py`.

**Interfaces:** Consumes `Source`. Produces `parse_eu(html: str, source: Source) -> list[dict]` (provision dicts without date fields).

- [ ] Step 1: failing test (fixtures are trimmed copies of the real markup)

```python
# tests/corpus/test_parse_eu.py
from corpus.parse_eu import parse_eu
from corpus.sources import SOURCES

AI = next(s for s in SOURCES if s.key == "eu-ai-act")
RECITALS = next(s for s in SOURCES if s.key == "eu-ai-act-recitals")
GDPR = next(s for s in SOURCES if s.key == "gdpr")

OJ = """<div class="eli-subdivision" id="cpt_III"><div class="eli-subdivision" id="cpt_III.sct_3">
<div class="eli-subdivision" id="art_26"><p class="oj-ti-art">Article 26</p>
<div class="eli-title" id="art_26.tit_1"><p class="oj-sti-art">Obligations of deployers</p></div>
<div id="026.001"><p class="oj-normal">1. Deployers shall take measures.</p></div>
<div id="026.002"><p class="oj-normal">2. Deployers shall assign oversight:</p>
<table><tr><td><p class="oj-normal">(a)</p></td><td><p class="oj-normal">to competent persons;</p></td></tr></table></div>
</div></div></div>
<div class="eli-subdivision" id="art_99"><p class="oj-ti-art">Article 99</p>
<div class="eli-title" id="art_99.tit_1"><p class="oj-sti-art">Entry into force</p></div>
<p class="oj-normal">It shall apply from 25 May 2018.</p></div>
<div class="eli-subdivision" id="rct_12"><table><tr><td><p class="oj-normal">(12)</p></td>
<td><p class="oj-normal">The notion of AI system should be clearly defined.</p></td></tr></table></div>
<div class="eli-container" id="anx_III"><p class="oj-doc-ti">ANNEX III</p><p class="oj-doc-ti">High-risk AI systems</p>
<p class="oj-normal">The AI systems listed in any of the following areas:</p>
<table><tr><td><p class="oj-normal">1.</p></td><td><p class="oj-normal">Biometrics:</p></td></tr></table>
<table><tr><td><p class="oj-normal">2.</p></td><td><p class="oj-normal">Critical infrastructure.</p></td></tr></table></div>"""

CONSOLIDATED = """<p class="hd-modifiers">►M1 Regulation (EU) 2026/1744 of the European Parliament</p>
<div class="eli-subdivision" id="art_5"><p class="title-article-norm">Article 5</p>
<div class="eli-title" id="art_5.tit_1"><p class="stitle-article-norm">Prohibited AI practices</p></div>
<div class="norm"><span class="no-parag">1. </span><div class="norm inline-element">The following practices are prohibited.</div></div>
<p class="arrow">▼M1</p>
<div class="norm"><span class="no-parag">1a. </span><div class="norm inline-element">New prohibition.</div></div></div>"""


def _by_id(records):
    return {r["id"]: r for r in records}


def test_oj_articles_split_into_paragraphs_with_points_and_path():
    recs = _by_id(parse_eu(OJ, AI))
    p2 = recs["eu-ai-act/art-26/par-2"]
    assert p2["ref"] == "AI Act Art. 26(2)" and p2["kind"] == "article_paragraph"
    assert p2["text"] == "2. Deployers shall assign oversight: (a) to competent persons;"
    assert p2["path"] == ["Chapter III", "Section 3", "Article 26"]
    assert (p2["chapter"], p2["section"], p2["article"], p2["paragraph"]) == ("III", "3", "26", "2")
    assert p2["title"] == "Obligations of deployers" and p2["licence"] == AI.licence


def test_article_without_numbered_paragraphs_is_one_record():
    rec = _by_id(parse_eu(OJ, GDPR))["gdpr/art-99"]
    assert rec["kind"] == "article" and rec["text"] == "It shall apply from 25 May 2018." and rec["paragraph"] is None


def test_recitals_only_for_recital_sources_and_text_sources():
    recs = _by_id(parse_eu(OJ, RECITALS))
    assert set(recs) == {"eu-ai-act/rct-12"}
    assert recs["eu-ai-act/rct-12"]["text"] == "The notion of AI system should be clearly defined."
    assert "gdpr/rct-12" in _by_id(parse_eu(OJ, GDPR))


def test_annex_split_into_points():
    recs = _by_id(parse_eu(OJ, AI))
    assert recs["eu-ai-act/anx-III/pt-1"]["ref"] == "AI Act Annex III, point 1"
    assert recs["eu-ai-act/anx-III/pt-2"]["text"].endswith("Critical infrastructure.")
    assert recs["eu-ai-act/anx-III"]["text"].startswith("ANNEX III High-risk AI systems")


def test_consolidated_markup_strips_markers_and_records_amendments():
    recs = _by_id(parse_eu(CONSOLIDATED, AI))
    assert recs["eu-ai-act/art-5/par-1"]["amended_by"] == []
    new = recs["eu-ai-act/art-5/par-1a"]
    assert new["text"] == "1a. New prohibition." and new["amended_by"] == ["Regulation (EU) 2026/1744"]
```

- [ ] Step 2: run → FAIL. Step 3: implement

```python
# corpus/parse_eu.py
"""EU Official Journal XHTML (original and consolidated markup) → provision records."""
import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, Tag

from corpus.sources import Source

_MARKER = re.compile(r"[▼►]\s*[A-Z]\d*|◄")
_MODIFIER = re.compile(r"►\s*(M\d+)\s+((?:Regulation|Directive|Decision) \((?:EU|EC|EEC)\) (?:No )?\d+/\d+)")
_OJ_PARA = re.compile(r"^\d{3}\.(\d{3}[a-z]?)$")
_TITLE_CLASSES = {"oj-ti-art", "oj-sti-art", "title-article-norm", "stitle-article-norm"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _MARKER.sub(" ", text)).strip()


def _text(el: Tag) -> str:
    return _clean(el.get_text(" ", strip=True))


def _modifiers(soup: BeautifulSoup) -> Dict[str, str]:
    return {code: act for code, act in _MODIFIER.findall(soup.get_text(" "))}


def _codes(raw: str) -> List[str]:
    return re.findall(r"[▼►]\s*(M\d+)", raw)


def _location(el: Tag) -> Dict[str, Optional[str]]:
    chapter = section = None
    for parent in el.parents:
        pid = parent.get("id", "") if isinstance(parent, Tag) else ""
        m = re.fullmatch(r"cpt_([IVXLC]+)(?:\.sct_(\d+))?", pid)
        if m:
            chapter = chapter or m.group(1)
            section = section or m.group(2)
    return {"chapter": chapter, "section": section}


def _base(src: Source) -> dict:
    return {"framework": src.framework, "instrument": src.instrument, "binding": src.binding,
            "licence": src.licence, "source_url": src.url, "amended_by": [], "guidance": None}


def _paragraphs(art: Tag):
    """Yield (label, element, preceding_marker_text) for numbered paragraphs, both markups."""
    prev = ""
    for child in art.find_all(recursive=False):
        cls = set(child.get("class") or [])
        m = _OJ_PARA.match(child.get("id", ""))
        if m:
            yield m.group(1).lstrip("0") or "0", child, prev
        elif "norm" in cls and child.find("span", class_="no-parag", recursive=False):
            label = child.find("span", class_="no-parag", recursive=False).get_text(strip=True).rstrip(".")
            yield label, child, prev
        prev = child.get_text(" ", strip=True) if len(child.get_text(strip=True)) < 12 else ""


def _article(art: Tag, src: Source, mods: Dict[str, str]) -> List[dict]:
    num = art["id"].split("_", 1)[1]
    title_el = art.find(class_=["oj-sti-art", "stitle-article-norm"])
    loc = _location(art)
    path = [p for p in (f"Chapter {loc['chapter']}" if loc["chapter"] else None,
                        f"Section {loc['section']}" if loc["section"] else None) if p] + [f"Article {num}"]
    common = {**_base(src), **loc, "article": num, "title": _text(title_el) if title_el else "", "path": path}
    out = []
    for label, el, prev in _paragraphs(art):
        raw = el.get_text(" ") + " " + prev
        out.append({**common, "id": f"{src.prefix}/art-{num}/par-{label}", "ref": f"{src.short} Art. {num}({label})",
                    "kind": "article_paragraph", "paragraph": label, "text": _text(el),
                    "amended_by": sorted({mods[c] for c in _codes(raw) if c in mods})})
    if not out:
        body = [c for c in art.find_all(recursive=False) if not (set(c.get("class") or []) & _TITLE_CLASSES)
                and not c.get("id", "").endswith(".tit_1")]
        raw = " ".join(c.get_text(" ") for c in body)
        out.append({**common, "id": f"{src.prefix}/art-{num}", "ref": f"{src.short} Art. {num}", "kind": "article",
                    "paragraph": None, "text": _clean(raw),
                    "amended_by": sorted({mods[c] for c in _codes(raw) if c in mods})})
    return out


def _annex(anx: Tag, src: Source) -> List[dict]:
    num = anx["id"].split("_", 1)[1]
    common = {**_base(src), "chapter": None, "section": None, "article": None, "paragraph": None,
              "path": [f"Annex {num}"]}
    heads = [_text(p) for p in anx.find_all("p", class_="oj-doc-ti")]
    title = " ".join(h for h in heads if not h.upper().startswith("ANNEX"))
    out = [{**common, "id": f"{src.prefix}/anx-{num}", "ref": f"{src.short} Annex {num}", "kind": "annex",
            "title": title, "text": _text(anx)}]
    for table in anx.find_all("table", recursive=False):
        first = table.find("td")
        label = first.get_text(strip=True).rstrip(".") if first else ""
        if re.fullmatch(r"\d+", label):
            out.append({**common, "id": f"{src.prefix}/anx-{num}/pt-{label}",
                        "ref": f"{src.short} Annex {num}, point {label}", "kind": "annex_point",
                        "title": title, "text": _text(table)})
    if len(out) > 1:
        out[0]["text"] = " ".join(heads + [_text(p) for p in anx.find_all("p", class_="oj-normal", recursive=False)])
    return out


def _recital(rct: Tag, src: Source) -> dict:
    num = rct["id"].split("_", 1)[1]
    cells = rct.find_all("td")
    text = _text(cells[-1]) if len(cells) > 1 else re.sub(r"^\(\d+\)\s*", "", _text(rct))
    return {**_base(src), "id": f"{src.prefix}/rct-{num}", "ref": f"{src.short} Recital {num}", "kind": "recital",
            "title": "", "text": text, "path": [f"Recital {num}"], "chapter": None, "section": None,
            "article": None, "paragraph": None}


def parse_eu(html: str, source: Source) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    mods = _modifiers(soup)
    out: List[dict] = []
    if source.role == "text":
        for art in soup.find_all("div", id=re.compile(r"^art_\d+[a-z]?$")):
            out += _article(art, source, mods)
        for anx in soup.find_all("div", id=re.compile(r"^anx_[IVXLC]+[a-z]?$")):
            out += _annex(anx, source)
    for rct in soup.find_all("div", id=re.compile(r"^rct_\d+$")):
        out.append(_recital(rct, source))
    return out
```

- [ ] Step 4: run → PASS. Step 5: commit `feat(corpus): EU XHTML parser (OJ + consolidated markup)`.

### Task 3: NIST parsers

**Files:** Create `corpus/parse_nist.py`, `tests/corpus/test_parse_nist.py`.

**Interfaces:** Produces `parse_oscal(catalog: dict, source) -> list[dict]`, `parse_playbook(entries: list, source) -> list[dict]`.

- [ ] Step 1: failing test

```python
# tests/corpus/test_parse_nist.py
from corpus.parse_nist import parse_oscal, parse_playbook
from corpus.sources import SOURCES

SP = next(s for s in SOURCES if s.key == "nist-800-53")
CSF = next(s for s in SOURCES if s.key == "nist-csf")
RMF = next(s for s in SOURCES if s.key == "nist-ai-rmf")

CAT_800 = {"groups": [{"id": "ac", "title": "Access Control", "controls": [
    {"id": "ac-2", "class": "SP800-53", "title": "Account Management",
     "params": [{"id": "ac-2_prm_1", "label": "organization-defined account types"}],
     "props": [{"name": "label", "value": "AC-2"}],
     "parts": [{"id": "ac-2_smt", "name": "statement", "parts": [
         {"id": "ac-2_smt.a", "name": "item", "props": [{"name": "label", "value": "a."}],
          "prose": "Define {{ insert: param, ac-2_prm_1 }} allowed for use;"}]},
               {"id": "ac-2_gdn", "name": "guidance", "prose": "Guidance text."}],
     "controls": [{"id": "ac-2.1", "class": "SP800-53-enhancement", "title": "Automated Management",
                   "props": [{"name": "label", "value": "AC-2(1)"}],
                   "parts": [{"id": "ac-2.1_smt", "name": "statement", "prose": "Support management."}]},
                  {"id": "ac-2.10", "class": "SP800-53-enhancement", "title": "Withdrawn one",
                   "props": [{"name": "label", "value": "AC-2(10)"}, {"name": "status", "value": "withdrawn"}]}]}]}]}

CAT_CSF = {"groups": [{"id": "GV", "title": "GOVERN", "controls": [
    {"id": "GV.OC", "class": "category", "title": "Organizational Context",
     "parts": [{"id": "GV.OC_statement", "name": "statement", "prose": "Context is understood"}],
     "controls": [{"id": "GV.OC-01", "class": "subcategory", "title": "GV.OC-01",
                   "parts": [{"id": "GV.OC-01_statement", "name": "statement", "prose": "Mission is understood"}]}]}]}]}


def test_800_53_controls_enhancements_params_and_withdrawn():
    recs = {r["id"]: r for r in parse_oscal(CAT_800, SP)}
    assert set(recs) == {"nist-800-53/ac-2", "nist-800-53/ac-2.1"}
    ac2 = recs["nist-800-53/ac-2"]
    assert ac2["ref"] == "SP 800-53 AC-2" and ac2["kind"] == "control"
    assert ac2["text"] == "a. Define [organization-defined account types] allowed for use;"
    assert ac2["guidance"] == "Guidance text." and ac2["path"] == ["Access Control", "AC-2 Account Management"]
    assert recs["nist-800-53/ac-2.1"]["kind"] == "control_enhancement"


def test_csf_categories_and_subcategories():
    recs = {r["id"]: r for r in parse_oscal(CAT_CSF, CSF)}
    assert recs["nist-csf/GV.OC"]["kind"] == "category"
    sub = recs["nist-csf/GV.OC-01"]
    assert sub["kind"] == "subcategory" and sub["ref"] == "CSF GV.OC-01" and sub["text"] == "Mission is understood"
    assert sub["path"] == ["GOVERN", "GV.OC Organizational Context", "GV.OC-01"]


def test_playbook_entries():
    rec = parse_playbook([{"type": "Govern", "title": "GOVERN 1.1", "category": "GOVERN-1",
                           "description": "Legal requirements are understood.", "section_about": "About text."}], RMF)[0]
    assert rec["id"] == "nist-ai-rmf/govern-1.1" and rec["ref"] == "AI RMF GOVERN 1.1"
    assert rec["text"] == "Legal requirements are understood." and rec["guidance"] == "About text."
    assert rec["kind"] == "subcategory" and rec["path"] == ["Govern", "GOVERN-1", "GOVERN 1.1"]
```

- [ ] Step 2: run → FAIL. Step 3: implement

```python
# corpus/parse_nist.py
"""NIST OSCAL catalogs (SP 800-53, CSF 2.0) and the AI RMF Playbook → provision records."""
import re
from typing import Dict, List

from corpus.sources import Source

_PARAM = re.compile(r"\{\{\s*insert:\s*param,\s*([\w.\-]+)\s*\}\}")
_KIND = {"SP800-53": "control", "SP800-53-enhancement": "control_enhancement",
         "category": "category", "subcategory": "subcategory"}


def _prop(obj: dict, name: str):
    return next((p["value"] for p in obj.get("props", []) if p.get("name") == name), None)


def _param_labels(control: dict) -> Dict[str, str]:
    out = {}
    for p in control.get("params", []):
        choices = (p.get("select") or {}).get("choice")
        out[p["id"]] = p.get("label") or ("; ".join(choices) if choices else "organization-defined value")
    return out


def _prose(part: dict, params: Dict[str, str]) -> List[str]:
    label = _prop(part, "label")
    prose = _PARAM.sub(lambda m: f"[{params.get(m.group(1), 'organization-defined value')}]", part.get("prose", ""))
    lines = [" ".join(x for x in (label, prose) if x)] if (label or prose) else []
    for child in part.get("parts", []):
        lines += _prose(child, params)
    return lines


def _part(control: dict, name: str, params: Dict[str, str]) -> str:
    return " ".join(line for p in control.get("parts", []) if p.get("name") == name for line in _prose(p, params))


def _walk(controls: List[dict], src: Source, path: List[str], params_up: Dict[str, str]) -> List[dict]:
    out = []
    for c in controls:
        if _prop(c, "status") == "withdrawn":
            continue
        params = {**params_up, **_param_labels(c)}
        label = _prop(c, "label") or c["id"]
        kind = _KIND.get(c.get("class", ""), "control")
        heading = f"{label} {c['title']}" if c["title"] != label else label
        out.append({"id": f"{src.prefix}/{c['id']}", "ref": f"{src.short} {label}", "kind": kind,
                    "framework": src.framework, "instrument": src.instrument, "binding": src.binding,
                    "licence": src.licence, "source_url": src.url, "title": c["title"],
                    "text": _part(c, "statement", params), "guidance": _part(c, "guidance", params) or None,
                    "path": path + [heading], "chapter": None, "section": None, "article": None,
                    "paragraph": None, "amended_by": []})
        out += _walk(c.get("controls", []), src, path + [heading], params)
    return out


def parse_oscal(catalog: dict, source: Source) -> List[dict]:
    out = []
    for group in catalog["groups"]:
        out += _walk(group.get("controls", []), source, [group["title"]], {})
    return out


def parse_playbook(entries: List[dict], source: Source) -> List[dict]:
    return [{"id": f"{source.prefix}/{e['title'].lower().replace(' ', '-')}", "ref": f"{source.short} {e['title']}",
             "kind": "subcategory", "framework": source.framework, "instrument": source.instrument,
             "binding": source.binding, "licence": source.licence, "source_url": source.url,
             "title": e["title"], "text": e["description"].strip(), "guidance": (e.get("section_about") or "").strip() or None,
             "path": [e["type"], e["category"], e["title"]], "chapter": None, "section": None, "article": None,
             "paragraph": None, "amended_by": []} for e in entries]
```

Every record's `path` is its ancestors plus its own heading (same rule as EU articles).

- [ ] Step 4: run → PASS. Step 5: commit `feat(corpus): NIST OSCAL and AI RMF Playbook parsers`.

### Task 4: Build CLI and the corpus snapshot

**Files:** Create `corpus/build.py`, `backend/app/services/regs/__init__.py`, `tests/corpus/test_build.py`; generate `backend/app/data/regs/{regs-v1.jsonl,manifest.json,NOTICE.md}`.

**Interfaces:** Produces `split_long(rec, max_chars=6000) -> list[dict]`, `build(raw_dir, out_dir, registry=None) -> dict(stats)`, `load_provisions(path=DEFAULT) -> list[dict]` in `app.services.regs`.

- [ ] Step 1: failing test

```python
# tests/corpus/test_build.py
from corpus.build import split_long

def test_long_records_split_on_sentence_boundaries_with_part_ids():
    rec = {"id": "x/art-1", "ref": "X Art. 1", "text": ("Sentence one is here. " * 400).strip()}
    parts = split_long(rec, max_chars=1000)
    assert len(parts) > 1 and all(len(p["text"]) <= 1000 for p in parts)
    assert parts[0]["id"] == "x/art-1/part-1" and parts[1]["ref"] == "X Art. 1 (part 2)"
    assert " ".join(p["text"] for p in parts) == rec["text"]

def test_short_records_untouched():
    rec = {"id": "x", "ref": "X", "text": "short"}
    assert split_long(rec) == [rec]
```

- [ ] Step 2: run → FAIL. Step 3: implement `split_long`, `build`, and `load_provisions`

```python
# corpus/build.py
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
```

```python
# backend/app/services/regs/__init__.py
"""Regulatory provision corpus (built by corpus/build.py)."""
import json
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "regs"


def load_provisions(path: Path = DATA_DIR / "regs-v1.jsonl") -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
```

`NOTICE.md` (hand-written, committed): one line per source with instrument, URL, licence, plus "Contains no ISO or AICPA text."

- [ ] Step 4: run tests → PASS. Build requires Task 5's registry: do Task 5 Steps 1–3 before running `python -m corpus.build`.
- [ ] Step 5: run `python -m corpus.build`; expected stats: all 8 prefixes present, no empty text, no duplicate ids; spot-check `eu-ai-act/art-26/par-1`, `gdpr/art-33/par-1`, `nist-800-53/ac-2`, `nist-csf/GV.OC-01`, `nist-ai-rmf/govern-1.1`.
- [ ] Step 6: commit `feat(corpus): build CLI and regs-v1 snapshot with manifest and NOTICE`.

### Task 5: Applicability registry and resolver

**Files:** Create `backend/app/services/regs/registry.py`, `backend/app/data/regs/applicability.yaml`, `tests/corpus/test_registry.py`.

**Interfaces:** Produces `Dated(date: Optional[date], condition: Optional[str], basis: str, quote: str, note: Optional[str])`, `Registry.load(yaml_path, provisions)`, `Registry.dates(p) -> list[Dated]`, `Registry.applies(p, as_of: date, condition=None) -> Optional[bool]`, `Registry.fields(p) -> {"applies_from": "YYYY-MM-DD"|None, "applies_note": str|None, "date_basis": str|None}`, `Registry.unverified_quotes() -> list[str]`.

Rule schema (YAML list; later rules override earlier ones for the same provision):

```yaml
- id: ai-act-general
  match: {prefix: eu-ai-act, kinds: [article, article_paragraph]}
  dates: [{from: 2026-08-02}]
  basis: eu-ai-act/art-113
  quote: "It shall apply from 2 August 2026"
```

`match` keys: `prefix` (required), `kinds`, `chapters`, `sections` (pairs like `III.1`), `articles`, `paragraphs` (like `5(1a)`), `exclude_articles`, `exclude_paragraphs`. `dates` is a list of `{from, condition}`; `note` optional.

Rules to write (quotes must occur verbatim in the basis text; Step 4 enforces it):

| id | match | dates | basis | quote |
|---|---|---|---|---|
| ai-act-general | eu-ai-act articles/paragraphs | 2026-08-02 | eu-ai-act/art-113 | "It shall apply from 2 August 2026" |
| ai-act-ch1-2 | chapters I, II | 2025-02-02 | eu-ai-act/art-113 | "Chapters I and II shall apply from 2 February 2025" |
| ai-act-art5-new | paragraphs 5(1a), 5(1b) | 2026-12-02 | eu-ai-act/art-113 | "Article 5(1a) and (1b) which shall apply from 2 December 2026" |
| ai-act-gpai-gov | sections III.4; chapters V, VII, XII; article 78; exclude article 101 | 2025-08-02 | eu-ai-act/art-113 | "shall apply from 2 August 2025" |
| ai-act-high-risk | sections III.1, III.2, III.3; exclude 6(5) | 2027-12-02 cond `annex_iii`; 2028-08-02 cond `annex_i` | eu-ai-act/art-113 | "2 December 2027 as regards AI systems classified as high-risk pursuant to Article 6(2) and Annex III" |
| ai-act-penalties-102-110 | articles 102–110 | 2026-07-27 | eu-ai-act/art-113 | "Articles 102 to 110 shall apply from 27 July 2026" |
| ai-act-recitals-annexes | kinds recital, annex, annex_point | none, note "interpretive (recital) or applies through the articles that reference it (annex)" | eu-ai-act/art-113 | "It shall apply from 2 August 2026" |
| gdpr | gdpr articles/paragraphs | 2018-05-25 | gdpr/art-99/par-2 | "It shall apply from 25 May 2018" |
| nis2 | nis2 articles/paragraphs | 2024-10-18, note "directive: obligations apply through national transposing law" | nis2/art-41/par-1 | quote taken from the built text |
| dora | dora articles/paragraphs | 2025-01-17 | dora/art-64 | "It shall apply from 17 January 2025" |
| cra-general / cra-art14 / cra-ch4 | per Art. 71(2) of the built text | dates from the built text | cra/art-71/par-2 | quotes from the built text |
| nist-voluntary | prefixes nist-800-53, nist-csf, nist-ai-rmf | none, note "voluntary framework; no legal effective date" | (none) | (none) |

For NIS2 and CRA the dates and quotes are read from the built corpus (Task 4 Step 5) before writing the rule; the quote test makes a wrong date impossible to commit.

- [ ] Step 1: failing tests

```python
# tests/corpus/test_registry.py
from datetime import date
from app.services.regs.registry import Registry

P = [
    {"id": "eu-ai-act/art-113", "kind": "article", "text": "It shall apply from 2 August 2026. However: (a) Chapters I and II shall apply from 2 February 2025; (c) Chapter III, Sections 1, 2, and 3, with the exception of Article 6(5), shall apply from: (i) 2 December 2027 as regards AI systems classified as high-risk pursuant to Article 6(2) and Annex III; and (ii) 2 August 2028 as regards AI systems classified as high-risk pursuant to Article 6(1) and Annex I;", "chapter": "XIII", "section": None, "article": "113", "paragraph": None},
    {"id": "eu-ai-act/art-5/par-1", "kind": "article_paragraph", "text": "x", "chapter": "II", "section": None, "article": "5", "paragraph": "1"},
    {"id": "eu-ai-act/art-26/par-1", "kind": "article_paragraph", "text": "x", "chapter": "III", "section": "3", "article": "26", "paragraph": "1"},
    {"id": "eu-ai-act/art-6/par-5", "kind": "article_paragraph", "text": "x", "chapter": "III", "section": "1", "article": "6", "paragraph": "5"},
]
RULES = [
    {"id": "general", "match": {"prefix": "eu-ai-act", "kinds": ["article", "article_paragraph"]},
     "dates": [{"from": "2026-08-02"}], "basis": "eu-ai-act/art-113", "quote": "It shall apply from 2 August 2026"},
    {"id": "ch1-2", "match": {"prefix": "eu-ai-act", "chapters": ["I", "II"]},
     "dates": [{"from": "2025-02-02"}], "basis": "eu-ai-act/art-113", "quote": "Chapters I and II shall apply from 2 February 2025"},
    {"id": "high-risk", "match": {"prefix": "eu-ai-act", "sections": ["III.1", "III.2", "III.3"], "exclude_paragraphs": ["6(5)"]},
     "dates": [{"from": "2027-12-02", "condition": "annex_iii"}, {"from": "2028-08-02", "condition": "annex_i"}],
     "basis": "eu-ai-act/art-113", "quote": "2 December 2027 as regards AI systems classified as high-risk pursuant to Article 6(2) and Annex III"},
]
REG = Registry(RULES, P)
by = {p["id"]: p for p in P}


def test_later_rules_override_and_conditions():
    assert REG.applies(by["eu-ai-act/art-5/par-1"], date(2026, 10, 20)) is True
    hr = by["eu-ai-act/art-26/par-1"]
    assert REG.applies(hr, date(2026, 10, 20)) is False
    assert REG.applies(hr, date(2027, 12, 2), "annex_iii") is True
    assert REG.applies(hr, date(2027, 12, 2), "annex_i") is False
    assert REG.fields(hr)["applies_from"] == "2027-12-02" and "annex_i" in REG.fields(hr)["applies_note"]


def test_exclusions_fall_back_to_earlier_rules():
    assert REG.fields(by["eu-ai-act/art-6/par-5"])["applies_from"] == "2026-08-02"


def test_unmatched_provisions_have_no_date():
    assert REG.applies({"id": "gdpr/art-1", "kind": "article"}, date(2026, 1, 1)) is None


def test_quotes_are_checked_against_basis_text():
    assert REG.unverified_quotes() == []
    bad = Registry(RULES + [{"id": "bad", "match": {"prefix": "eu-ai-act"}, "dates": [{"from": "2030-01-01"}],
                             "basis": "eu-ai-act/art-113", "quote": "shall apply from 1 January 2030"}], P)
    assert bad.unverified_quotes() == ["bad"]
```

Plus a corpus-backed test (skips if the corpus is not built):

```python
# tests/corpus/test_registry_corpus.py
import pytest
from app.services.regs import DATA_DIR, load_provisions
from app.services.regs.registry import Registry

def test_every_committed_rule_quote_occurs_in_its_basis_provision():
    path = DATA_DIR / "regs-v1.jsonl"
    if not path.exists():
        pytest.skip("corpus not built")
    reg = Registry.load(DATA_DIR / "applicability.yaml", load_provisions(path))
    assert reg.unverified_quotes() == []
```

- [ ] Step 2: run → FAIL. Step 3: implement

```python
# backend/app/services/regs/registry.py
"""Applicability dates for provisions. Each rule cites the provision (and verbatim quote) that sets it."""
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class Dated:
    date: Optional[date]
    condition: Optional[str]
    basis: str
    quote: str
    note: Optional[str]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _matches(m: dict, p: dict) -> bool:
    """`kinds` filters; chapters/sections/articles/paragraphs are alternatives (any may match); exclusions win."""
    if not p["id"].startswith(m["prefix"] + "/"):
        return False
    art, par = p.get("article"), p.get("paragraph")
    ref = f"{art}({par})" if par else art
    if art in [str(a) for a in m.get("exclude_articles", [])] or ref in m.get("exclude_paragraphs", []):
        return False
    if "kinds" in m and p.get("kind") not in m["kinds"]:
        return False
    scopes = {"chapters": p.get("chapter"), "sections": f"{p.get('chapter')}.{p.get('section')}",
              "articles": art, "paragraphs": ref}
    given = [k for k in scopes if k in m]
    return not given or any(scopes[k] in [str(x) for x in m[k]] for k in given)


class Registry:
    def __init__(self, rules: List[dict], provisions: List[dict]):
        self.rules = rules
        self.text = {p["id"]: _norm(p.get("text", "")) for p in provisions}

    @classmethod
    def load(cls, path: Path, provisions: List[dict]) -> "Registry":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or [], provisions)

    def _rule(self, p: dict) -> Optional[dict]:
        hit = None
        for rule in self.rules:
            if _matches(rule["match"], p):
                hit = rule
        return hit

    def dates(self, p: dict) -> List[Dated]:
        rule = self._rule(p)
        if not rule:
            return []
        entries = rule.get("dates") or [{}]
        return [Dated(date.fromisoformat(str(d["from"])) if d.get("from") else None, d.get("condition"),
                      rule.get("basis", ""), rule.get("quote", ""), rule.get("note")) for d in entries]

    def applies(self, p: dict, as_of: date, condition: Optional[str] = None) -> Optional[bool]:
        """True/False once known; None when undated, or when it depends on a condition the caller did not give."""
        ds = [d for d in self.dates(p) if d.date]
        if not ds:
            return None
        if condition is not None:
            chosen = [d for d in ds if d.condition in (None, condition)]
            return any(as_of >= d.date for d in chosen) if chosen else None
        unconditional = [d for d in ds if d.condition is None]
        if unconditional:
            return as_of >= unconditional[0].date
        if as_of < min(d.date for d in ds):
            return False
        return True if as_of >= max(d.date for d in ds) else None

    def fields(self, p: dict) -> dict:
        ds = self.dates(p)
        dated = [d for d in ds if d.date]
        if not ds:
            return {"applies_from": None, "applies_note": None, "date_basis": None}
        note = "; ".join(f"{d.date.isoformat()} if {d.condition}" for d in dated if d.condition) or None
        notes = "; ".join(x for x in (note, ds[0].note) if x) or None
        return {"applies_from": min(d.date for d in dated).isoformat() if dated else None,
                "applies_note": notes, "date_basis": ds[0].basis or None}

    def unverified_quotes(self) -> List[str]:
        return [r["id"] for r in self.rules if r.get("quote") and _norm(r["quote"]) not in self.text.get(r.get("basis", ""), "")]
```

(`None` = depends on the system's classification; the caller supplies `condition`.)

- [ ] Step 4: write `applicability.yaml` per the table, build the corpus (Task 4 Step 5), run `pytest tests/corpus -q` → all pass including the corpus-backed quote test.
- [ ] Step 5: commit `feat(regs): applicability registry with quoted legal basis per date`.

### Task 6: Lexical index

**Files:** Create `backend/app/services/regs/lexical.py`, `tests/corpus/test_lexical.py`.

**Interfaces:** Produces `tokenize(text) -> list[str]`, `BM25(docs: list[tuple[str, str]], k1=1.5, b=0.75)`, `BM25.search(query, k=10) -> list[tuple[str, float]]`, `BM25.from_provisions(provisions)` (indexes `ref + title + text`).

- [ ] Step 1: failing test

```python
# tests/corpus/test_lexical.py
from app.services.regs.lexical import BM25, tokenize

def test_tokenize_normalises_article_references():
    assert tokenize("What does Article 26(1) require?") == ["art", "26", "1", "require"]

def test_reference_queries_rank_the_cited_provision_first():
    idx = BM25.from_provisions([
        {"id": "a/art-26/par-1", "ref": "AI Act Art. 26(1)", "title": "Obligations of deployers", "text": "Deployers shall take measures."},
        {"id": "a/art-16", "ref": "AI Act Art. 16", "title": "Obligations of providers", "text": "Providers shall ensure compliance."},
        {"id": "g/art-33/par-1", "ref": "GDPR Art. 33(1)", "title": "Notification of a personal data breach", "text": "The controller shall notify the breach within 72 hours."},
    ])
    assert idx.search("Article 26 deployers", k=1)[0][0] == "a/art-26/par-1"
    assert idx.search("72 hours breach notification", k=1)[0][0] == "g/art-33/par-1"
    assert idx.search("zebra", k=3) == []
```

- [ ] Step 2: run → FAIL. Step 3: implement

```python
# backend/app/services/regs/lexical.py
"""Small in-memory BM25 over provisions: exact terms and article references dense retrieval misses."""
import math
import re
from collections import Counter
from typing import Iterable, List, Tuple

_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "by", "with", "is", "are", "be", "what",
         "does", "do", "which", "that", "this", "it", "as", "at", "from", "under", "shall", "how", "when"}
_ALIASES = {"article": "art", "articles": "art", "paragraph": "par", "recital": "rct", "annex": "anx"}


def tokenize(text: str) -> List[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        t = _ALIASES.get(t, t)
        if t not in _STOP:
            out.append(t)
    return out


class BM25:
    def __init__(self, docs: Iterable[Tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.ids, self.tfs, self.lens = [], [], []
        df: Counter = Counter()
        for doc_id, text in docs:
            toks = tokenize(text)
            tf = Counter(toks)
            self.ids.append(doc_id)
            self.tfs.append(tf)
            self.lens.append(len(toks))
            df.update(tf.keys())
        n = len(self.ids)
        self.avg = sum(self.lens) / n if n else 0.0
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    @classmethod
    def from_provisions(cls, provisions: Iterable[dict]) -> "BM25":
        return cls((p["id"], " ".join(x for x in (p.get("ref"), p.get("title"), p.get("text")) if x)) for p in provisions)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        q = [t for t in tokenize(query) if t in self.idf]
        if not q:
            return []
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                scores.append((self.ids[i], s))
        return sorted(scores, key=lambda x: -x[1])[:k]
```

- [ ] Step 4: run → PASS. Step 5: commit `feat(regs): BM25 lexical index over provisions`.

### Task 7: Provision-level golden set and lexical eval

**Files:** Create `eval/regs_golden.jsonl`, `eval/run_regs.py`, `tests/corpus/test_run_regs.py`.

**Interfaces:** Consumes `load_provisions`, `BM25`, `eval.metrics.recall_at_k/mrr/ndcg_at_k`. Produces `label_hits(ranked_ids, labels) -> list[str]` (maps provision ids to the first matching label, deduped; a label matches an id equal to it or prefixed by `label + "/"`), CLI `python eval/run_regs.py --mode lexical|dense|fused`.

Golden set rules: ~40 entries `{query, relevant: [ids or article-level ids], type: reference|concept|temporal, frameworks}`, labelled by reading the built provisions (never from retriever output), `_meta.label_status = "draft-claude-2026-10-20: pending human review"`. Coverage: ≥ 6 per EU instrument, ≥ 3 per NIST framework, ≥ 6 reference-style ("What does Article 26 of the AI Act require?"), ≥ 4 cross-framework concept questions ("incident notification deadlines").

- [ ] Step 1: failing test

```python
# tests/corpus/test_run_regs.py
from eval.run_regs import label_hits

def test_article_level_labels_match_their_paragraphs_once():
    ranked = ["x/art-26/par-2", "x/art-26/par-1", "y/art-33/par-1", "z/art-1"]
    assert label_hits(ranked, ["x/art-26", "y/art-33/par-1"]) == ["x/art-26", "y/art-33/par-1", "z/art-1"]
```

- [ ] Step 2: run → FAIL. Step 3: implement

```python
# eval/run_regs.py
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
```

(Check `eval/metrics.py` argument order before wiring; adapt the three calls to the real signatures.)

- [ ] Step 4: write `eval/regs_golden.jsonl` by reading provisions; run `python eval/run_regs.py --mode lexical`; record the baseline in the report.
- [ ] Step 5: commit `feat(eval): provision-level golden set and lexical retrieval baseline`.

### Task 8: Pinecone sizing and upsert script (stops before writing)

**Files:** Create `corpus/upsert.py`, `tests/corpus/test_upsert.py`.

**Interfaces:** Produces `to_record(p) -> dict` (Pinecone record: `_id`, `content` = ref + title + text (+ guidance, truncated 6000 chars), flat metadata without nulls, `applies_from` as int yyyymmdd), `estimate(provisions) -> dict`, CLI `python -m corpus.upsert` (dry run) / `--yes` (upsert to `regs-v1`).

- [ ] Step 1: failing test

```python
# tests/corpus/test_upsert.py
from corpus.upsert import estimate, to_record

P = {"id": "eu-ai-act/art-26/par-1", "ref": "AI Act Art. 26(1)", "title": "Obligations of deployers", "text": "Deployers shall...",
     "framework": "EU AI Act", "kind": "article_paragraph", "binding": "regulation", "applies_from": "2027-12-02",
     "applies_note": None, "path": ["Chapter III", "Section 3", "Article 26"], "guidance": None, "amended_by": []}

def test_record_is_flat_without_nulls_and_dates_are_ints():
    r = to_record(P)
    assert r["_id"] == P["id"] and r["content"].startswith("AI Act Art. 26(1) Obligations of deployers")
    assert r["applies_from"] == 20271202 and "applies_note" not in r and "guidance" not in r
    assert r["path"] == "Chapter III > Section 3 > Article 26"

def test_estimate_reports_tokens_and_storage():
    e = estimate([P] * 100)
    assert e["records"] == 100 and e["est_embed_tokens"] > 0 and e["est_storage_mb"] > 0
```

- [ ] Step 2: run → FAIL. Step 3: implement

```python
# corpus/upsert.py
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
```

- [ ] Step 4: run tests → PASS; run `python -m corpus.upsert` (dry run) and compare with Starter limits (2 GB storage; 5M embedding tokens per month per model). **Stop and ask the user before `--yes`.**
- [ ] Step 5: commit `feat(corpus): regs-v1 upsert script with dry-run sizing`.

After approval (outside this plan's automatic steps): `python -m corpus.upsert --yes`, then `python eval/run_regs.py --mode dense` and `--mode fused`; record the comparison and pick the fusion default for sub-project 2.
