# Crosswalks (sub-project 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attach published framework-to-framework mappings (with source and path) to every cited provision and checklist item, use official links in cross-framework retrieval only if it measurably helps, and add Implementing Regulation (EU) 2024/2690 so NIS2 links reach real provisions.

**Architecture:** A build step (`corpus/crosswalks.py`) turns the source tables (NIST ×2, ENISA, the regulation's own headings; plus locally fetched community tables) into id-to-id link records. A runtime module (`services/regs/crosswalk.py`) loads them into a graph and answers `related(...)` with shortest, official-first paths. Ask sources, checklist items and the voluntary practices consume it; retrieval gains an optional boost/inject step behind a setting.

**Tech Stack:** Python 3.11 stdlib (zipfile + ElementTree for xlsx), pypdf (dev-only, for one community PDF), FastAPI, Streamlit 1.66. No new runtime dependencies.

## Global Constraints

- Link records hold ids only, never mapping or standard text; ISO entries are identifiers shown as "licensed text, bring your copy".
- Official links (NIST CSF↔800-53, NIST 800-53↔ISO 27001, ENISA NIS2, 2024/2690 headings) are committed; community links (Razilio, Pauloweit, Microsoft 42001) are fetched locally into gitignored files and used only when present.
- Shown with matches: "Mappings show related requirements, not equivalence"; community → "community mapping, not endorsed by NIST"; 42001 → "maps ISO/IEC FDIS 42001".
- Licensed ISO nodes are leaves: paths never pass through them.
- Retrieval expansion is kept only if cross recall rises above 0.483 at k = 6 and k = 10 with overall recall/MRR within 0.01; otherwise the setting stays `off`.
- Ask before the Pinecone upsert of the new 2024/2690 records (exact count and estimate shown).
- Never `git commit -a`; stage paths explicitly. Downloaded files are untrusted: read them with `python -I`.
- Backend tests: `.venv-relock/Scripts/python -m pytest -p no:cacheprovider tests -q`. UI smoke: `PYTHONPATH=frontend .venv-revive-ui/Scripts/python tests/ui/smoke_app.py`.

---

## File structure

| File | Responsibility |
|---|---|
| `corpus/sources.py`, `corpus/parse_eu.py` | 2024/2690 source; decimal-numbered annex points |
| `backend/app/data/regs/applicability.yaml` | 2024/2690 date rule |
| `backend/app/data/regs/facts.yaml`, `backend/app/schemas/chat.py`, `obligations.yaml` | split `online_platform`; 13 new checklist items |
| `corpus/crosswalks.py` | xlsx reader, id normalisers, per-source parsers, `fetch`, `build` |
| `backend/app/data/regs/crosswalk_sources.yaml` | manifest of the seven sources |
| `backend/app/data/regs/crosswalks.jsonl`, `crosswalk_report.json` | committed official links + build report |
| `backend/app/services/regs/crosswalk.py` | graph, `related`, `related_many`, `linked_ids` |
| `backend/app/services/regs/ask.py`, `checklist.py`, `prompts/regs_prompt.py` | attach matches, expansion, practices narrowing, prompt header |
| `frontend/services/presenters.py`, `views/ask.py`, `views/checklist.py` | matching-requirement lists |
| `eval/run_regs.py`, `eval/regs_golden.jsonl`, `eval/regs_profiles.jsonl` | crosswalk modes; new golden rows |

---

### Task 1: Implementing Regulation 2024/2690 in the corpus

**Files:** Modify `corpus/sources.py`, `corpus/parse_eu.py`, `backend/app/data/regs/applicability.yaml`; Tests `tests/corpus/test_parse_eu.py`, `tests/regs/test_status_corpus.py`, `tests/corpus/test_build.py`.

**Interfaces (produced):** provision ids `nis2-ir/art-N[/par-M]`, `nis2-ir/anx-1` (annex head), `nis2-ir/anx-1/pt-S.P` (47 points); annex point `path` = `["Annex 1", "<section heading with (Article 21(2), point (x) …)>", "S.P <title>"]`; ref `NIS2 IR Annex, point S.P`.

- [ ] **Step 1: failing tests.** Append to `tests/corpus/test_parse_eu.py`:

```python
DECIMAL_ANNEX = """<html><body><div id="anx_1">
<p class="oj-doc-ti">ANNEX</p><p class="oj-doc-ti">Technical and methodological requirements referred to in Article 2</p>
<p class="oj-ti-grseq-1">1. Policy on the security of network and information systems (Article 21(2), point (a) of Directive (EU) 2022/2555)</p>
<p class="oj-ti-grseq-1">1.1. Policy on the security of network and information systems</p>
<table><tr><td>1.1.1. For the purpose of Article 21(2), point (a), the policy shall set out the approach.</td></tr></table>
<table><tr><td>1.1.2. The policy shall be reviewed at planned intervals.</td></tr></table>
<p class="oj-ti-grseq-1">1.2. Roles, responsibilities and authorities</p>
<table><tr><td>1.2.1. Roles shall be assigned.</td></tr></table>
<p class="oj-ti-grseq-1">11. Access control (Article 21(2), points (i) and (j), of Directive (EU) 2022/2555)</p>
<p class="oj-ti-grseq-1">11.1. Access control policy</p>
<table><tr><td>11.1.1. An access control policy shall be established.</td></tr></table>
</div></body></html>"""


def test_decimal_numbered_annex_becomes_one_record_per_point():
    from corpus.sources import Source
    src = Source("nis2-ir", "nis2-ir", "NIS2 IR", "NIS2", "Implementing Regulation (EU) 2024/2690", "u", "eu-xhtml",
                 "l", "regulation")
    recs = {r["id"]: r for r in parse_eu(DECIMAL_ANNEX, src)}
    assert list(recs) == ["nis2-ir/anx-1", "nis2-ir/anx-1/pt-1.1", "nis2-ir/anx-1/pt-1.2", "nis2-ir/anx-1/pt-11.1"]
    p = recs["nis2-ir/anx-1/pt-1.1"]
    assert p["kind"] == "annex_point" and p["ref"] == "NIS2 IR Annex, point 1.1"
    assert p["title"] == "Policy on the security of network and information systems"
    assert p["text"].startswith("1.1.1. For the purpose") and "1.1.2. The policy shall be reviewed" in p["text"]
    assert "(Article 21(2), point (a)" in p["path"][1]
    assert "(Article 21(2), points (i) and (j)" in recs["nis2-ir/anx-1/pt-11.1"]["path"][1]
```

(`parse_eu` is already imported at the top of that file.) Append to `tests/regs/test_status_corpus.py` parametrize list:
`("nis2-ir/art-2/par-1", None, "applies", "2024-11-07"),` and `("nis2-ir/anx-1/pt-5.1", None, "applies", "2024-11-07"),`.
Append to `tests/corpus/test_build.py`:

```python
def test_implementing_regulation_2024_2690_is_in_the_corpus():
    import pytest
    from app.services.regs import DATA_DIR, load_provisions
    if not (DATA_DIR / "regs-v1.jsonl").exists():
        pytest.skip("corpus not built")
    ids = [p["id"] for p in load_provisions() if p["id"].startswith("nis2-ir/")]
    points = [i for i in ids if i.startswith("nis2-ir/anx-1/pt-")]
    assert len(points) == 47 and "nis2-ir/anx-1/pt-5.1" in points and "nis2-ir/anx-1/pt-13.3" in points
    assert any(i.startswith("nis2-ir/art-16") for i in ids) and any(i.startswith("nis2-ir/art-7") for i in ids)
```

- [ ] **Step 2:** FAIL (`pt-1.1` not produced; corpus lacks `nis2-ir`).
- [ ] **Step 3: implement.** `corpus/sources.py`, add after the NIS2 line:

```python
    _eu("nis2-ir", "nis2-ir", "NIS2 IR", "NIS2", "Commission Implementing Regulation (EU) 2024/2690", "32024R2690"),
```

`corpus/parse_eu.py`, add above `_annex` and call it first inside `_annex`:

```python
_DEC_SECTION = re.compile(r"^(\d+)\.\s+\S")
_DEC_POINT = re.compile(r"^(\d+\.\d+)\.\s+(.+)$")


def _decimal_points(anx: Tag, src: Source, common: dict, num: str) -> List[dict]:
    """Annexes numbered 1. / 1.1. / 1.1.1. in plain headings and tables (e.g. Implementing Regulation
    2024/2690): one record per second-level point, its sub-points as text, its section heading in the path."""
    out, section, point = [], None, None
    for block in anx.find_all(["p", "table"], recursive=False):
        text = _text(block)
        if block.name == "p" and "oj-ti-grseq-1" in (block.get("class") or []):
            m = _DEC_POINT.match(text)
            if m:
                point = {**common, "id": f"{src.prefix}/anx-{num}/pt-{m.group(1)}",
                         "ref": f"{src.short} Annex, point {m.group(1)}", "kind": "annex_point",
                         "title": m.group(2), "text": "", "path": [f"Annex {num}", section, f"{m.group(1)} {m.group(2)}"]}
                out.append(point)
                continue
            if _DEC_SECTION.match(text):
                section, point = text, None
                continue
        if point is not None and text:
            point["text"] = f"{point['text']} {text}".strip()
    return out
```

and at the start of `_annex`, after `out = [...]` (the annex head record) is built:

```python
    decimal = _decimal_points(anx, src, common, num)
    if decimal:
        out[0]["text"] = " ".join(heads + [_text(p) for p in anx.find_all("p", class_="oj-ti-grseq-1", recursive=False)
                                           if _DEC_SECTION.match(_text(p)) and not _DEC_POINT.match(_text(p))])
        return out + decimal
```

`applicability.yaml`, after the NIS2 block:

```yaml
# ------------------------------------------------------ NIS2 Implementing Regulation (EU) 2024/2690
- id: nis2-ir
  match: {prefix: nis2-ir, kinds: [article, article_paragraph, annex, annex_point]}
  dates: [{from: 2024-11-07}]
  basis: nis2-ir/art-16
  quote: "This Regulation shall enter into force on the twentieth day following that of its publication in the Official Journal of the European Union"
  note: "published in the Official Journal (L series) on 18.10.2024, so in force from 7 November 2024; directly applicable"

- id: nis2-ir-recitals
  match: {prefix: nis2-ir, kinds: [recital]}
  dates: [{}]
  basis: nis2-ir/art-16
  note: "recital: guides interpretation, no date of its own"
```

`backend/app/services/regs/ask.py`, `_FRAMEWORK_ALIASES["NIS2"]` gains `r"2024/2690"`.

- [ ] **Step 4:** `python -m corpus.build` (raw file already present; other sources come from cache, nothing re-downloaded) → `regs-v1.jsonl` and `manifest.json` regenerated; full suite green. Record the new record count.
- [ ] **Step 5:** commit `feat(corpus): Implementing Regulation (EU) 2024/2690 (47 annex points) with its date`.

### Task 2: Checklist items for 2024/2690

**Files:** Modify `backend/app/data/regs/facts.yaml`, `backend/app/schemas/chat.py`, `backend/app/data/regs/obligations.yaml`, `backend/app/services/regs/checklist.py`, `eval/regs_profiles.jsonl`; Tests `tests/regs/test_catalog.py`, `tests/regs/test_checklist.py`, `tests/regs/test_profiles_file.py`.

**Interfaces:** NIS2 sector values `online_marketplace | search_engine | social_network` replace `online_platform`; catalog entries may carry `include: [<article-key>, …]` whose provisions are appended to the item's paragraphs.

- [ ] **Step 1: failing tests.** In `tests/regs/test_catalog.py` the count assertion becomes
`assert counts == {"eu-ai-act": 32, "gdpr": 34, "nis2": 4, "nis2-ir": 13, "dora": 19, "cra": 10}` with `"nis2-ir"` added to the prefix tuple. In `tests/regs/test_checklist.py` replace 99 by 112 in `test_nothing_known…` and `test_summary_text…`, and add:

```python
def test_implementing_regulation_items_follow_the_entity_type():
    cl = build_checklist(PROFILES["eu_cloud_large"]["profile"], date(2026, 10, 24))
    now = {i["article"]: i for i in cl["groups"]["now"]}
    assert {"nis2-ir/art-2", "nis2-ir/art-3", "nis2-ir/art-7"} <= set(now)
    assert any(p["id"] == "nis2-ir/anx-1/pt-5.1" for p in now["nis2-ir/art-2"]["paragraphs"])  # include: annex
    assert now["nis2-ir/art-2"]["status"] == "applies" and now["nis2-ir/art-2"]["applies_from"] == "2024-11-07"
```

`tests/regs/test_profiles_file.py`: 12 → 13 profiles.
- [ ] **Step 2:** FAIL. **Step 3: implement.**
- `facts.yaml` `nis2_sector`: values `[cloud, data_centre, cdn, dns_tld, trust_services, electronic_comms, managed_services, online_marketplace, search_engine, social_network, other_annex, none]`; labels add `online_marketplace: online marketplace, search_engine: online search engine, social_network: social networking platform`, remove `online_platform`.
- `schemas/chat.py` `Nis2Sector`: same values.
- `obligations.yaml`: in `nis2_entity` and `nis2/art-27` replace `online_platform` with `online_marketplace, search_engine, social_network`; add to `defs`:

```yaml
  nis2_ir_entity: {all: [{use: nis2_entity}, {nis2_sector: [dns_tld, cloud, data_centre, cdn, managed_services, online_marketplace, search_engine, social_network, trust_services]}]}
```

and append (quotes are checked by `catalog_problems`; if an article has numbered paragraphs, the check names the `/par-1` id to use):

```yaml
- article: nis2-ir/art-2
  when: {use: nis2_ir_entity}
  include: [nis2-ir/anx-1]
  basis: nis2-ir/art-1
  quote: "This Regulation, with regard to DNS service providers, TLD name registries, cloud computing service providers, data centre service providers, content delivery network providers, managed service providers, managed security service providers"
  note: "The Annex sets out the measures point by point; each section names the NIS2 Art. 21(2) point it implements."
- article: nis2-ir/art-3
  when: {use: nis2_ir_entity}
  basis: nis2-ir/art-3/par-1
  quote: "An incident shall be considered to be significant for the purposes of Article 23(3) of Directive (EU) 2022/2555 with regard to the relevant entities"
- article: nis2-ir/art-4
  when: {use: nis2_ir_entity}
  basis: nis2-ir/art-4
  quote: "Incidents that individually are not considered a significant incident within the meaning of Article 3, shall be considered collectively as one significant incident"
- article: nis2-ir/art-5
  when: {all: [{use: nis2_ir_entity}, {nis2_sector: dns_tld}]}
  basis: nis2-ir/art-5
  quote: "With regard to DNS service providers, an incident shall be considered significant"
  note: "Only if you are a DNS service provider."
- article: nis2-ir/art-6
  when: {all: [{use: nis2_ir_entity}, {nis2_sector: dns_tld}]}
  basis: nis2-ir/art-6
  quote: "With regard to TLD name registries, an incident shall be considered significant"
  note: "Only if you are a TLD name registry."
- article: nis2-ir/art-7
  when: {all: [{use: nis2_ir_entity}, {nis2_sector: cloud}]}
  basis: nis2-ir/art-7
  quote: "With regard to cloud computing service providers, an incident shall be considered significant"
- article: nis2-ir/art-8
  when: {all: [{use: nis2_ir_entity}, {nis2_sector: data_centre}]}
  basis: nis2-ir/art-8
  quote: "With regard to data centre service providers, an incident shall be considered significant"
- article: nis2-ir/art-9
  when: {all: [{use: nis2_ir_entity}, {nis2_sector: cdn}]}
  basis: nis2-ir/art-9
  quote: "With regard to content delivery network providers, an incident shall be considered significant"
- article: nis2-ir/art-10
  when: {all: [{use: nis2_ir_entity}, {nis2_sector: managed_services}]}
  basis: nis2-ir/art-10
  quote: "With regard to managed service providers and managed security service providers, an incident shall be considered significant"
- article: nis2-ir/art-11
  when: {all: [{use: nis2_ir_entity}, {nis2_sector: online_marketplace}]}
  basis: nis2-ir/art-11
  quote: "With regard to providers of online marketplaces, an incident shall be considered significant"
- article: nis2-ir/art-12
  when: {all: [{use: nis2_ir_entity}, {nis2_sector: search_engine}]}
  basis: nis2-ir/art-12
  quote: "With regard to providers of online search engines, an incident shall be considered significant"
- article: nis2-ir/art-13
  when: {all: [{use: nis2_ir_entity}, {nis2_sector: social_network}]}
  basis: nis2-ir/art-13
  quote: "With regard to providers of social networking services platforms, an incident shall be considered significant"
- article: nis2-ir/art-14
  when: {all: [{use: nis2_ir_entity}, {nis2_sector: trust_services}]}
  basis: nis2-ir/art-14
  quote: "With regard to trust service providers, an incident shall be considered significant"
```

- `checklist.py` `build_checklist`: `paras = index[ob["article"]] + [p for key in ob.get("include", []) for p in index[key]]`; `catalog_problems` checks every `include` key is in the index (`f"{a}: include {key} not in the corpus"`).
- `eval/regs_profiles.jsonl`: `eu_cloud_large` gains `"nis2-ir/art-2", "nis2-ir/art-3", "nis2-ir/art-7"` in `binds` and `"nis2-ir/art-5", "nis2-ir/art-13"` in `not`; `dns_micro` gains `"nis2-ir/art-2", "nis2-ir/art-5", "nis2-ir/art-6"` in `binds` and `"nis2-ir/art-7"` in `not`; new row:

```json
{"name": "eu_social_network_large", "as_of": "2026-10-24", "profile": {"ai_system": false, "gpai_model": false, "eu_established": true, "eu_market": true, "size": "large", "personal_data": true, "data_roles": ["controller"], "sensitive_data": false, "large_scale_core": true, "automated_decisions": false, "third_country_transfers": true, "nis2_sector": "social_network", "financial_entity": false, "product_roles": []}, "binds": ["nis2/art-21", "nis2/art-27", "nis2-ir/art-2", "nis2-ir/art-13", "gdpr/art-37"], "depends": [], "not": ["nis2-ir/art-11", "nis2-ir/art-12", "nis2-ir/art-7"]}
```

- [ ] **Step 4:** full suite green. **Step 5:** commit `feat(regs): checklist items for Implementing Regulation 2024/2690`.

### Task 3: Crosswalk build pipeline

**Files:** Create `corpus/crosswalks.py`, `backend/app/data/regs/crosswalk_sources.yaml`, `tests/corpus/test_crosswalks.py`; Modify `.gitignore`, `requirements-dev.in`, `requirements-dev.txt`. Outputs committed: `backend/app/data/regs/crosswalks.jsonl`, `backend/app/data/regs/crosswalk_report.json`.

**Interfaces (produced):** link record `{"from": str, "to": str, "source": str, "kind": "official"|"community", "note"?: str}`; `read_xlsx(path) -> Dict[str, List[Dict[str, str]]]`; `nist_controls(cell) -> List[str]`; `csf_ids(cell) -> List[str]`; `iso27001_ids(cell) -> List[str]`; `rmf_iso42001_pairs(lines) -> List[Tuple[str, str]]`; `nis2_ir_headings(provisions) -> List[Tuple[str, str, str]]`; CLI `python -m corpus.crosswalks fetch|build`.

- [ ] **Step 1: failing tests** — `tests/corpus/test_crosswalks.py`:

```python
import io
import json
import zipfile
from pathlib import Path

import pyexpat

from corpus import crosswalks as cw

DATA = Path(__file__).resolve().parents[2] / "backend/app/data/regs"


def test_expat_resists_entity_expansion():
    assert pyexpat.version_info >= (2, 4, 1)  # billion-laughs protection; ElementTree never loads external entities


def test_nist_control_ids_normalise_to_corpus_ids():
    assert cw.nist_controls("AC-02(01)\nPM-11, SR-03") == ["nist-800-53/ac-2.1", "nist-800-53/pm-11", "nist-800-53/sr-3"]
    assert cw.nist_controls("") == []


def test_csf_ids_keep_categories_and_subcategories_only_from_csf2():
    assert cw.csf_ids("GV.SC-01, GV.SC, PR.AC-1 ID.RA-05") == ["nist-csf/GV.SC-01", "nist-csf/GV.SC", "nist-csf/ID.RA-05"]


def test_iso27001_ids_from_every_source_shape():
    assert cw.iso27001_ids("5.2, A.5.1, A5.7, А.5.19 ") == [
        "iso-27001/5.2", "iso-27001/A.5.1", "iso-27001/A.5.7", "iso-27001/A.5.19"]
    assert cw.iso27001_ids("4.0999999999999996") == ["iso-27001/4.1"]
    assert cw.iso27001_ids("Mandatory Clause: 6.1") == ["iso-27001/6.1"]
    assert cw.iso27001_ids("Annex A: A.8.15") == ["iso-27001/A.8.15"]
    assert cw.iso27001_ids("A.9.4, 11.2, A.5.99") == []  # 2013 numbering and out-of-range ids are rejected


def test_rmf_to_iso42001_lines():
    lines = ["Govern 1.1 Legal and regulatory requirements", "involving AI are understood,",
             "4.1 Understanding the organization and its context", "B.2.2 AI policy",
             "AI RMF ISO/IEC FDIS 42001", "Map 1.2 Interdisciplinary AI actors", "6.1.2 AI risk assessment"]
    assert cw.rmf_iso42001_pairs(lines) == [("nist-ai-rmf/govern-1.1", "iso-42001-fdis/4.1"),
                                            ("nist-ai-rmf/govern-1.1", "iso-42001-fdis/B.2.2"),
                                            ("nist-ai-rmf/map-1.2", "iso-42001-fdis/6.1.2")]


def test_nis2_ir_headings_link_annex_points_to_article_21_2():
    provs = [{"id": "nis2-ir/anx-1/pt-11.1", "path": ["Annex 1", "11. Access control (Article 21(2), points (i) and (j), of Directive (EU) 2022/2555)", "11.1 Access control policy"]},
             {"id": "nis2-ir/art-2/par-1", "path": ["Article 2"]}]
    assert cw.nis2_ir_headings(provs) == [("nis2-ir/anx-1/pt-11.1", "nis2/art-21/par-2", "Art. 21(2) points (i), (j)")]


def _xlsx(rows):
    """Minimal one-sheet workbook with inline strings."""
    cells = "".join(f'<row r="{i}">' + "".join(f'<c r="{c}{i}" t="inlineStr"><is><t>{v}</t></is></c>'
                                              for c, v in r.items()) + "</row>" for i, r in enumerate(rows, 1))
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rel = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook {ns} {rel}><sheets><sheet name="Relationships" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/></Relationships>')
        z.writestr("xl/worksheets/sheet1.xml", f"<worksheet {ns}><sheetData>{cells}</sheetData></worksheet>")
    return buf.getvalue()


def test_read_xlsx_and_csf_800_53_parser(tmp_path):
    path = tmp_path / "x.xlsx"
    path.write_bytes(_xlsx([{"A": "Focal", "C": "Ref"}, {"A": "GV"}, {"A": "GV.OC-01", "C": "PM-11"},
                            {"A": "GV.OC-02", "C": "PM-09\nPM-30(01)"}]))
    assert cw.read_xlsx(path)["Relationships"][2] == {"A": "GV.OC-01", "C": "PM-11"}
    assert cw.nist_csf_800_53(path) == [("nist-csf/GV.OC-01", "nist-800-53/pm-11"), ("nist-csf/GV.OC-02", "nist-800-53/pm-9"),
                                        ("nist-csf/GV.OC-02", "nist-800-53/pm-30.1")]


def test_committed_official_links_resolve_and_carry_no_text():
    from app.services.regs import load_provisions
    ids = {p["id"] for p in load_provisions()}
    links = [json.loads(l) for l in (DATA / "crosswalks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {l["kind"] for l in links} == {"official"}
    assert {l["source"] for l in links} == {"nist-csf-800-53", "nist-800-53-iso27001", "enisa-nis2", "nis2-ir-headings"}
    for l in links:
        assert set(l) <= {"from", "to", "source", "kind", "note"}
        assert all(x in ids or x.startswith("iso-") for x in (l["from"], l["to"])), l
    report = json.loads((DATA / "crosswalk_report.json").read_text(encoding="utf-8"))
    assert report["nist-csf-800-53"]["links"] > 500 and report["enisa-nis2"]["links"] > 200
    assert report["nis2-ir-headings"]["links"] == 47
    assert len(report["nist-csf-800-53"]["unresolved"]) <= 25


def test_lookup_gate_sampled_raw_rows_exist_in_the_index():
    """Local only: re-read rows from the raw tables and check each link was built."""
    import pytest
    raw = cw.RAW / "nist-csf-800-53"
    if not raw.exists():
        pytest.skip("raw crosswalk files not fetched (python -m corpus.crosswalks fetch)")
    links = {(l["from"], l["to"]) for l in map(json.loads, (DATA / "crosswalks.jsonl").read_text(encoding="utf-8").splitlines())}
    for src in cw.load_manifest():
        if src["kind"] != "official" or src["id"] == "nis2-ir-headings":
            continue
        ids = _ids()
        pairs = cw.PARSERS[src["id"]](cw.RAW / src["id"] / src["file"])
        resolvable = [p for p in pairs if all(x in ids or x.startswith("iso-") for x in p)]
        sample = resolvable[:: max(1, len(resolvable) // 25)]
        missing = [p for p in sample if p not in links]
        assert sample and missing == [], (src["id"], missing[:5])


def _ids():
    from app.services.regs import load_provisions
    return {p["id"] for p in load_provisions()}
```

- [ ] **Step 2:** FAIL (no module).
- [ ] **Step 3: implement.** `.gitignore` adds `backend/app/data/regs/crosswalks.community.jsonl`. `requirements-dev.in` adds `pypdf  # corpus/crosswalks.py reads one community PDF crosswalk (build time only)`; regenerate `requirements-dev.txt` with the header command and install it into `.venv-relock`.

`backend/app/data/regs/crosswalk_sources.yaml` (sha256 = our hashes of the downloaded files, printed by `fetch`):

```yaml
# Published mappings used for crosswalks (spec §13). Only id-to-id links are extracted; no mapping text is stored.
# official: committed (crosswalks.jsonl). community: fetched locally, built into the gitignored
# crosswalks.community.jsonl, shown only when present.
- id: nist-csf-800-53
  title: NIST CSF 2.0 to SP 800-53 Rev. 5.2.0
  publisher: NIST
  kind: official
  url: https://csrc.nist.gov/csrc/media/projects/olir/documents/submissions/Cybersecurity_Framework_v2-0_Concept_Crosswalk_800-53_5_2_0_draft.xlsx
  file: Cybersecurity_Framework_v2-0_Concept_Crosswalk_800-53_5_2_0_draft.xlsx
  olir: 186
  sha256: <from fetch>
  reuse: U.S. Government work (17 U.S.C. 105)
  caveat: Not comprehensive (OLIR). The OLIR-listed SHA3-256 does not match the downloaded file.
- id: nist-800-53-iso27001
  title: NIST SP 800-53 Rev. 5 to ISO/IEC 27001:2022
  publisher: NIST
  kind: official
  url: https://csrc.nist.gov/csrc/media/Projects/olir/documents/submissions/sp800-53r5-to-iso-27001-mapping-2022-OLIR-2023-10-12-UPDATED.xlsx
  file: sp800-53r5-to-iso-27001-mapping-2022-OLIR-2023-10-12-UPDATED.xlsx
  olir: 155
  sha256: <from fetch>
  reuse: U.S. Government work (17 U.S.C. 105); ISO identifiers only
  caveat: Focal document 800-53 5.1.1. "Organizations should not assume security requirement and control equivalency based solely on the mapping tables."
- id: enisa-nis2
  title: ENISA NIS2 Technical Implementation Guidance, mapping table v1.2
  publisher: ENISA
  kind: official
  url: https://www.enisa.europa.eu/sites/default/files/2025-09/ENISA_Technical_Implementation_Guidance_Mapping_table_version_1.2.xlsx
  file: ENISA_Technical_Implementation_Guidance_Mapping_table_version_1.2.xlsx
  sha256: <from fetch>
  reuse: "Source: ENISA. Reproduction of ENISA material is authorised provided the source is acknowledged (enisa.europa.eu legal notice)."
  caveat: Not legally binding (ENISA). Only the ISO/IEC 27001:2022 and NIST CSF 2.0 columns are used.
- id: nis2-ir-headings
  title: Implementing Regulation (EU) 2024/2690 annex headings
  publisher: European Commission
  kind: official
  url: http://publications.europa.eu/resource/celex/32024R2690
  file: (corpus)
  reuse: Commission Decision 2011/833/EU
  caveat: Each annex section heading names the NIS2 Art. 21(2) point it implements.
- id: razil-csf-iso27001
  title: NIST CSF 2.0 to ISO/IEC 27001:2022 Annex A (Razilio)
  publisher: Razilio (private sector, OLIR 154)
  kind: community
  url: https://irp.cdn-website.com/0bfa7e96/files/uploaded/Cybersecurity_Framework_v2.0_Concept_Crosswalk_Template_v1.0.xlsx
  file: Cybersecurity_Framework_v2.0_Concept_Crosswalk_Template_v1.0.xlsx
  olir: 154
  sha256: <from fetch>
  reuse: No licence stated; not redistributed.
  caveat: Community mapping, not endorsed by NIST.
- id: pauloweit-iso27001-csf
  title: ISO/IEC 27001:2022 to NIST CSF 2.0 (J. Pauloweit)
  publisher: J. Pauloweit (individual, OLIR 218)
  kind: community
  url: https://github.com/JoergPauloweit/ISO27001-NIST-Mapping/raw/main/Cybersecurity%20Framework%20v2.0_Concept%20Crosswalk_Template_Version%202_Pauloweit.xlsx
  file: Cybersecurity_Framework_v2.0_Concept_Crosswalk_Template_Version_2_Pauloweit.xlsx
  olir: 218
  sha256: <from fetch>
  reuse: No licence stated; not redistributed.
  caveat: Community mapping, not endorsed by NIST.
- id: msft-airmf-iso42001
  title: NIST AI RMF to ISO/IEC FDIS 42001 (Microsoft)
  publisher: Microsoft (via NIST AI Resource Center)
  kind: community
  url: https://airc.nist.gov/docs/NIST_AI_RMF_to_ISO_IEC_42001_Crosswalk.pdf
  file: NIST_AI_RMF_to_ISO_IEC_42001_Crosswalk.pdf
  sha256: <from fetch>
  reuse: No licence stated; not redistributed.
  caveat: Community mapping, not endorsed by NIST. Maps ISO/IEC FDIS 42001 (pre-publication), not 42001:2023.
```

`corpus/crosswalks.py`:

```python
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
```

- [ ] **Step 4:** `python -m corpus.crosswalks fetch` (files already present; prints hashes → paste into the manifest), `python -I -m corpus.crosswalks build`; read `crosswalk_report.json` (unresolved ids listed, e.g. controls new or changed between 800-53 5.1.1 and 5.2.0, ENISA point ids that do not match the corpus); fix normalisers for any systematic mismatch, then the suite → green.
- [ ] **Step 5:** commit `feat(corpus): crosswalk build from NIST, ENISA and the 2024/2690 headings` (stage the manifest, the two committed outputs, `corpus/crosswalks.py`, tests, `.gitignore`, dev requirements).

### Task 4: Crosswalk lookups

**Files:** Create `backend/app/services/regs/crosswalk.py`, `tests/regs/test_crosswalk.py`.

**Interfaces (produced):** `related(pid, hops=2, official_only=False, per_group=4) -> {"items": [Match], "more": {group: int}}`; `related_many(ids, hops=2, official_only=False, per_group=4)` same shape (union); `linked_ids(ids, prefix, hops=2) -> List[str]`. `Match = {"id", "group", "ref", "title", "licensed": bool, "kind": "official"|"community", "source", "source_title", "hops", "via": [ref, …], "note"}`.

- [ ] **Step 1: failing tests**

```python
from app.services.regs import crosswalk as xw

LINKS = [
    {"from": "nist-csf/GV.SC-01", "to": "nist-800-53/sr-3", "source": "nist-csf-800-53", "kind": "official"},
    {"from": "nis2-ir/anx-1/pt-5.1", "to": "nist-csf/GV.SC-01", "source": "enisa-nis2", "kind": "official"},
    {"from": "nis2-ir/anx-1/pt-5.1", "to": "nis2/art-21/par-2", "source": "nis2-ir-headings", "kind": "official", "note": "Art. 21(2) point (d)"},
    {"from": "nist-800-53/sr-3", "to": "iso-27001/A.5.21", "source": "nist-800-53-iso27001", "kind": "official"},
    {"from": "iso-27001/A.5.21", "to": "nist-csf/GV.SC-07", "source": "razil-csf-iso27001", "kind": "community"},
]


def _use(monkeypatch, links=LINKS):
    monkeypatch.setattr(xw, "_links", lambda: links)
    xw.graph.cache_clear()


def test_two_hop_paths_with_sources_and_notes(monkeypatch):
    _use(monkeypatch)
    out = xw.related("nis2/art-21/par-2")
    by_id = {m["id"]: m for m in out["items"]}
    assert by_id["nis2-ir/anx-1/pt-5.1"]["hops"] == 1 and by_id["nis2-ir/anx-1/pt-5.1"]["note"] == "Art. 21(2) point (d)"
    csf = by_id["nist-csf/GV.SC-01"]
    assert csf["hops"] == 2 and csf["via"] == ["NIS2 IR Annex, point 5.1"] and csf["source"] == "enisa-nis2"
    assert "nist-800-53/sr-3" not in by_id  # three hops away


def test_licensed_ids_are_leaves_and_labelled(monkeypatch):
    _use(monkeypatch)
    out = xw.related("nist-800-53/sr-3")
    iso = next(m for m in out["items"] if m["id"] == "iso-27001/A.5.21")
    assert iso["licensed"] and iso["ref"] == "ISO/IEC 27001:2022 A.5.21" and iso["group"] == "ISO/IEC 27001:2022"
    assert "nist-csf/GV.SC-07" not in {m["id"] for m in out["items"]}  # never walks through a licensed id


def test_official_only_and_caps(monkeypatch):
    _use(monkeypatch)
    assert all(m["kind"] == "official" for m in xw.related("iso-27001/A.5.21", official_only=True)["items"])
    many = [{"from": "nist-csf/GV.SC-01", "to": f"nist-800-53/sr-{i}", "source": "nist-csf-800-53", "kind": "official"} for i in range(1, 9)]
    _use(monkeypatch, many)
    out = xw.related("nist-csf/GV.SC-01", per_group=3)
    assert len(out["items"]) == 3 and out["more"] == {"NIST SP 800-53": 5}


def test_related_many_and_linked_ids(monkeypatch):
    _use(monkeypatch)
    out = xw.related_many(["nis2/art-21/par-1", "nis2/art-21/par-2"])
    assert "nist-csf/GV.SC-01" in {m["id"] for m in out["items"]}
    assert xw.linked_ids(["nis2/art-21/par-2"], "nist-csf/") == ["nist-csf/GV.SC-01"]


def test_committed_links_reach_iso_from_csf():
    xw.graph.cache_clear()
    out = xw.related("nist-csf/GV.SC-05", official_only=True)
    assert any(m["group"] == "NIST SP 800-53" for m in out["items"])
    assert any(m["group"] == "ISO/IEC 27001:2022" for m in out["items"])
```

- [ ] **Step 2:** FAIL. **Step 3: implement**

```python
"""Crosswalk lookups (spec §13.4): matching requirements in other frameworks, with source and path."""
import json
from functools import lru_cache
from typing import Dict, List

import yaml

from app.services.regs import DATA_DIR
from app.services.regs.ask import regs_corpus

GROUPS = {"eu-ai-act": "EU AI Act", "gdpr": "GDPR", "nis2": "NIS2", "nis2-ir": "NIS2 Implementing Regulation 2024/2690",
          "dora": "DORA", "cra": "Cyber Resilience Act", "nist-800-53": "NIST SP 800-53", "nist-csf": "NIST CSF 2.0",
          "nist-ai-rmf": "NIST AI RMF", "iso-27001": "ISO/IEC 27001:2022", "iso-42001-fdis": "ISO/IEC FDIS 42001"}
CAVEAT = "Mappings show related requirements, not equivalence."


@lru_cache(maxsize=1)
def sources() -> Dict[str, dict]:
    return {s["id"]: s for s in yaml.safe_load((DATA_DIR / "crosswalk_sources.yaml").read_text(encoding="utf-8"))}


def _links() -> List[dict]:
    out = []
    for name in ("crosswalks.jsonl", "crosswalks.community.jsonl"):
        path = DATA_DIR / name
        if path.exists():
            out += [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


@lru_cache(maxsize=1)
def graph() -> Dict[str, List[dict]]:
    g: Dict[str, List[dict]] = {}
    for l in _links():
        g.setdefault(l["from"], []).append({**l, "other": l["to"]})
        g.setdefault(l["to"], []).append({**l, "other": l["from"]})
    for edges in g.values():
        edges.sort(key=lambda e: (e["kind"] != "official", e["other"]))
    return g


def _describe(node: str) -> dict:
    corpus, _ = regs_corpus()
    prefix = node.split("/", 1)[0]
    group = GROUPS.get(prefix, prefix)
    p = corpus.get(node)
    if p:
        return {"group": group, "ref": p["ref"], "title": p.get("title") or "", "licensed": False}
    return {"group": group, "ref": f"{group} {node.split('/', 1)[1]}", "title": "", "licensed": True}


def _walk(starts: List[str], hops: int, official_only: bool) -> List[dict]:
    g, seen, found = graph(), set(starts), []
    frontier = [(s, [], "official") for s in starts]
    for depth in range(1, hops + 1):
        nxt = []
        for node, via, kind in frontier:
            for e in g.get(node, []):
                if (official_only and e["kind"] != "official") or e["other"] in seen:
                    continue
                seen.add(e["other"])
                path_kind = "community" if "community" in (kind, e["kind"]) else "official"
                found.append({"id": e["other"], "kind": path_kind, "source": e["source"], "hops": depth, "via": via,
                              "note": e.get("note")})
                if not e["other"].startswith("iso-"):  # licensed ids are leaves
                    nxt.append((e["other"], via + [_describe(e["other"])["ref"]], path_kind))
        frontier = nxt
    return found


def _shape(found: List[dict], per_group: int) -> dict:
    items, more, counts = [], {}, {}
    for f in sorted(found, key=lambda f: (f["kind"] != "official", f["hops"], f["id"])):
        d = _describe(f["id"])
        n = counts[d["group"]] = counts.get(d["group"], 0) + 1
        if n > per_group:
            more[d["group"]] = more.get(d["group"], 0) + 1
            continue
        items.append({**f, **d, "source_title": sources().get(f["source"], {}).get("title", f["source"])})
    return {"items": items, "more": more}


def related(pid: str, hops: int = 2, official_only: bool = False, per_group: int = 4) -> dict:
    return _shape(_walk([pid], hops, official_only), per_group)


def related_many(ids: List[str], hops: int = 2, official_only: bool = False, per_group: int = 4) -> dict:
    return _shape(_walk(list(ids), hops, official_only), per_group)


def linked_ids(ids: List[str], prefix: str, hops: int = 2) -> List[str]:
    return sorted({f["id"] for f in _walk(list(ids), hops, False) if f["id"].startswith(prefix)})
```

- [ ] **Step 4:** PASS. **Step 5:** commit `feat(regs): crosswalk lookups with official-first shortest paths`.

### Task 5: Matches on Ask sources and checklist items

**Files:** Modify `backend/app/services/regs/ask.py`, `backend/app/services/regs/checklist.py`, `backend/app/schemas/chat.py`, `backend/app/services/chat/streaming.py`, `backend/app/routers/chat.py`; Tests `tests/regs/test_ask.py`, `tests/regs/test_checklist.py`, `tests/regs/test_stream_regs.py`.

- [ ] **Step 1: failing tests**

```python
# tests/regs/test_ask.py
def test_retrieved_sources_carry_their_crosswalk(monkeypatch):
    from app.services.regs import crosswalk as xw
    monkeypatch.setattr(xw, "related", lambda pid, **kw: {"items": [{"id": "x", "group": "NIST CSF 2.0"}], "more": {}})

    class S:
        REGS_NAMESPACE, REGS_CANDIDATES, REGS_TOP_K, REGS_MAX_PER_FRAMEWORK, REGS_RECITAL_WEIGHT = "regs-v1", 30, 6, 3, 0.9
        REGS_CROSSWALK_MODE = "off"

    corpus = {"gdpr/art-33/par-1": {**GDPR33, "ref": "GDPR Art. 33(1)", "title": "", "text": "t", "framework": "GDPR", "source_url": "u"}}
    sources, _ = ask.retrieve_regs("q", None, date(2026, 10, 24), None, S,
                                   lambda **kw: [{"_id": "gdpr/art-33/par-1", "_score": 0.6}], corpus, REG)
    assert sources[0]["crosswalk"]["items"][0]["group"] == "NIST CSF 2.0"
```

```python
# tests/regs/test_checklist.py
def test_items_carry_crosswalk_and_practices_narrow_to_linked_csf():
    cl = build_checklist(PROFILES["eu_cloud_large"]["profile"], date(2026, 10, 24))
    art21 = next(i for i in cl["groups"]["now"] if i["article"] == "nis2/art-21")
    groups = {m["group"] for m in art21["crosswalk"]["items"]}
    assert "NIS2 Implementing Regulation 2024/2690" in groups
    csf = next(p for p in cl["practices"] if p["framework"] == "NIST CSF 2.0")
    assert csf["narrowed"] and all(i["id"].startswith("nist-csf/") and "-" in i["id"] for g in csf["groups"] for i in g["items"])
    gdpr32 = next(i for g in ("now", "coming") for i in cl["groups"][g] if i["article"] == "gdpr/art-32")
    assert gdpr32["crosswalk"]["items"] == []
```

`tests/regs/test_stream_regs.py`, in `test_regs_stream_emits_route_step_and_regs_fields`: `assert "crosswalk" in src`.

- [ ] **Step 2:** FAIL. **Step 3: implement.**
- `ask.retrieve_regs`: after diversification, `for s in chosen: s["crosswalk"] = crosswalk.related(s["doc_id"])` (import inside the function to avoid an import cycle: `from app.services.regs import crosswalk`).
- `checklist.build_checklist`: `item["crosswalk"] = crosswalk.related_many([p["id"] for p in paras], per_group=4)` for binding and depends items; `_practices(prof, catalog, index, binding_items)`: for the CSF practice, `linked = crosswalk.linked_ids([p["id"] for it in binding_items for p in it["paragraphs"]], "nist-csf/")`; when non-empty, groups are built from those ids (title = provision text), `narrowed: True`, reason "Linked to your NIS2 items by ENISA's mapping (official)"; otherwise the previous behaviour with `narrowed: False`. The AI RMF practice is unchanged (`narrowed: False`).
- `schemas/chat.py` `SourceHit` gains `crosswalk: Optional[Dict[str, Any]] = Field(None, description="Matching requirements in other frameworks (spec §13).")` and `via: Optional[Dict[str, str]] = Field(None, description="Set when the source was added through a crosswalk link.")`; `streaming._regs_source_fields` and `routers/chat._build_chat_response` pass both.

- [ ] **Step 4:** full suite green. **Step 5:** commit `feat(regs): crosswalk matches on Ask sources and checklist items`.

### Task 6: Crosswalk-aware retrieval (measured)

**Files:** Modify `backend/app/core/config.py`, `backend/app/services/regs/ask.py`, `backend/app/services/prompts/regs_prompt.py`, `eval/run_regs.py`; Tests `tests/regs/test_ask.py`, `tests/regs/test_regs_prompt.py`.

**Interfaces:** settings `REGS_CROSSWALK_MODE: str = "off"` (`off|boost|inject`), `REGS_CROSSWALK_BOOST: float = 1.15`, `REGS_CROSSWALK_INJECT: int = 3`; injected sources carry `via = {"ref": parent ref, "source": source title}`; `run_regs.py --mode diversified --crosswalk boost|inject`.

- [ ] **Step 1: failing tests**

```python
# tests/regs/test_ask.py
LINKED = {"items": [{"id": "nist-csf/GV.SC-01", "licensed": False, "kind": "official", "source_title": "NIST CSF 2.0 to SP 800-53"}], "more": {}}


def _settings(mode):
    class S:
        REGS_NAMESPACE, REGS_CANDIDATES, REGS_TOP_K, REGS_MAX_PER_FRAMEWORK, REGS_RECITAL_WEIGHT = "regs-v1", 30, 2, 3, 0.9
        REGS_CROSSWALK_MODE, REGS_CROSSWALK_BOOST, REGS_CROSSWALK_INJECT = mode, 1.15, 3
    return S


CORPUS = {"nist-800-53/sr-3": {"id": "nist-800-53/sr-3", "ref": "SP 800-53 SR-3", "title": "", "text": "a", "framework": "NIST SP 800-53", "binding": "voluntary_framework", "source_url": "u"},
          "nist-csf/GV.SC-01": {"id": "nist-csf/GV.SC-01", "ref": "CSF GV.SC-01", "title": "", "text": "b", "framework": "NIST CSF 2.0", "binding": "voluntary_framework", "source_url": "u"},
          "gdpr/art-28/par-1": {"id": "gdpr/art-28/par-1", "ref": "GDPR Art. 28(1)", "title": "", "text": "c", "framework": "GDPR", "binding": "regulation", "source_url": "u"}}
HITS3 = [{"_id": "nist-800-53/sr-3", "_score": 0.6}, {"_id": "gdpr/art-28/par-1", "_score": 0.55}, {"_id": "nist-csf/GV.SC-01", "_score": 0.5}]


def test_boost_lifts_pool_candidates_linked_to_the_top_hit(monkeypatch):
    from app.services.regs import crosswalk as xw
    monkeypatch.setattr(xw, "related", lambda pid, **kw: LINKED if pid == "nist-800-53/sr-3" else {"items": [], "more": {}})
    off, _ = ask.retrieve_regs("q", None, date(2026, 10, 24), None, _settings("off"), lambda **kw: HITS3, CORPUS, REG)
    boost, _ = ask.retrieve_regs("q", None, date(2026, 10, 24), None, _settings("boost"), lambda **kw: HITS3, CORPUS, REG)
    assert [s["doc_id"] for s in off] == ["nist-800-53/sr-3", "gdpr/art-28/par-1"]
    assert [s["doc_id"] for s in boost] == ["nist-800-53/sr-3", "nist-csf/GV.SC-01"]


def test_inject_adds_linked_provisions_marked_via_with_the_parents_score(monkeypatch):
    from app.services.regs import crosswalk as xw
    monkeypatch.setattr(xw, "related", lambda pid, **kw: LINKED if pid == "nist-800-53/sr-3" else {"items": [], "more": {}})
    hits = HITS3[:2]
    out, _ = ask.retrieve_regs("q", None, date(2026, 10, 24), None, _settings("inject"), lambda **kw: hits, CORPUS, REG)
    added = out[-1]
    assert added["doc_id"] == "nist-csf/GV.SC-01" and added["score"] == 0.6
    assert added["via"] == {"ref": "SP 800-53 SR-3", "source": "NIST CSF 2.0 to SP 800-53"}
```

```python
# tests/regs/test_regs_prompt.py
def test_via_sources_name_their_link_in_the_header():
    src = [{**SRC[0], "via": {"ref": "SP 800-53 SR-3", "source": "NIST CSF 2.0 to SP 800-53 Rev. 5.2.0"}}]
    user = build_regs_messages([], "q", src, date(2026, 10, 24))[-1].content
    assert "| linked to SP 800-53 SR-3 by NIST CSF 2.0 to SP 800-53 Rev. 5.2.0" in user
```

- [ ] **Step 2:** FAIL. **Step 3: implement.** `config.py` (after `REGS_RECITAL_WEIGHT`):

```python
    REGS_CROSSWALK_MODE: str = Field(default="off", description="off | boost | inject: use official crosswalk links in retrieval (kept only if eval/run_regs.py shows it helps).")
    REGS_CROSSWALK_BOOST: float = Field(default=1.15, description="boost: rank weight for pool candidates linked to the top hit.")
    REGS_CROSSWALK_INJECT: int = Field(default=3, description="inject: most linked provisions added.")
```

`ask.retrieve_regs`, replacing its final `return`:

```python
    mode = getattr(settings, "REGS_CROSSWALK_MODE", "off")
    chosen = diversify(cands, settings.REGS_TOP_K, settings.REGS_MAX_PER_FRAMEWORK)
    if chosen and mode == "boost":
        linked = {m["id"] for m in crosswalk.related(chosen[0]["doc_id"], official_only=True, per_group=99)["items"]}
        for c in cands:
            if c["doc_id"] in linked:
                c["rank_score"] *= settings.REGS_CROSSWALK_BOOST
        chosen = diversify(cands, settings.REGS_TOP_K, settings.REGS_MAX_PER_FRAMEWORK)
    elif chosen and mode == "inject":
        have, extra = {c["doc_id"] for c in chosen}, []
        for parent in chosen[:3]:
            for m in crosswalk.related(parent["doc_id"], hops=1, official_only=True, per_group=2)["items"]:
                if m["licensed"] or m["id"] in have or m["id"] not in corpus:
                    continue
                have.add(m["id"])
                extra.append({**_candidate(corpus[m["id"]], parent["score"], registry, as_of, condition, settings),
                              "via": {"ref": parent["ref"], "source": m["source_title"]}})
        chosen += extra[: settings.REGS_CROSSWALK_INJECT]
    for s in chosen:
        s["crosswalk"] = crosswalk.related(s["doc_id"])
    return chosen, top
```

with the candidate dict construction moved into `_candidate(p, score, registry, as_of, condition, settings) -> dict` (same fields as today) and used by the loop above it.

`regs_prompt.build_regs_context`: `header += f" | linked to {s['via']['ref']} by {s['via']['source']}" if s.get("via") else ""`.

`eval/run_regs.py`: `--crosswalk {off,boost,inject}` (default off) sets `REGS_CROSSWALK_MODE` on the copied settings for `diversified`; report file name `regs_diversified-<mode>_<ts>.json`.

- [ ] **Step 4:** PASS; full suite green. **Step 5:** commit `feat(regs): optional crosswalk boost/inject in regs retrieval (off by default)`.

### Task 7: UI

**Files:** Modify `frontend/services/presenters.py`, `frontend/views/ask.py`, `frontend/views/checklist.py`; Tests `tests/test_ui_crosswalk.py`.

- [ ] **Step 1: failing tests**

```python
from frontend.services.presenters import chip_label, crosswalk_lines

XW = {"items": [
    {"id": "nist-csf/GV.SC-01", "group": "NIST CSF 2.0", "ref": "CSF GV.SC-01", "title": "", "licensed": False, "kind": "official",
     "source_title": "ENISA NIS2 Technical Implementation Guidance, mapping table v1.2", "hops": 2, "via": ["NIS2 IR Annex, point 5.1"], "note": None},
    {"id": "iso-27001/A.5.21", "group": "ISO/IEC 27001:2022", "ref": "ISO/IEC 27001:2022 A.5.21", "title": "", "licensed": True,
     "kind": "community", "source_title": "NIST CSF 2.0 to ISO/IEC 27001:2022 Annex A (Razilio)", "hops": 1, "via": [], "note": None}],
    "more": {"NIST CSF 2.0": 3}}


def test_crosswalk_lines_group_label_and_caveats():
    groups = crosswalk_lines(XW)
    assert [g for g, _, _ in groups] == ["NIST CSF 2.0", "ISO/IEC 27001:2022"]
    csf_lines, csf_more = groups[0][1], groups[0][2]
    assert csf_lines[0][0] == "CSF GV.SC-01" and csf_lines[0][1] == "green"
    assert "via NIS2 IR Annex, point 5.1" in csf_lines[0][2] and csf_more == 3
    iso = groups[1][1][0]
    assert iso[1] == "gray" and "community mapping, not endorsed by NIST" in iso[2] and "licensed text, bring your copy" in iso[2]


def test_via_sources_are_marked_on_chips():
    src = {"ref": "CSF GV.SC-01", "status": "voluntary", "applies_from": None, "via": {"ref": "SP 800-53 SR-3", "source": "s"}}
    assert chip_label(3, src) == "3 · CSF GV.SC-01 · via SP 800-53 SR-3"
```

- [ ] **Step 2:** FAIL. **Step 3: implement.** `presenters.py`:

```python
CROSSWALK_CAVEAT = ("Matches come from published mappings and show related requirements, not equivalence. "
                    "Community mappings are not endorsed by NIST.")


def crosswalk_lines(xw: Optional[Dict[str, Any]]) -> List[Tuple[str, List[Tuple[str, str, str]], int]]:
    """[(group, [(ref, badge color, caption)], hidden count)] in first-seen group order."""
    groups: Dict[str, List[Tuple[str, str, str]]] = {}
    for m in (xw or {}).get("items") or []:
        notes = [m["source_title"]]
        if m.get("via"):
            notes.append("via " + " → ".join(m["via"]))
        if m.get("note"):
            notes.append(m["note"])
        if m["kind"] == "community":
            notes.append("community mapping, not endorsed by NIST")
        if m.get("licensed"):
            notes.append("licensed text, bring your copy")
        label = m["ref"] + (f" · {m['title']}" if m.get("title") else "")
        groups.setdefault(m["group"], []).append((label, "green" if m["kind"] == "official" else "gray", " · ".join(notes)))
    more = (xw or {}).get("more") or {}
    return [(g, lines, more.get(g, 0)) for g, lines in groups.items()]
```

and in `chip_label`, first line: `if src.get("via"): return f"{n} · {src['ref']} · via {src['via']['ref']}"`.

`views/ask.py` `evidence_drawer`: after the passage block,

```python
    groups = crosswalk_lines(src.get("crosswalk"))
    if groups:
        st.markdown("**Matching requirements**")
        for group, lines, hidden in groups:
            st.caption(group)
            for label, color, note in lines:
                st.badge(label, color=color)
                st.caption(note)
            if hidden:
                st.caption(f"and {hidden} more")
        st.caption(CROSSWALK_CAVEAT)
    elif src.get("ref"):
        st.caption("No published crosswalk for this provision.")
```

`views/checklist.py` `item_card`: after the note caption,

```python
        groups = crosswalk_lines(item.get("crosswalk"))
        if groups:
            with st.expander(f"Matches in other frameworks · {sum(len(l) + h for _, l, h in groups)}"):
                for group, lines, hidden in groups:
                    st.markdown(f"**{group}**")
                    for label, color, note in lines:
                        st.markdown(f":{color}-badge[{label}] {note}")
                    if hidden:
                        st.caption(f"and {hidden} more")
                st.caption(CROSSWALK_CAVEAT)
```

and the practices expander shows "Linked to your NIS2 items" when `pr.get("narrowed")`.

- [ ] **Step 4:** UI tests + smoke green. **Step 5:** commit `feat(ui): matching requirements in the evidence drawer and checklist cards`.

### Task 8: Upsert, evals, live check, docs, merge

- [ ] `corpus/upsert.py` gains `--only PREFIX` (filters provisions by id prefix before estimating/upserting); test in `tests/corpus/test_upsert.py`:

```python
def test_only_prefix_selects_new_records():
    from corpus.upsert import select
    provs = [{"id": "nis2-ir/art-1"}, {"id": "nis2/art-1"}, {"id": "nis2-ir/anx-1/pt-1.1"}]
    assert [p["id"] for p in select(provs, "nis2-ir/")] == ["nis2-ir/art-1", "nis2-ir/anx-1/pt-1.1"]
    assert select(provs, None) == provs
```

  with `def select(provisions, prefix): return [p for p in provisions if not prefix or p["id"].startswith(prefix)]`.
- [ ] Dry run `python -m corpus.upsert --only nis2-ir/` → show records, tokens, storage; **ask the user**; on approval `--yes`.
- [ ] Golden: add four rows to `eval/regs_golden.jsonl`, labels chosen by reading the annex point titles printed from the corpus (never from retriever output): supply-chain security policy (point titled "Supply chain security policy"), business continuity and disaster recovery plan (point titled "Business continuity and disaster recovery plan"), significant incidents for cloud providers (`nis2-ir/art-7`), and multi-factor authentication (the access-control point whose title names authentication). Update `test_every_golden_label_exists_in_the_corpus` expectations only if it counts rows.
- [ ] Evals: `run_regs.py --mode diversified --k 6|10` with `--crosswalk off|boost|inject`; `--mode abstain`; record by type; set `REGS_CROSSWALK_MODE` default to the winner only if it passes the gate, else keep `off`.
- [ ] Live: local API + UI: Ask "What must a cloud provider do about supply chain security under NIS2?" (expect NIS2 IR Annex 5.1 with CSF/ISO matches), and a cloud-provider checklist (NIS2 Art. 21 card shows 2024/2690 and CSF matches; narrowed CSF practices).
- [ ] Spec §14 results, CHANGELOG, README (crosswalks paragraph + test count), memory; commit; `git merge --no-ff` into `revive/2026-10`; rerun suite and smoke on the merge; no push, keep the branch.
