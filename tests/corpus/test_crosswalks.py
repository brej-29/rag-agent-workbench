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
    assert report["nis2-ir-headings"]["links"] == 53
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
