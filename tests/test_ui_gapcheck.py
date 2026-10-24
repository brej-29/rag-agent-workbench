import csv
import io

from frontend.services.backend_client import iter_sse
from frontend.services.presenters import copy_clause, crosswalk_lines, gap_csv, gap_markdown, status_label, verdict_label

RESULTS = [{"selection": "gdpr/art-33", "ref": "GDPR Art. 33", "title": "Notification", "counts": {}, "tokens": 0,
            "duties": [{"provision": "gdpr/art-33/par-1", "ref": "GDPR Art. 33(1)", "verdict": "missing", "claimed": "missing",
                        "law_quote": "notify the supervisory authority", "policy_quote": "", "passage_id": "", "reason": "Not addressed.", "problem": ""},
                       {"provision": "gdpr/art-33/par-2", "ref": "GDPR Art. 33(2)", "verdict": "unverified", "claimed": "covered",
                        "law_quote": "x", "policy_quote": "y", "passage_id": "p1", "reason": "r", "problem": "policy quote not found in the cited passage"}]}]
STANDARDS = {"iso-27001": {"name": "ISO/IEC 27001:2022", "clauses": {"A.5.19": {"title": "Supplier relationships", "text": "t"}}}}


def test_verdict_labels_and_licensed_status():
    assert verdict_label("covered") == ("green", "covered") and verdict_label("unverified") == ("gray", "unverified, review")
    assert status_label({"status": "licensed_copy"}) == ("blue", "your copy")


def test_gap_exports():
    md = gap_markdown(RESULTS, {"covered": 0, "partial": 0, "missing": 1, "unverified": 1}, "Breach procedure")
    assert "# Gap check: Breach procedure" in md and "not legal advice" in md
    assert "- **GDPR Art. 33(1)** missing: Not addressed. (law: “notify the supervisory authority”)" in md
    assert "unverified, review (claimed covered: policy quote not found in the cited passage)" in md
    rows = list(csv.DictReader(io.StringIO(gap_csv(RESULTS))))
    assert [r["verdict"] for r in rows] == ["missing", "unverified"] and rows[1]["claimed"] == "covered"


def test_copy_clause_lookup_and_crosswalk_caption():
    assert copy_clause("iso-27001/A.5.19", STANDARDS)["title"] == "Supplier relationships"
    assert copy_clause("iso-42001-fdis/B.2.2", STANDARDS) is None and copy_clause("nist-csf/GV.SC-01", STANDARDS) is None
    xw = {"items": [{"id": "iso-27001/A.5.19", "group": "ISO/IEC 27001:2022", "ref": "ISO/IEC 27001:2022 A.5.19", "title": "",
                     "licensed": True, "kind": "official", "source_title": "S", "via": [], "note": None}], "more": {}}
    (_, lines, _), = crosswalk_lines(xw, STANDARDS)
    assert "from your copy: Supplier relationships" in lines[0][2] and "bring your copy" not in lines[0][2]


def test_iter_sse_yields_every_event():
    lines = ["event: plan", 'data: {"est_tokens": 9}', "", "event: article", 'data: {"ref": "GDPR Art. 33"}', ""]
    assert list(iter_sse(lines)) == [("plan", {"est_tokens": 9}), ("article", {"ref": "GDPR Art. 33"})]
