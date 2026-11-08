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


def test_csf_category_labels_with_names_cite_by_id():
    cat = {"groups": [{"id": "GV", "title": "GOVERN", "controls": [
        {"id": "GV.OC", "class": "category", "title": "Organizational Context",
         "props": [{"name": "label", "value": "Organizational Context (GV.OC)"}],
         "parts": [{"id": "GV.OC_statement", "name": "statement", "prose": "Context is understood"}]}]}]}
    rec = parse_oscal(cat, CSF)[0]
    assert rec["ref"] == "CSF GV.OC" and rec["path"] == ["GOVERN", "Organizational Context (GV.OC)"]
