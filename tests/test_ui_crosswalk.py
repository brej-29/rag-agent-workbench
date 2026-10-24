from frontend.services.presenters import chip_label, crosswalk_compact, crosswalk_lines

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


def test_crosswalk_compact_puts_three_refs_on_one_line_and_keeps_the_rest():
    xw = {"items": [{"group": "NIST CSF 2.0", "ref": f"GV.SC-0{i}", "title": "t", "kind": "official",
                     "source_title": "CSF to 800-53", "licensed": False, "id": f"nist-csf/GV.SC-0{i}"}
                    for i in range(1, 5)], "more": {"NIST CSF 2.0": 2}}
    [g] = crosswalk_compact(xw)
    assert g["line"] == ":green-badge[GV.SC-01] :green-badge[GV.SC-02] :green-badge[GV.SC-03]"
    assert g["notes"] == "CSF to 800-53" and [label for label, _, _ in g["rest"]] == ["GV.SC-04 · t"]
    assert g["hidden"] == 2
