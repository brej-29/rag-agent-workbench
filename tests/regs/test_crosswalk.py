import pytest

from app.services.regs import crosswalk as xw


@pytest.fixture(autouse=True)
def _fresh_graph():
    """The graph is cached; never let a test's fake links leak into later tests."""
    yield
    xw.graph.cache_clear()

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


def test_no_siblings_from_the_starting_instrument(monkeypatch):
    _use(monkeypatch, LINKS + [{"from": "nis2-ir/anx-1/pt-1.1", "to": "nis2/art-21/par-2", "source": "nis2-ir-headings", "kind": "official"}])
    ids = {m["id"] for m in xw.related("nis2-ir/anx-1/pt-5.1")["items"]}
    assert "nis2/art-21/par-2" in ids and "nis2-ir/anx-1/pt-1.1" not in ids  # same act, reached through the hub


def test_two_hop_matches_reached_by_more_paths_rank_first(monkeypatch):
    links = [{"from": "nis2-ir/anx-1/pt-5.1", "to": f"nist-csf/{c}", "source": "enisa-nis2", "kind": "official"}
             for c in ("GV.OC-03", "GV.SC-01", "GV.SC-04")]
    links += [{"from": "nist-csf/GV.OC-03", "to": "nist-800-53/ac-1", "source": "nist-csf-800-53", "kind": "official"},
              {"from": "nist-csf/GV.SC-01", "to": "nist-800-53/sr-3", "source": "nist-csf-800-53", "kind": "official"},
              {"from": "nist-csf/GV.SC-04", "to": "nist-800-53/sr-3", "source": "nist-csf-800-53", "kind": "official"}]
    _use(monkeypatch, links)
    sp = [m["id"] for m in xw.related("nis2-ir/anx-1/pt-5.1")["items"] if m["group"] == "NIST SP 800-53"]
    assert sp == ["nist-800-53/sr-3", "nist-800-53/ac-1"]
