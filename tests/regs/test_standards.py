from app.services.regs.standards import copy_sources, coverage, detect_standard, parse_clauses

# Synthetic copy: real clause ids, invented wording. No ISO text enters the repository.
SYNTH = """ISO/IEC 27001:2022 (synthetic test copy)
Contents
1 Scope ........ 1
4.1 Understanding the context ........ 2
Annex A (normative) Controls ........ 9
Bibliography ........ 20
1 Scope
This synthetic document exists only for tests.
4 Context of the organization
4.1 Understanding the context
The organization shall determine issues relevant to its purpose.
6.1.2 Risk assessment
The organization shall define a risk assessment process.
Annex A
(normative)
5.19 Supplier relationships
Processes shall be defined to manage supplier risks.
8.5 Secure sign-in
Secure sign-in shall be implemented, for example with multiple factors.
"""


def test_detects_the_standard_named_first():
    assert detect_standard(SYNTH) == "iso-27001"
    assert detect_standard("ISO/IEC 42001 ... refers to ISO/IEC 27001") == "iso-42001"
    assert detect_standard("a plain policy") is None


def test_parses_clauses_past_a_table_of_contents():
    clauses = parse_clauses(SYNTH, "iso-27001")
    assert list(clauses) == ["4.1", "6.1.2", "A.5.19", "A.8.5"]
    assert clauses["A.5.19"] == {"title": "Supplier relationships", "text": "Processes shall be defined to manage supplier risks."}
    assert coverage(clauses, "iso-27001") == {"clauses": 4, "annex_a": 2, "annex_a_expected": 93}


def test_copy_sources_prefer_named_clauses_and_are_marked_private():
    std = [{"name": "iso-27001", "clauses": parse_clauses(SYNTH, "iso-27001")}]
    out = copy_sources("What does Annex A 5.19 say about suppliers?", std)
    assert out[0]["ref"] == "ISO/IEC 27001:2022 A.5.19" and out[0]["private"] and out[0]["status"] == "licensed_copy"
    assert out[0]["framework"] == "ISO/IEC 27001:2022 (your copy)"
    assert copy_sources("anything", None) == []
