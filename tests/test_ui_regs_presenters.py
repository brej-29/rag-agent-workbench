"""Compliance-path wording in the UI: provision status chips, route trace line, abstention verdicts."""
from frontend.services.presenters import chip_label, status_label, step_line, trace_summary, verdict

GDPR = {"ref": "GDPR Art. 33(1)", "framework": "GDPR", "status": "applies", "applies_from": "2018-05-25", "score": 0.62}


def test_status_label_per_status():
    assert status_label(GDPR) == ("green", "in force since 2018-05-25")
    assert status_label({**GDPR, "status": "not_yet", "applies_from": "2027-12-02"}) == ("orange", "applies from 2027-12-02")
    assert status_label({**GDPR, "status": "depends"})[0] == "violet"
    assert status_label({**GDPR, "status": "voluntary"}) == ("gray", "voluntary framework")
    assert status_label({**GDPR, "status": "no_date"}) == ("gray", "no date of its own")
    assert status_label({"title": "legacy", "score": 0.5}) is None


def test_chip_label_uses_reference_and_status_on_regs_sources():
    assert chip_label(1, GDPR) == "1 · GDPR Art. 33(1) · in force since 2018-05-25"
    title = "A very long legacy document title that goes on"
    assert chip_label(2, {"title": title, "score": 0.512}) == f"2 · {title[:42]} · 0.51"


def test_route_step_lines():
    assert "not cover" in step_line({"id": "route", "status": "done", "detail": {"abstain_reason": "out_of_scope"}})
    assert "licensed" in step_line({"id": "route", "status": "done", "detail": {"abstain_reason": "licensed"}})
    line = step_line({"id": "route", "status": "done",
                      "detail": {"intent": "ask", "profile_updates": {"ai_roles": ["deployer"], "high_risk": "annex_iii"}}})
    assert "question about the rules" in line and "deployer" in line and "Annex III" in line
    assert "not available yet" in step_line({"id": "route", "status": "done",
                                             "detail": {"intent": "checklist", "abstain_reason": "not_available"}})


def test_verdicts_and_summaries_for_regs_abstentions():
    assert verdict({"insufficient_context": True, "abstain_reason": "out_of_scope"}) == ("gray", "Not covered here")
    assert verdict({"insufficient_context": True, "abstain_reason": "licensed"}) == ("gray", "Licensed standard")
    assert verdict({"insufficient_context": True, "abstain_reason": "not_available"}) == ("blue", "Not available yet")
    assert verdict({"insufficient_context": True})[1] == "Not in the knowledge base"
    assert "outside the covered rules" in trace_summary(True, 1200, 0, "out_of_scope")
    assert "nothing relevant" in trace_summary(True, 1200, 0)


def test_dates_note_names_the_high_risk_classes():
    from frontend.services.presenters import dates_note

    assert dates_note("2027-12-02 if annex_iii; 2028-08-02 if annex_i") == \
        "2027-12-02 for Annex III systems; 2028-08-02 for Annex I systems"
