"""Compliance path through the compiled graph (regs namespace); legacy namespaces unchanged."""
import json
from unittest.mock import MagicMock, patch

from app.services.chat import graph

REGS = {"query": "When must a personal data breach be notified?", "namespace": "regs-v1", "top_k": 5,
        "use_web_fallback": True, "min_score": 0.25, "max_web_results": 5, "chat_history": [], "as_of": "2026-10-20"}
HITS = [{"_id": "gdpr/art-33/par-1", "_score": 0.62}, {"_id": "eu-ai-act/rct-1", "_score": 0.05}]
ROUTE = '{"intent": "%s", "profile_updates": []}'


def _route(intent, *facts):
    return json.dumps({"intent": intent, "profile_updates": [
        {"fact": f, "values": v, "evidence": e} for f, v, e in facts]})


def _llm(*contents):
    llm = MagicMock()
    llm.invoke.side_effect = [MagicMock(content=c, usage_metadata=None, response_metadata={}) for c in contents]
    return llm


def _run(state, llm, hits=HITS):
    graph._graph = None
    try:
        with patch.object(graph, "get_llm", return_value=llm), \
                patch.object(graph, "pinecone_search", return_value=hits) as search:
            return graph.get_chat_graph().invoke(dict(state)), search
    finally:
        graph._graph = None


def test_regs_answer_cites_provisions_with_status_and_never_uses_web():
    llm = _llm(ROUTE % "ask", "Within 72 hours [1].")
    with patch.object(graph, "is_tavily_configured", return_value=True):
        out, search = _run(REGS, llm)
    assert [s["ref"] for s in out["retrieved"]] == ["GDPR Art. 33(1)"]  # recital is below the 0.30 floor
    assert out["retrieved"][0]["status"] == "applies" and out["web_fallback_used"] is False
    assert out["answer"] == "Within 72 hours [1]." and out["intent"] == "ask" and not out["insufficient_context"]
    assert search.call_args.kwargs["namespace"] == "regs-v1"
    prompt = llm.invoke.call_args_list[1].args[0][-1].content
    assert "status on 2026-10-20: in force since 2018-05-25" in prompt


def test_out_of_scope_question_abstains_before_any_llm_or_search_call():
    llm = _llm()
    out, search = _run({**REGS, "query": "What does HIPAA require?"}, llm)
    assert out["insufficient_context"] and out["abstain_reason"] == "out_of_scope" and "HIPAA" in out["answer"]
    llm.invoke.assert_not_called()
    search.assert_not_called()


def test_checklist_intent_builds_the_checklist_without_retrieval_or_generation():
    llm = _llm(_route("checklist", ("ai_roles", ["deployer"], "We deploy"), ("high_risk", ["annex_iii"], "CV screener")))
    out, search = _run({**REGS, "query": "We deploy a CV screener in the EU. What do we need to do?",
                        "profile": {"eu_established": True}}, llm)
    assert out["intent"] == "checklist" and not out["insufficient_context"] and out.get("abstain_reason") is None
    arts = {i["article"] for i in out["checklist"]["groups"]["coming"]}
    assert "eu-ai-act/art-26" in arts and out["answer"].startswith("For this profile on 2026-10-20")
    assert llm.invoke.call_count == 1
    search.assert_not_called()


def test_fact_only_follow_up_after_a_checklist_rebuilds_it():
    route = _route("ask", ("ai_roles", ["provider"], "We are also the provider"))
    out, _ = _run({**REGS, "query": "We are also the provider.", "previous_intent": "checklist",
                   "profile": {"ai_roles": ["deployer"], "high_risk": "annex_iii", "eu_established": True}}, _llm(route))
    # List facts accumulate: the message adds a role, it does not replace the ones already known.
    assert out["intent"] == "checklist" and out["profile"]["ai_roles"] == ["deployer", "provider"]
    assert "eu-ai-act/art-16" in {i["article"] for i in out["checklist"]["groups"]["coming"]}


def test_router_profile_updates_drive_the_high_risk_condition():
    route = _route("ask", ("high_risk", ["annex_iii"], "Annex III hiring tool"))
    out, _ = _run({**REGS, "query": "We run an Annex III hiring tool. Must deployers keep logs?", "as_of": "2028-01-01"},
                  _llm(route, "Yes [1]."),
                  hits=[{"_id": "eu-ai-act/art-26/par-6", "_score": 0.7}])
    assert out["profile_updates"] == {"high_risk": "annex_iii"}
    assert out["retrieved"][0]["status"] == "applies" and out["retrieved"][0]["applies_from"] == "2027-12-02"


def test_nothing_above_the_regs_floor_abstains_with_regs_message():
    out, _ = _run({**REGS, "query": "Who won the 2018 FIFA World Cup?"}, _llm(ROUTE % "ask"),
                  hits=[{"_id": "gdpr/art-33/par-1", "_score": 0.1}])
    assert out["insufficient_context"] and out["answer"] == graph.REGS_ABSTENTION and out["abstain_reason"] == "no_match"


def test_named_framework_missing_from_sources_adds_a_warning():
    out, _ = _run({**REGS, "query": "What does DORA say about breach notification?"}, _llm(ROUTE % "ask", "x [1]."))
    assert any("DORA" in w for w in out["warnings"])


def test_legacy_namespace_path_unchanged():
    hits = [{"_id": "x:0", "_score": 0.7, "fields": {"chunk_text": "t", "doc_id": "x", "title": "T", "source": "wiki"}}]
    llm = _llm("legacy answer [1].")
    out, _ = _run({**REGS, "namespace": "eval"}, llm, hits=hits)
    assert out.get("intent") is None and out["answer"] == "legacy answer [1]." and llm.invoke.call_count == 1
    assert out["retrieved"][0]["source"] == "wiki"


def test_answer_citations_are_normalised_before_verification():
    out, _ = _run(REGS, _llm(ROUTE % "ask", "Within 72 hours【1】 or【9】."))
    assert out["answer"] == "Within 72 hours[1] or[9]." and out["unverified_citations"] == [9]


def test_your_copy_answers_licensed_questions_without_any_dense_hit():
    copy = [{"name": "iso-27001", "clauses": {"A.5.19": {"title": "Supplier relationships",
                                                        "text": "Processes shall be defined to manage supplier risks."}}}]
    llm = _llm(ROUTE % "ask", "Define supplier processes [1].")
    out, _ = _run({**REGS, "query": "What does ISO/IEC 27001 A.5.19 require?", "standards": copy}, llm, hits=[])
    assert not out["insufficient_context"] and out["retrieved"][0]["ref"] == "ISO/IEC 27001:2022 A.5.19"
    assert "licensed standard text you supplied" in llm.invoke.call_args_list[1].args[0][-1].content


def test_gap_check_requests_point_to_the_page():
    out, _ = _run({**REGS, "query": "Check our incident policy against NIS2"}, _llm(ROUTE % "gap_check"))
    assert out["intent"] == "gap_check" and "Gap check page" in out["answer"]
