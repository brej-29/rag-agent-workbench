"""Compliance path through the compiled graph (regs namespace); legacy namespaces unchanged."""
from unittest.mock import MagicMock, patch

from app.services.chat import graph

REGS = {"query": "When must a personal data breach be notified?", "namespace": "regs-v1", "top_k": 5,
        "use_web_fallback": True, "min_score": 0.25, "max_web_results": 5, "chat_history": [], "as_of": "2026-10-20"}
HITS = [{"_id": "gdpr/art-33/par-1", "_score": 0.62}, {"_id": "eu-ai-act/rct-1", "_score": 0.05}]
ROUTE = ('{"intent": "%s", "profile_updates": {"roles": [], "ai_system": null, "gpai_model": null, '
         '"high_risk": null, "personal_data": null, "eu_market": null, "sector": null}}')


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


def test_checklist_intent_is_routed_and_explained():
    out, search = _run({**REGS, "query": "Which rules apply to us?"}, _llm(ROUTE % "checklist"))
    assert out["intent"] == "checklist" and out["abstain_reason"] == "not_available"
    assert out["answer"] == graph.NOT_YET["checklist"] and out["insufficient_context"]
    search.assert_not_called()


def test_router_profile_updates_drive_the_high_risk_condition():
    route = ROUTE.replace('"high_risk": null', '"high_risk": "annex_iii"') % "ask"
    out, _ = _run({**REGS, "query": "Must deployers keep logs?", "as_of": "2028-01-01"}, _llm(route, "Yes [1]."),
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
