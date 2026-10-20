"""/chat/stream on the compliance path: route step, deterministic replies, regs fields in `done`."""
from unittest.mock import MagicMock, patch

from tests.test_streaming import _collect, _parse
from app.services.chat import graph
from app.services.chat.streaming import stream_chat_response

STATE = {"query": "When must a personal data breach be notified?", "namespace": "regs-v1", "top_k": 5,
         "use_web_fallback": True, "min_score": 0.25, "max_web_results": 5, "chat_history": [], "as_of": "2026-10-20"}
ROUTE = ('{"intent": "ask", "profile_updates": {"roles": ["controller"], "ai_system": null, "gpai_model": null, '
         '"high_risk": null, "personal_data": true, "eu_market": null, "sector": null}}')


async def _astream(messages, config=None):  # noqa: ARG001
    yield MagicMock(content="Within 72 hours [1].", usage_metadata=None, response_metadata={})


def _run(state, hits):
    router_llm = MagicMock()
    router_llm.invoke.return_value = MagicMock(content=ROUTE, usage_metadata=None, response_metadata={})
    gen_llm = MagicMock()
    gen_llm.astream = _astream
    with patch.object(graph, "get_llm", return_value=router_llm), \
            patch.object(graph, "pinecone_search", return_value=hits), \
            patch("app.services.chat.streaming.get_llm", return_value=gen_llm), \
            patch("app.services.chat.streaming._record_metrics"):
        events = [_parse(f) for f in _collect(stream_chat_response(dict(state), {}))]
    return events, router_llm


def test_regs_stream_emits_route_step_and_regs_fields():
    events, _ = _run(STATE, [{"_id": "gdpr/art-33/par-1", "_score": 0.62}])
    steps = [d["id"] for e, d in events if e == "step"]
    assert steps[:3] == ["route", "retrieve", "gate"]
    gate = next(d for e, d in events if e == "step" and d["id"] == "gate")
    assert gate["detail"]["floor"] == 0.30
    done = events[-1][1]
    assert done["as_of"] == "2026-10-20" and done["intent"] == "ask"
    assert done["profile_updates"] == {"roles": ["controller"], "personal_data": True}
    src = done["sources"][0]
    assert (src["ref"], src["framework"], src["status"], src["applies_from"]) == (
        "GDPR Art. 33(1)", "GDPR", "applies", "2018-05-25")


def test_scope_guard_reply_streams_without_retrieval_or_generation():
    events, router_llm = _run({**STATE, "query": "What does SOC 2 require for availability?"}, [])
    assert [(d["id"], d["status"]) for e, d in events if e == "step"] == [("route", "done"), ("generate", "skipped")]
    route = next(d for e, d in events if e == "step" and d["id"] == "route")
    assert route["detail"]["abstain_reason"] == "licensed"
    assert "licensed standard" in next(d["text"] for e, d in events if e == "token")
    assert events[-1][1]["abstain_reason"] == "licensed" and events[-1][1]["insufficient_context"] is True
    router_llm.invoke.assert_not_called()


def test_regs_request_fields_reach_the_graph_and_the_cache_key():
    from datetime import date

    from app.routers.chat import _cache_query, _regs_inputs
    from app.schemas.chat import ChatRequest

    req = ChatRequest(query="q", as_of="2027-01-01", frameworks=["GDPR"], profile={"high_risk": "annex_iii"})
    assert _regs_inputs(req) == {"as_of": "2027-01-01", "frameworks": ["GDPR"],
                                 "profile": {"roles": [], "high_risk": "annex_iii"}}
    assert _cache_query(req, "eval") == "q"
    later = ChatRequest(query="q", as_of="2028-01-01", frameworks=["GDPR"], profile={"high_risk": "annex_iii"})
    assert _cache_query(req, "regs-v1") != _cache_query(later, "regs-v1")
    assert date.today().isoformat() in _cache_query(ChatRequest(query="q"), "regs-v1")
