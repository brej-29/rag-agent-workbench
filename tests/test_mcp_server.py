"""MCP server: the workbench exposed to any MCP client with its guardrails intact.
In-memory client (mcp SDK v2); the graph and Pinecone are faked."""
import json
from datetime import date
from unittest.mock import MagicMock, patch

import anyio
from mcp.client.client import Client

from app import mcp_server
from app.schemas.chat import Profile
from app.services.regs import crosswalk as xw
from app.services.regs.ask import regs_corpus
from app.services.regs import gapcheck
from app.services.regs.checklist import build_checklist, load_catalog
from tests.regs.test_gapcheck import POLICY, _llm
from tests.regs.test_standards import SYNTH

ANSWERED = {
    "answer": "A vector database stores embeddings [1].", "insufficient_context": False,
    "retrieved": [{"doc_id": "2b529e6a", "title": "Vector database", "url": "https://w/vdb",
                   "source": "wiki", "score": 0.62, "chunk_text": "A vector database stores..."}],
    "web_results": [], "grounded": True, "faithfulness_score": 1.0,
}
ABSTAINED = {"answer": "I was unable to find sufficient information...", "insufficient_context": True,
             "retrieved": [], "web_results": []}


def _call(tool, args, state=None, hits=None):
    graph = MagicMock()
    graph.invoke.return_value = state

    async def go():
        async with Client(mcp_server.mcp) as client:
            return await client.call_tool(tool, args)

    with patch.object(mcp_server, "get_chat_graph", return_value=graph), \
         patch.object(mcp_server, "pinecone_search", return_value=hits or []):
        return anyio.run(go), graph


def test_tools_are_listed_with_output_schemas():
    async def go():
        async with Client(mcp_server.mcp) as client:
            return await client.list_tools()

    tools = {t.name: t for t in anyio.run(go).tools}
    assert {"ask", "search_knowledge_base"} <= set(tools)
    assert tools["ask"].output_schema["properties"]["abstained"]


def test_ask_returns_grounded_answer_with_numbered_sources():
    result, graph = _call("ask", {"question": "What is a vector database?"}, state=ANSWERED)
    out = result.structured_content
    assert out["abstained"] is False and out["grounded"] is True
    assert out["sources"][0] == {"n": 1, "title": "Vector database", "url": "https://w/vdb", "doc_id": "2b529e6a",
                                 "score": 0.62, "ref": "", "status": None, "applies_from": None}
    assert out["dangling_citations"] == []
    sent = graph.invoke.call_args.args[0]
    assert sent["query"] == "What is a vector database?" and sent["use_web_fallback"] is False


def test_ask_surfaces_abstention_explicitly():
    result, _ = _call("ask", {"question": "Who won the 2018 World Cup?"}, state=ABSTAINED)
    out = result.structured_content
    assert out["abstained"] is True and out["sources"] == []


def test_search_returns_scored_hits_without_llm():
    hit = {"_id": "abc:0", "_score": 0.5, "fields": {"chunk_text": "t", "doc_id": "abc", "title": "T", "source": "wiki", "url": ""}}
    result, graph = _call("search_knowledge_base", {"query": "q", "top_k": 3}, hits=[hit])
    out = result.structured_content
    assert out["hits"][0]["doc_id"] == "abc" and out["hits"][0]["score"] == 0.5
    graph.invoke.assert_not_called()


def test_guardrails_resource_explains_how_to_read_results():
    async def go():
        async with Client(mcp_server.mcp) as client:
            return await client.read_resource("rag://guardrails")

    text = anyio.run(go).contents[0].text
    assert "abstained" in text and "never" in text.lower()


REGS = {"answer": "Within 72 hours [1].", "insufficient_context": False, "as_of": "2026-10-20", "warnings": [],
        "retrieved": [{"doc_id": "gdpr/art-33/par-1", "title": "GDPR Art. 33(1) — Notification", "url": "u",
                       "source": "GDPR", "score": 0.62, "chunk_text": "...", "ref": "GDPR Art. 33(1)",
                       "status": "applies", "applies_from": "2018-05-25"}], "web_results": []}


def test_ask_defaults_to_the_regulatory_corpus_with_provision_status():
    result, graph = _call("ask", {"question": "When must a breach be notified?", "as_of": "2026-10-20",
                                  "frameworks": ["GDPR"]}, state=REGS)
    sent = graph.invoke.call_args.args[0]
    assert sent["namespace"] == "regs-v1" and sent["as_of"] == "2026-10-20" and sent["frameworks"] == ["GDPR"]
    out = result.structured_content
    src = out["sources"][0]
    assert (src["ref"], src["status"], src["applies_from"]) == ("GDPR Art. 33(1)", "applies", "2018-05-25")
    assert out["as_of"] == "2026-10-20"


def test_ask_rejects_a_malformed_date():
    result, graph = _call("ask", {"question": "q", "as_of": "20 Oct"}, state=REGS)
    assert result.is_error and "YYYY-MM-DD" in result.content[0].text
    graph.invoke.assert_not_called()


def _session(fn):
    async def go():
        async with Client(mcp_server.mcp) as client:
            return await fn(client)
    return anyio.run(go)


PROFILE = {"ai_system": True, "ai_roles": ["deployer"], "eu_established": True, "high_risk": "annex_iii"}


def test_checklist_tool_matches_the_api_builder():
    result, _ = _call("checklist", {"profile": PROFILE, "as_of": "2026-10-24"})
    expected = build_checklist(PROFILE, date(2026, 10, 24))
    assert result.structured_content["result"] == json.loads(json.dumps(expected))


def test_checklist_rejects_unknown_facts_and_values():
    bad_value, _ = _call("checklist", {"profile": {"size": "huge"}})
    assert bad_value.is_error and "'micro'" in bad_value.content[0].text
    unknown, _ = _call("checklist", {"profile": {"sector": "cloud"}})
    assert unknown.is_error and "sector" in unknown.content[0].text


def test_crosswalk_tool_accepts_an_id_or_its_ref():
    ref = regs_corpus()[0]["nist-csf/GV.SC-01"]["ref"]
    by_id, _ = _call("crosswalk", {"provision": "nist-csf/GV.SC-01"})
    by_ref, _ = _call("crosswalk", {"provision": ref.lower()})
    out = by_id.structured_content["result"]
    assert out == by_ref.structured_content["result"] and out["provision"] == "nist-csf/GV.SC-01"
    assert out["items"] and out["items"] == json.loads(json.dumps(xw.related("nist-csf/GV.SC-01")))["items"]


def test_crosswalk_tool_rejects_an_unknown_provision():
    result, _ = _call("crosswalk", {"provision": "GDPR Art. 999"})
    assert result.is_error and "Unknown provision" in result.content[0].text


def _read(uri):
    return json.loads(_session(lambda c: c.read_resource(uri)).contents[0].text)


def test_resources_describe_facts_rules_and_mapping_sources():
    assert [f["id"] for f in _read("rag://facts")] == list(Profile.model_fields)
    rules = _read("rag://obligations")
    assert len(rules["obligations"]) == len(load_catalog()["obligations"]) and "eu_nexus" in rules["defs"]
    assert {s["kind"] for s in _read("rag://crosswalk-sources")} == {"official", "community"}


def test_guardrails_cover_every_job():
    text = _session(lambda c: c.read_resource("rag://guardrails")).contents[0].text
    assert "Compliance Workbench" in text and all(w in text for w in ("checklist", "crosswalk", "unverified"))


def test_each_job_has_a_prompt():
    names = {p.name for p in _session(lambda c: c.list_prompts()).prompts}
    assert {"grounded_answer", "compliance_checklist", "crosswalk_lookup", "policy_gap_check"} <= names
    got = _session(lambda c: c.get_prompt("compliance_checklist", {"situation": "We deploy a hiring AI in Germany"}))
    body = got.messages[0].content.text
    assert "rag://facts" in body and "only the facts" in body and "hiring AI" in body


def _gap(args):
    llm, seen = _llm('{"duties": []}'), []

    async def on_progress(progress, total, message):
        seen.append((progress, total, message))

    async def go():
        async with Client(mcp_server.mcp) as client:
            return await client.call_tool("gap_check", args, progress_callback=on_progress)

    with patch.object(mcp_server, "get_llm", return_value=llm):
        return anyio.run(go), seen, llm


def test_every_job_is_a_read_only_tool():
    tools = {t.name: t for t in _session(lambda c: c.list_tools()).tools}
    assert set(tools) == {"ask", "search_knowledge_base", "checklist", "crosswalk", "gap_check"}
    assert all(t.annotations.read_only_hint for t in tools.values())


def test_gap_check_reports_progress_once_per_article():
    result, seen, _ = _gap({"policy_text": POLICY, "articles": ["gdpr/art-33", "gdpr/art-34"]})
    out = result.structured_content["result"]
    assert [r["selection"] for r in out["results"]] == ["gdpr/art-33", "gdpr/art-34"]
    assert [(p, t) for p, t, _ in seen] == [(1, 2), (2, 2)] and seen[0][2] == out["results"][0]["ref"]
    # The fake judge answers nothing, so every duty fails safe to unverified.
    assert out["summary"]["unverified"] == sum(len(r["duties"]) for r in out["results"]) and out["plan"]["units"]


def test_gap_check_refuses_runs_over_budget_before_any_model_call(monkeypatch):
    monkeypatch.setattr(gapcheck, "MAX_TOKENS", 10)
    result, seen, llm = _gap({"policy_text": POLICY, "articles": ["gdpr/art-33"]})
    assert result.is_error and "limit 10" in result.content[0].text
    llm.invoke.assert_not_called()
    assert seen == []


def test_gap_check_applies_the_api_limits():
    too_many, _, _ = _gap({"policy_text": POLICY, "articles": [f"gdpr/art-{n}" for n in range(30, 43)]})
    assert too_many.is_error
    empty, _, _ = _gap({"policy_text": "", "articles": ["gdpr/art-33"]})
    assert empty.is_error


def test_ask_reads_an_attached_copy_into_the_graph_state():
    _, graph = _call("ask", {"question": "What does ISO/IEC 27001 A.5.19 require?", "standards": [{"text": SYNTH}]},
                     state=REGS)
    copies = graph.invoke.call_args.args[0]["standards"]
    assert [c["name"] for c in copies] == ["iso-27001"] and "A.5.19" in copies[0]["clauses"]


def test_ask_rejects_a_copy_it_cannot_identify():
    result, graph = _call("ask", {"question": "q", "standards": [{"text": "A plain policy.\n1 Scope"}]}, state=REGS)
    assert result.is_error and "ISO/IEC 27001" in result.content[0].text
    graph.invoke.assert_not_called()
