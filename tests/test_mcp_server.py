"""MCP server: the workbench exposed to any MCP client with its guardrails intact.
In-memory client (mcp SDK v2); the graph and Pinecone are faked."""
import json
from unittest.mock import MagicMock, patch

import anyio
from mcp.client.client import Client

from app import mcp_server

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
    assert out["sources"][0] == {"n": 1, "title": "Vector database", "url": "https://w/vdb",
                                 "doc_id": "2b529e6a", "score": 0.62}
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
