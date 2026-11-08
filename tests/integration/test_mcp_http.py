"""MCP over Streamable HTTP at /mcp, protected by the same X-API-Key as the REST API."""
from tests.integration.conftest import INTEGRATION_API_KEY

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "0"}}}
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def test_mcp_rejects_missing_api_key(integration_client):
    r = integration_client.post("/mcp", json=INIT, headers=HEADERS, follow_redirects=False)
    assert r.status_code == 403


def test_mcp_rejects_wrong_api_key(integration_client):
    r = integration_client.post("/mcp", json=INIT, headers={**HEADERS, "X-API-Key": "wrong"})
    assert r.status_code == 403


def test_mcp_initializes_with_api_key(integration_client):
    r = integration_client.post("/mcp", json=INIT, headers={**HEADERS, "X-API-Key": INTEGRATION_API_KEY},
                               follow_redirects=False)  # real MCP clients do not follow POST redirects
    assert r.status_code == 200, r.text
    assert "rag-agent-workbench" in r.text


def test_mcp_answers_as_a_stream_so_progress_can_reach_the_client(integration_client):
    r = integration_client.post("/mcp", json=INIT, headers={**HEADERS, "X-API-Key": INTEGRATION_API_KEY},
                                follow_redirects=False)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
