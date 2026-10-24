import json
import logging
from unittest.mock import MagicMock, patch

from tests.integration.conftest import INTEGRATION_API_KEY
from tests.regs.test_standards import SYNTH

H = {"X-API-Key": INTEGRATION_API_KEY}


def test_parse_policy_and_standard(integration_client):
    r = integration_client.post("/workspace/parse", headers=H, json={"kind": "policy", "title": "P", "text": "Line one.\n\nLine two."})
    assert r.json()["passages"][0]["id"] == "p1"
    r = integration_client.post("/workspace/parse", headers=H, json={"kind": "standard", "title": "copy", "text": SYNTH}).json()
    assert r["standard"] == "iso-27001" and "A.5.19" in r["clauses"] and r["coverage"]["annex_a_expected"] == 93
    bad = integration_client.post("/workspace/parse", headers=H, json={"kind": "standard", "title": "x", "text": "no standard here"})
    assert bad.status_code == 422


def test_gapcheck_plan_and_budget(integration_client):
    body = {"policy_text": "We notify the controller without undue delay after becoming aware of a breach.", "articles": ["gdpr/art-33"]}
    p = integration_client.post("/gapcheck/plan", headers=H, json=body).json()
    assert p["units"][0]["ref"] == "GDPR Art. 33" and p["units"][0]["duties"] == 5 and p["est_tokens"] > 0
    with patch("app.services.regs.gapcheck.MAX_TOKENS", 10):
        r = integration_client.post("/gapcheck", headers=H, json=body)
    assert r.status_code == 422 and "fewer" in r.json()["detail"]


def test_gapcheck_stream_never_touches_pinecone_or_logs_the_policy(integration_client, caplog):
    secret = "ZEBRA-7731 confidential clause"
    policy = f"{secret}. When we act as a processor we notify the controller without undue delay after becoming aware of a breach."
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps({"duties": []}), usage_metadata=None, response_metadata={})
    with patch("app.routers.workspace.get_llm", return_value=llm), \
            patch("app.services.pinecone_store.search") as search, caplog.at_level(logging.DEBUG):
        resp = integration_client.post("/gapcheck/stream", headers=H, json={"policy_text": policy, "articles": ["gdpr/art-33"]})
    assert [l for l in resp.text.splitlines() if l.startswith("event:")] == ["event: plan", "event: article", "event: done"]
    search.assert_not_called()
    assert secret not in caplog.text
