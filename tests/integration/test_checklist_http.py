from tests.integration.conftest import INTEGRATION_API_KEY

H = {"X-API-Key": INTEGRATION_API_KEY}


def test_facts_and_checklist_endpoints(integration_client):
    facts = integration_client.get("/checklist/facts", headers=H).json()
    assert facts[0]["id"] == "ai_system" and len(facts) == 19
    body = {"profile": {"ai_roles": ["deployer"], "high_risk": "annex_iii", "eu_established": True},
            "as_of": "2028-01-01"}
    cl = integration_client.post("/checklist", json=body, headers=H).json()
    art26 = next(i for i in cl["groups"]["now"] if i["article"] == "eu-ai-act/art-26")
    assert art26["status"] == "applies" and art26["paragraphs"][0]["text"]
    assert cl["as_of"] == "2028-01-01"


def test_checklist_requires_the_api_key_and_valid_profiles(integration_client):
    assert integration_client.post("/checklist", json={}).status_code == 403
    bad = integration_client.post("/checklist", json={"profile": {"size": "huge"}}, headers=H)
    assert bad.status_code == 422
