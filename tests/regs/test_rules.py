from app.services.regs.rules import evaluate

DEFS = {"hr": {"high_risk": ["annex_iii", "annex_i"]},
        "deployer": {"all": [{"ai_system": True}, {"ai_roles": "deployer"}]}}


def test_fact_tests_on_scalars_lists_and_unknowns():
    assert evaluate({"size": ["medium", "large"]}, {"size": "large"}, {}) == (True, {"size": "large"})
    assert evaluate({"ai_roles": "provider"}, {"ai_roles": ["deployer"]}, {}) == (False, {"ai_roles": ["deployer"]})
    assert evaluate({"ai_uses": ["chatbot"]}, {"ai_uses": []}, {}) == (False, {"ai_uses": []})
    assert evaluate({"eu_established": False}, {"eu_established": False}, {}) == (True, {"eu_established": False})
    assert evaluate({"size": "large"}, {}, {}) == (None, {"size": None})


def test_all_any_not_in_three_valued_logic_with_explanations():
    p = {"ai_system": True, "ai_roles": ["deployer"]}
    assert evaluate({"all": [{"use": "deployer"}, {"use": "hr"}]}, p, DEFS) == (None, {"high_risk": None})
    assert evaluate({"all": [{"use": "deployer"}, {"use": "hr"}]}, {**p, "high_risk": "annex_iii"}, DEFS) == (
        True, {"ai_system": True, "ai_roles": ["deployer"], "high_risk": "annex_iii"})
    assert evaluate({"all": [{"use": "hr"}, {"ai_roles": "provider"}]}, p, DEFS) == (False, {"ai_roles": ["deployer"]})
    assert evaluate({"any": [{"size": "large"}, {"ai_roles": "deployer"}]}, p, DEFS) == (True, {"ai_roles": ["deployer"]})
    assert evaluate({"any": [{"size": "large"}, {"ai_roles": "provider"}]}, p, DEFS) == (None, {"size": None})
    assert evaluate({"any": [{"size": "large"}, {"ai_roles": "provider"}]}, {**p, "size": "small"}, DEFS) == (
        False, {"size": "small", "ai_roles": ["deployer"]})
    assert evaluate({"not": {"size": "large"}}, {"size": "small"}, {}) == (True, {"size": "small"})
    assert evaluate({"not": {"size": "large"}}, {}, {}) == (None, {"size": None})
