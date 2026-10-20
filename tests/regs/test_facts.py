from typing import get_args

from app.schemas.chat import Profile
from app.services.regs.facts import load_facts, normalize_profile


def _literal_values(field):
    ann = Profile.model_fields[field].annotation
    inner = get_args(ann)[0]                      # Optional[X] -> X
    if get_args(inner) and not isinstance(get_args(inner)[0], str):
        inner = get_args(inner)[0]                # List[Literal] -> Literal
    return set(get_args(inner))


def test_facts_match_the_profile_model():
    facts = load_facts()
    assert [f["id"] for f in facts] == list(Profile.model_fields)
    for f in facts:
        assert f["type"] in ("bool", "choice", "multi") and f["label"] and f["group"] and f["basis"]
        if f["type"] != "bool":
            assert set(f["values"]) <= _literal_values(f["id"]), f["id"]
            assert set(f["labels"]) == set(f["values"]), f["id"]
        else:
            assert f["short"], f["id"]


def test_normalize_profile_drops_unknowns_and_fills_implied_facts():
    assert normalize_profile({"ai_roles": ["deployer"], "high_risk": "unknown", "size": None}) == \
        {"ai_roles": ["deployer"], "ai_system": True}
    assert normalize_profile({"data_roles": ["processor"]})["personal_data"] is True
    assert normalize_profile({"gpai_systemic_risk": True})["gpai_model"] is True
    assert normalize_profile({"ai_roles": ["provider"], "ai_system": False})["ai_system"] is False
