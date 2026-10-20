from unittest.mock import MagicMock

from app.services.regs.router import understand_request

NULLS = '"ai_system": null, "gpai_model": null, "personal_data": null, "eu_market": null, "size": null'


def _llm(content):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


def test_returns_intent_and_stated_profile_facts_with_strict_schema():
    llm = _llm('{"intent": "checklist", "profile_updates": {"ai_roles": ["deployer"], "high_risk": "annex_iii", '
               '"size": null, "ai_system": true, "gpai_model": null, "personal_data": null, "eu_market": true}}')
    out, usage = understand_request(llm, "We deploy a CV screening tool in the EU. What do we need to do?", {})
    assert out == {"intent": "checklist",
                   "profile_updates": {"ai_roles": ["deployer"], "high_risk": "annex_iii", "ai_system": True, "eu_market": True}}
    assert usage["total_tokens"] == 0
    assert llm.invoke.call_args.kwargs["response_format"]["json_schema"]["strict"] is True
    assert "<request>" in llm.invoke.call_args.args[0][-1].content


def test_invalid_updates_are_dropped():
    out, _ = understand_request(_llm('{"intent": "ask", "profile_updates": {"ai_roles": ["wizard"], "high_risk": "very", '
                                  + NULLS + '}}'), "q", {})
    assert out == {"intent": "ask", "profile_updates": {}}


def test_failures_fall_back_to_ask():
    broken = MagicMock()
    broken.invoke.side_effect = RuntimeError("down")
    assert understand_request(broken, "q", {})[0] == {"intent": "ask", "profile_updates": {}}
    assert understand_request(_llm("not json"), "q", {})[0] == {"intent": "ask", "profile_updates": {}}
    assert understand_request(_llm('{"intent": "dance", "profile_updates": {}}'), "q", {})[0]["intent"] == "ask"


def test_router_eval_scoring_and_labels():
    import json
    from pathlib import Path

    from eval.run_router import score
    from app.services.regs.router import INTENTS

    rows = [json.loads(l) for l in open(Path(__file__).resolve().parents[2] / "eval/regs_router.jsonl", encoding="utf-8")]
    rows = [r for r in rows if not r.get("_meta")]
    assert len(rows) == 30 and {r["intent"] for r in rows} == set(INTENTS)
    rep = score(rows[:2], lambda q: "ask" if "Article 26" in q else "checklist")
    assert rep["accuracy"] == 0.5 and rep["misses"][0]["predicted"] == "checklist"


def test_previous_checklist_turns_fact_only_messages_into_a_checklist():
    from app.services.regs.router import apply_previous_intent

    assert apply_previous_intent("ask", {"size": "large"}, "checklist", "We have about 300 staff.") == "checklist"
    assert apply_previous_intent("clarify", {"ai_roles": ["provider"]}, "checklist", "we are also the provider") == "checklist"
    assert apply_previous_intent("ask", {"ai_roles": ["deployer"]}, "checklist", "As a deployer, what does Art. 26 say?") == "ask"
    assert apply_previous_intent("ask", {}, "checklist", "Thanks") == "ask"
    assert apply_previous_intent("gap_check", {"size": "large"}, "checklist", "Check our policy") == "gap_check"
    assert apply_previous_intent("ask", {"size": "large"}, None, "We have 300 staff") == "ask"
