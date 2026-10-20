from unittest.mock import MagicMock

from app.services.regs.router import understand_request

import json


def _route(intent, *facts):
    """Router reply in the evidence format: (fact, values, evidence copied from the request)."""
    return json.dumps({"intent": intent, "profile_updates": [
        {"fact": f, "values": v, "evidence": e} for f, v, e in facts]})


def _llm(content):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


Q = "We deploy a CV screening tool in the EU. What do we need to do?"


def test_returns_intent_and_facts_backed_by_the_users_words_with_strict_schema():
    llm = _llm(_route("checklist", ("ai_roles", ["deployer"], "We deploy"), ("high_risk", ["annex_iii"], "CV screening tool"),
                      ("eu_market", ["true"], "in the EU")))
    out, usage = understand_request(llm, Q, {})
    assert out == {"intent": "checklist",
                   "profile_updates": {"ai_roles": ["deployer"], "high_risk": "annex_iii", "eu_market": True}}
    assert usage["total_tokens"] == 0
    assert llm.invoke.call_args.kwargs["response_format"]["json_schema"]["strict"] is True
    assert "<request>" in llm.invoke.call_args.args[0][-1].content


def test_facts_without_verbatim_evidence_are_dropped():
    out, _ = understand_request(_llm(_route("checklist", ("sensitive_data", ["false"], "no health data"),
                                            ("fria_trigger", ["false"], ""), ("size", ["medium"], "a company"),
                                            ("eu_established", ["true"], "in the eu"))), Q, {})
    assert out["profile_updates"] == {"eu_established": True}  # evidence match ignores case


def test_invalid_values_are_dropped():
    out, _ = understand_request(_llm(_route("ask", ("ai_roles", ["wizard"], "We deploy"), ("high_risk", ["very"], "CV screening"),
                                            ("eu_market", ["yes", "no"], "in the EU"))), Q, {})
    assert out == {"intent": "ask", "profile_updates": {}}


def test_headcount_in_the_evidence_sets_the_size_band():
    q = "We are a 40-person company, soon about 300 staff."
    out, _ = understand_request(_llm(_route("ask", ("size", ["medium"], "40-person company"))), q, {})
    assert out["profile_updates"] == {"size": "small"}
    out, _ = understand_request(_llm(_route("ask", ("size", ["small"], "about 300 staff"))), q, {})
    assert out["profile_updates"] == {"size": "large"}


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


def test_router_prompt_spells_out_value_meanings_and_asks_for_evidence():
    from app.services.regs.router import _SYSTEM

    assert "small = small (10-49 staff)" in _SYSTEM
    assert "evidence" in _SYSTEM and "copied exactly" in _SYSTEM
