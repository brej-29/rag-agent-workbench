"""/chat/stream `step` events: the UI's live pipeline trace.

Emitted as each phase completes (phase-level, not per node).  The backend sends
facts only (ids, counts, scores, ms); all wording lives in the UI.  The gate step
carries every candidate score and the relevance floor so the UI can draw why the
system answered or abstained.
"""
from unittest.mock import patch

from tests.test_streaming import (
    _collect, _initial_state, _mock_llm, _mock_post_gen, _mock_post_gen_abstention, _parse, _pre_gen_state,
)
from app.services.chat.streaming import stream_chat_response


def _run(pre_gen, abstain=False):
    def prepare(state):
        state["retrieved"] = [] if abstain else state["retrieved"]
        return abstain, state["retrieved"], [] if abstain else ["msgs"]

    with patch("app.services.chat.streaming._run_pre_generation_pipeline", return_value=pre_gen), \
         patch("app.services.chat.streaming._prepare_generation_inputs", side_effect=prepare), \
         patch("app.services.chat.streaming.get_llm", return_value=_mock_llm()), \
         patch("app.services.chat.streaming._run_post_generation",
               side_effect=_mock_post_gen_abstention if abstain else _mock_post_gen):
        return [_parse(f) for f in _collect(stream_chat_response(_initial_state(), {}))]


def _steps(events):
    return [(d["id"], d["status"]) for e, d in events if e == "step"]


def test_answer_path_step_order_brackets_the_tokens():
    events = _run(_pre_gen_state())
    names = [e if e != "step" else f"step:{d['id']}:{d['status']}" for e, d in events]
    first_token = names.index("token")
    assert names[:first_token] == ["step:retrieve:done", "step:gate:done", "step:generate:running"]
    assert names[-3:] == ["step:generate:done", "step:verify:done", "done"]


def test_gate_step_reports_scores_against_the_floor():
    pre = _pre_gen_state()
    pre["retrieved"] = [{"score": 0.85, "chunk_text": "a"}, {"score": 0.12, "chunk_text": "b"}]
    gate = next(d for e, d in _run(pre) if e == "step" and d["id"] == "gate")
    assert gate["detail"]["scores"] == [0.85, 0.12]
    assert gate["detail"]["floor"] > 0 and gate["detail"]["abstain"] is False


def test_abstention_skips_generation_and_verification():
    events = _run(_pre_gen_state(), abstain=True)
    assert _steps(events) == [("retrieve", "done"), ("gate", "done"), ("generate", "skipped")]
    gate = next(d for e, d in events if e == "step" and d["id"] == "gate")
    assert gate["detail"]["abstain"] is True and gate["detail"]["kept"] == 0


def test_crag_and_web_steps_appear_only_when_they_ran():
    pre = _pre_gen_state()
    pre.update(crag_iterations=2, corrective_action="rewrite", web_fallback_used=True,
               web_results=[{"title": "w", "url": "u", "chunk_text": "t"}])
    ids = [i for i, _ in _steps(_run(pre))]
    assert ids[:3] == ["retrieve", "correct", "web"]
    assert "correct" not in [i for i, _ in _steps(_run(_pre_gen_state()))]


def test_done_sources_carry_doc_id():
    pre = _pre_gen_state()
    pre["retrieved"][0]["doc_id"] = "abc123"
    done = next(d for e, d in _run(pre) if e == "done")
    assert done["sources"][0]["doc_id"] == "abc123"
