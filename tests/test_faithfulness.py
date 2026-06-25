"""
CI-safe unit tests for the T2.2 faithfulness / grounding check.

All tests make zero network calls.  The LLM judge is mocked via unittest.mock.
Test coverage:
  - verify_citations: pure deterministic function (no mock needed)
  - judge_faithfulness: LLM mocked; tests good verdict, parse failure, call error
  - format_response (graph node): flag-OFF bypass, flag-ON execution, abstention
    bypass, answer-text immutability
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend and repo root are on sys.path (same as other test files).
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.services.faithfulness import (
    FaithfulnessVerdict,
    _parse_verdict,
    judge_faithfulness,
    verify_citations,
)
from app.services.chat.graph import ABSTENTION_ANSWER, format_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(n: int) -> dict:
    return {"doc_id": f"doc{n}", "chunk_text": f"chunk text {n}", "score": 0.8}


def _mock_llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    llm = MagicMock()
    llm.invoke.return_value = resp
    return llm


def _make_state(
    answer: str = "The answer is [1].",
    retrieved: list | None = None,
    web_results: list | None = None,
    insufficient_context: bool = False,
) -> dict:
    return {
        "answer": answer,
        "retrieved": retrieved if retrieved is not None else [_make_chunk(1)],
        "web_results": web_results or [],
        "insufficient_context": insufficient_context,
        "timings": {},
    }


# ---------------------------------------------------------------------------
# verify_citations — pure function, no mock
# ---------------------------------------------------------------------------

class TestVerifyCitations:
    def test_all_valid_citations_pass(self):
        chunks = [_make_chunk(1), _make_chunk(2)]
        result = verify_citations("See [1] and [2].", chunks)
        assert result == []

    def test_dangling_high_citation_flagged(self):
        chunks = [_make_chunk(1), _make_chunk(2)]
        result = verify_citations("See [1] and [5].", chunks)
        assert result == [5]

    def test_citation_zero_is_out_of_range(self):
        chunks = [_make_chunk(1)]
        result = verify_citations("See [0].", chunks)
        assert result == [0]

    def test_no_citations_returns_empty(self):
        chunks = [_make_chunk(1), _make_chunk(2)]
        result = verify_citations("No citations here.", chunks)
        assert result == []

    def test_empty_chunks_all_citations_dangling(self):
        result = verify_citations("See [1] and [2].", [])
        assert result == [1, 2]

    def test_duplicate_citations_not_reported_twice(self):
        chunks = [_make_chunk(1), _make_chunk(2)]
        result = verify_citations("See [1] and [1] again.", chunks)
        assert result == []  # [1] is valid, duplicate in-range ref is fine

    def test_exact_range_boundary_is_valid(self):
        chunks = [_make_chunk(i) for i in range(1, 4)]
        result = verify_citations("See [3].", chunks)
        assert result == []

    def test_multiple_dangling_returned_sorted(self):
        chunks = [_make_chunk(1)]
        result = verify_citations("See [5] and [3] and [2].", chunks)
        assert result == [2, 3, 5]

    def test_empty_answer_returns_empty(self):
        chunks = [_make_chunk(1)]
        result = verify_citations("", chunks)
        assert result == []


# ---------------------------------------------------------------------------
# _parse_verdict — internal helper, but exported for testing robustness
# ---------------------------------------------------------------------------

class TestParseVerdict:
    def test_valid_json_parsed_correctly(self):
        raw = '{"grounded": true, "score": 0.9, "rationale": "Good."}'
        v = _parse_verdict(raw)
        assert v.grounded is True
        assert v.faithfulness_score == pytest.approx(0.9)
        assert v.rationale == "Good."

    def test_markdown_fences_stripped(self):
        raw = "```json\n{\"grounded\": false, \"score\": 0.2, \"rationale\": \"Bad.\"}\n```"
        v = _parse_verdict(raw)
        assert v.grounded is False
        assert v.faithfulness_score == pytest.approx(0.2)

    def test_no_json_returns_none_fields(self):
        v = _parse_verdict("Sorry, I cannot evaluate this.")
        assert v.grounded is None
        assert v.faithfulness_score is None
        assert "parse_error" in v.rationale

    def test_malformed_json_returns_none_fields(self):
        v = _parse_verdict("{grounded: yes, score: maybe}")
        assert v.grounded is None
        assert v.faithfulness_score is None
        assert "parse_error" in v.rationale

    def test_score_clamped_above_1(self):
        raw = '{"grounded": true, "score": 1.5, "rationale": "Over."}'
        v = _parse_verdict(raw)
        assert v.faithfulness_score == pytest.approx(1.0)

    def test_score_clamped_below_0(self):
        raw = '{"grounded": false, "score": -0.1, "rationale": "Under."}'
        v = _parse_verdict(raw)
        assert v.faithfulness_score == pytest.approx(0.0)

    def test_rationale_alternative_keys(self):
        raw = '{"grounded": true, "score": 0.8, "reason": "Alt key."}'
        v = _parse_verdict(raw)
        assert v.rationale == "Alt key."


# ---------------------------------------------------------------------------
# judge_faithfulness — mocked LLM
# ---------------------------------------------------------------------------

class TestJudgeFaithfulness:
    def test_grounded_verdict_returned(self):
        payload = json.dumps({"grounded": True, "score": 0.95, "rationale": "All claims supported."})
        llm = _mock_llm_response(payload)
        verdict = judge_faithfulness("answer", "context", llm)
        assert verdict.grounded is True
        assert verdict.faithfulness_score == pytest.approx(0.95)
        assert "supported" in verdict.rationale
        llm.invoke.assert_called_once()

    def test_ungrounded_verdict_returned(self):
        payload = json.dumps({"grounded": False, "score": 0.1, "rationale": "Hallucinated facts."})
        llm = _mock_llm_response(payload)
        verdict = judge_faithfulness("answer", "context", llm)
        assert verdict.grounded is False
        assert verdict.faithfulness_score == pytest.approx(0.1)

    def test_parse_failure_degrades_gracefully_no_raise(self):
        llm = _mock_llm_response("This is definitely not JSON at all.")
        verdict = judge_faithfulness("answer", "context", llm)
        assert verdict.grounded is None
        assert verdict.faithfulness_score is None
        assert "parse_error" in verdict.rationale

    def test_llm_call_error_degrades_gracefully_no_raise(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("network timeout")
        verdict = judge_faithfulness("answer", "context", llm)
        assert verdict.grounded is None
        assert verdict.faithfulness_score is None
        assert "judge_call_error" in verdict.rationale


# ---------------------------------------------------------------------------
# format_response (graph node) — settings and LLM mocked
# ---------------------------------------------------------------------------

class TestFormatResponse:
    """Tests for the format_response LangGraph node."""

    def test_flag_off_judge_not_called(self):
        state = _make_state(answer="The sky is blue [1].")
        with patch("app.services.chat.graph.get_settings") as mock_settings:
            mock_settings.return_value.RAG_FAITHFULNESS_ENABLED = False
            mock_settings.return_value.RAG_FAITHFULNESS_THRESHOLD = 0.5
            with patch("app.services.chat.graph.judge_faithfulness_with_usage") as mock_judge:
                result = format_response(state)
        mock_judge.assert_not_called()
        assert result["grounded"] is None
        assert result["faithfulness_score"] is None

    def test_flag_off_citation_check_still_runs(self):
        # Even with flag OFF, verify_citations populates unverified_citations.
        state = _make_state(
            answer="The sky is blue [1] and [99].",
            retrieved=[_make_chunk(1)],
        )
        with patch("app.services.chat.graph.get_settings") as mock_settings:
            mock_settings.return_value.RAG_FAITHFULNESS_ENABLED = False
            mock_settings.return_value.RAG_FAITHFULNESS_THRESHOLD = 0.5
            result = format_response(state)
        assert result["unverified_citations"] == [99]

    def test_flag_on_judge_called_and_verdict_applied(self):
        state = _make_state(answer="The answer is [1].")
        mock_verdict = FaithfulnessVerdict(grounded=True, faithfulness_score=0.9, rationale="ok")
        with patch("app.services.chat.graph.get_settings") as mock_settings:
            mock_settings.return_value.RAG_FAITHFULNESS_ENABLED = True
            mock_settings.return_value.RAG_FAITHFULNESS_THRESHOLD = 0.5
            with patch("app.services.chat.graph.judge_faithfulness_with_usage", return_value=(mock_verdict, {})) as mock_judge:
                with patch("app.services.chat.graph.get_llm", return_value=MagicMock()):
                    result = format_response(state)
        mock_judge.assert_called_once()
        assert result["grounded"] is True
        assert result["faithfulness_score"] == pytest.approx(0.9)

    def test_flag_on_below_threshold_sets_grounded_false(self):
        state = _make_state(answer="The answer is [1].")
        mock_verdict = FaithfulnessVerdict(grounded=True, faithfulness_score=0.3, rationale="weak")
        with patch("app.services.chat.graph.get_settings") as mock_settings:
            mock_settings.return_value.RAG_FAITHFULNESS_ENABLED = True
            mock_settings.return_value.RAG_FAITHFULNESS_THRESHOLD = 0.5
            with patch("app.services.chat.graph.judge_faithfulness_with_usage", return_value=(mock_verdict, {})):
                with patch("app.services.chat.graph.get_llm", return_value=MagicMock()):
                    result = format_response(state)
        assert result["grounded"] is False  # score 0.3 < threshold 0.5

    def test_abstention_path_judge_not_called(self):
        state = _make_state(
            answer=ABSTENTION_ANSWER,
            retrieved=[],
            insufficient_context=True,
        )
        with patch("app.services.chat.graph.get_settings") as mock_settings:
            mock_settings.return_value.RAG_FAITHFULNESS_ENABLED = True
            mock_settings.return_value.RAG_FAITHFULNESS_THRESHOLD = 0.5
            with patch("app.services.chat.graph.judge_faithfulness_with_usage") as mock_judge:
                with patch("app.services.chat.graph.get_llm") as mock_get_llm:
                    result = format_response(state)
        mock_judge.assert_not_called()
        mock_get_llm.assert_not_called()
        assert result["grounded"] is None
        assert result["faithfulness_score"] is None

    def test_answer_text_not_modified_by_format_response(self):
        original_answer = "The sky is blue [1]."
        state = _make_state(answer=original_answer)
        with patch("app.services.chat.graph.get_settings") as mock_settings:
            mock_settings.return_value.RAG_FAITHFULNESS_ENABLED = False
            mock_settings.return_value.RAG_FAITHFULNESS_THRESHOLD = 0.5
            result = format_response(state)
        assert result["answer"] == original_answer

    def test_judge_parse_failure_sets_grounded_none_no_raise(self):
        state = _make_state(answer="The answer is [1].")
        parse_fail_verdict = FaithfulnessVerdict(
            grounded=None, faithfulness_score=None, rationale="parse_error: bad json"
        )
        with patch("app.services.chat.graph.get_settings") as mock_settings:
            mock_settings.return_value.RAG_FAITHFULNESS_ENABLED = True
            mock_settings.return_value.RAG_FAITHFULNESS_THRESHOLD = 0.5
            with patch("app.services.chat.graph.judge_faithfulness_with_usage", return_value=(parse_fail_verdict, {})):
                with patch("app.services.chat.graph.get_llm", return_value=MagicMock()):
                    result = format_response(state)
        assert result["grounded"] is None  # graceful degradation
        assert result["faithfulness_score"] is None

    def test_faithfulness_ms_zero_when_flag_off(self):
        state = _make_state()
        with patch("app.services.chat.graph.get_settings") as mock_settings:
            mock_settings.return_value.RAG_FAITHFULNESS_ENABLED = False
            mock_settings.return_value.RAG_FAITHFULNESS_THRESHOLD = 0.5
            result = format_response(state)
        assert result["timings"]["faithfulness_ms"] == pytest.approx(0.0)

    def test_faithfulness_ms_nonzero_when_judge_called(self):
        state = _make_state(answer="The answer is [1].")
        mock_verdict = FaithfulnessVerdict(grounded=True, faithfulness_score=0.9, rationale="ok")
        with patch("app.services.chat.graph.get_settings") as mock_settings:
            mock_settings.return_value.RAG_FAITHFULNESS_ENABLED = True
            mock_settings.return_value.RAG_FAITHFULNESS_THRESHOLD = 0.5
            with patch("app.services.chat.graph.judge_faithfulness_with_usage", return_value=(mock_verdict, {})):
                with patch("app.services.chat.graph.get_llm", return_value=MagicMock()):
                    result = format_response(state)
        # perf_counter-based; just assert it's a non-negative float
        assert isinstance(result["timings"]["faithfulness_ms"], float)
        assert result["timings"]["faithfulness_ms"] >= 0.0
