"""
CI-safe unit tests for T2.5 — history-aware query contextualization.

Coverage:
  (a) Flag OFF  → contextualize_query node is a pass-through; LLM not called.
  (b) Flag ON + no history → no LLM call (no-op for first-turn requests).
  (c) Flag ON + history → rewritten query used; state["query"] and
      state["contextualized_query"] reflect the rewrite.
  (d) LLM error → falls back to original query; state["contextualized_query"] is None.
  (e) contextualize_followup with history → calls llm.invoke and returns text + usage.
  (f) contextualize_followup without history → returns original query with zero usage.
  (g) contextualize_followup with LLM error → returns original query with zero usage.

All tests make ZERO network calls.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.services.chat.graph import contextualize_query
from app.services.contextualize import contextualize_followup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(enabled: bool = True) -> MagicMock:
    s = MagicMock()
    s.RAG_CONTEXTUALIZE_ENABLED = enabled
    return s


def _history() -> list:
    return [
        {"role": "user", "content": "What is retrieval-augmented generation?"},
        {"role": "assistant", "content": "RAG is a technique that combines retrieval with generation."},
    ]


def _base_state(query: str = "What about its advantages?", history: list | None = None) -> Dict[str, Any]:
    return {
        "query": query,
        "chat_history": _history() if history is None else history,
        "timings": {},
    }


def _mock_llm_response(text: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> MagicMock:
    """Return a mock LLM whose invoke() returns an AIMessage-like object with usage."""
    response = MagicMock()
    response.content = text
    if prompt_tokens or completion_tokens:
        response.usage_metadata = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    else:
        # MagicMock auto-creates attributes; isinstance(MagicMock(), dict) is False
        # so extract_token_usage will safely return zeros for no-usage case.
        del response.usage_metadata  # force AttributeError path in extract_token_usage
    llm = MagicMock()
    llm.invoke.return_value = response
    return llm


# ---------------------------------------------------------------------------
# (a) Flag OFF — exact pass-through, no LLM call
# ---------------------------------------------------------------------------

class TestContextualizeQueryFlagOff:
    def test_flag_off_returns_unchanged_state(self):
        state = _base_state()
        original_query = state["query"]
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=False)):
            result = contextualize_query(state)
        assert result["query"] == original_query

    def test_flag_off_no_llm_call(self):
        state = _base_state()
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=False)):
            with patch("app.services.chat.graph.get_llm") as mock_get_llm:
                contextualize_query(state)
        mock_get_llm.assert_not_called()

    def test_flag_off_contextualized_query_not_set(self):
        state = _base_state()
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=False)):
            result = contextualize_query(state)
        assert "contextualized_query" not in result or result.get("contextualized_query") is None


# ---------------------------------------------------------------------------
# (b) Flag ON + no history → no LLM call
# ---------------------------------------------------------------------------

class TestContextualizeQueryNoHistory:
    def test_no_history_no_llm_call(self):
        state = _base_state(history=[])
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=True)):
            with patch("app.services.chat.graph.get_llm") as mock_get_llm:
                contextualize_query(state)
        mock_get_llm.assert_not_called()

    def test_no_history_query_unchanged(self):
        original_query = "What about its advantages?"
        state = _base_state(query=original_query, history=[])
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=True)):
            result = contextualize_query(state)
        assert result["query"] == original_query

    def test_none_history_treated_as_no_history(self):
        state = _base_state()
        state["chat_history"] = None  # type: ignore[assignment]
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=True)):
            with patch("app.services.chat.graph.get_llm") as mock_get_llm:
                contextualize_query(state)
        mock_get_llm.assert_not_called()


# ---------------------------------------------------------------------------
# (c) Flag ON + history → rewritten query used for retrieval
# ---------------------------------------------------------------------------

class TestContextualizeQueryWithHistory:
    def test_rewritten_query_replaces_original(self):
        rewritten = "What are the advantages of retrieval-augmented generation?"
        state = _base_state()
        mock_llm = _mock_llm_response(rewritten, prompt_tokens=50, completion_tokens=12)
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=True)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm):
                result = contextualize_query(state)
        assert result["query"] == rewritten

    def test_contextualized_query_set_in_state(self):
        rewritten = "What are the advantages of retrieval-augmented generation?"
        state = _base_state()
        mock_llm = _mock_llm_response(rewritten, prompt_tokens=50, completion_tokens=12)
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=True)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm):
                result = contextualize_query(state)
        assert result["contextualized_query"] == rewritten

    def test_token_usage_accumulated_in_state(self):
        rewritten = "What are the advantages of retrieval-augmented generation?"
        state = _base_state()
        mock_llm = _mock_llm_response(rewritten, prompt_tokens=60, completion_tokens=15)
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=True)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm):
                result = contextualize_query(state)
        # Token accumulation from contextualize call
        usage_by_call = result.get("token_usage_by_call") or {}
        assert "contextualize" in usage_by_call
        assert usage_by_call["contextualize"]["prompt_tokens"] == 60
        assert usage_by_call["contextualize"]["completion_tokens"] == 15
        assert usage_by_call["contextualize"]["total_tokens"] == 75

    def test_llm_called_once(self):
        state = _base_state()
        mock_llm = _mock_llm_response("standalone question")
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=True)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm):
                contextualize_query(state)
        mock_llm.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# (d) LLM error → fallback to original query, contextualized_query is None
# ---------------------------------------------------------------------------

class TestContextualizeQueryErrorFallback:
    def test_llm_exception_falls_back_to_original(self):
        original_query = "What about its advantages?"
        state = _base_state(query=original_query)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("Groq timeout")
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=True)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm):
                result = contextualize_query(state)
        assert result["query"] == original_query

    def test_llm_error_sets_contextualized_query_none(self):
        state = _base_state()
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("Groq timeout")
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=True)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm):
                result = contextualize_query(state)
        assert result.get("contextualized_query") is None

    def test_llm_returns_empty_does_not_replace_query(self):
        original_query = "What about its advantages?"
        state = _base_state(query=original_query)
        mock_llm = _mock_llm_response("")  # empty response
        with patch("app.services.chat.graph.get_settings", return_value=_make_settings(enabled=True)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm):
                result = contextualize_query(state)
        # Empty text from LLM falls back to original query
        assert result["query"] == original_query


# ---------------------------------------------------------------------------
# (e) contextualize_followup — unit tests of the service function
# ---------------------------------------------------------------------------

class TestContextualizeFollowup:
    def test_with_history_calls_llm_and_returns_text(self):
        history = _history()
        llm = _mock_llm_response("What are the benefits of RAG?")
        text, usage = contextualize_followup("What about it?", history, llm)
        assert text == "What are the benefits of RAG?"
        llm.invoke.assert_called_once()

    def test_with_history_returns_zero_usage_when_no_metadata(self):
        history = _history()
        llm = _mock_llm_response("What are the benefits of RAG?")  # no usage_metadata
        _, usage = contextualize_followup("What about it?", history, llm)
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_no_history_returns_original_no_llm_call(self):
        llm = MagicMock()
        text, usage = contextualize_followup("Original question?", [], llm)
        assert text == "Original question?"
        llm.invoke.assert_not_called()
        assert usage["total_tokens"] == 0

    def test_llm_error_returns_original_and_zero_usage(self):
        history = _history()
        llm = MagicMock()
        llm.invoke.side_effect = ConnectionError("network failure")
        text, usage = contextualize_followup("What about it?", history, llm)
        assert text == "What about it?"
        assert usage["total_tokens"] == 0

    def test_llm_empty_response_returns_original(self):
        history = _history()
        llm = _mock_llm_response("")
        text, usage = contextualize_followup("What about it?", history, llm)
        assert text == "What about it?"
