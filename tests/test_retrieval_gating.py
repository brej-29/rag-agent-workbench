"""
CI-safe unit tests for retrieval gating (T1.3).

Tests cover:
  1. filter_chunks_by_score — pure function in rag_prompt.py
  2. generate_answer empty-context guard — deterministic abstention, Groq NOT called
  3. generate_answer normal path — Groq IS called when usable context exists

All tests make ZERO network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Add backend to sys.path for `from app... import ...`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.services.prompts.rag_prompt import filter_chunks_by_score  # noqa: E402
from app.services.chat.graph import ABSTENTION_ANSWER  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(score: float, text: str = "content", source: str = "wiki") -> Dict[str, Any]:
    return {"source": source, "title": "T", "url": "", "score": score, "chunk_text": text}


def _make_mock_settings(min_chunk_score: float = 0.25) -> MagicMock:
    s = MagicMock()
    s.RAG_MIN_CHUNK_SCORE = min_chunk_score
    s.RAG_RERANK_ENABLED = False  # reranking OFF by default in retrieval-gating tests
    s.RAG_DEFAULT_TOP_K = 5
    return s


# ---------------------------------------------------------------------------
# 1. filter_chunks_by_score — pure function
# ---------------------------------------------------------------------------

class TestFilterChunksByScore:
    def test_all_above_floor(self):
        chunks = [_make_chunk(0.9), _make_chunk(0.5), _make_chunk(0.3)]
        result = filter_chunks_by_score(chunks, min_chunk_score=0.25)
        assert result == chunks

    def test_all_below_floor(self):
        chunks = [_make_chunk(0.1), _make_chunk(0.2), _make_chunk(0.24)]
        result = filter_chunks_by_score(chunks, min_chunk_score=0.25)
        assert result == []

    def test_exactly_at_floor_is_included(self):
        chunk = _make_chunk(0.25)
        result = filter_chunks_by_score([chunk], min_chunk_score=0.25)
        assert result == [chunk]

    def test_mixed_above_and_below(self):
        above1 = _make_chunk(0.8, text="good")
        below = _make_chunk(0.1, text="weak")
        above2 = _make_chunk(0.4, text="ok")
        chunks = [above1, below, above2]
        result = filter_chunks_by_score(chunks, min_chunk_score=0.25)
        assert result == [above1, above2]

    def test_preserves_order(self):
        chunks = [_make_chunk(0.9, text="first"), _make_chunk(0.6, text="second")]
        result = filter_chunks_by_score(chunks, min_chunk_score=0.25)
        assert [c["chunk_text"] for c in result] == ["first", "second"]

    def test_empty_input(self):
        assert filter_chunks_by_score([], min_chunk_score=0.25) == []

    def test_zero_floor_passes_all(self):
        chunks = [_make_chunk(0.0), _make_chunk(0.001)]
        result = filter_chunks_by_score(chunks, min_chunk_score=0.0)
        assert result == chunks

    def test_score_key_missing_treated_as_zero(self):
        chunk_no_score = {"source": "wiki", "chunk_text": "text"}
        result = filter_chunks_by_score([chunk_no_score], min_chunk_score=0.25)
        assert result == []

    def test_web_chunks_not_excluded_when_caller_passes_them(self):
        # Caller must NOT pass web chunks to this function. If they do, web chunks
        # with score=0.0 would be filtered out.  This test documents the expectation
        # that only Pinecone chunks are passed; see graph.py for the separation.
        web_chunk = _make_chunk(0.0, source="web")
        result = filter_chunks_by_score([web_chunk], min_chunk_score=0.25)
        assert result == []  # score=0.0 < 0.25 → filtered; caller must not pass web chunks


# ---------------------------------------------------------------------------
# 2. generate_answer — empty-context guard (Groq MUST NOT be called)
# ---------------------------------------------------------------------------

class TestEmptyContextGuard:
    """
    Tests that when no Pinecone chunks pass the score floor and there are no
    web results, generate_answer returns ABSTENTION_ANSWER deterministically
    WITHOUT invoking the LLM.
    """

    def _call_generate_answer_with_state(
        self,
        retrieved: List[Dict],
        web_results: List[Dict],
        min_chunk_score: float = 0.25,
    ):
        """Call generate_answer with a controlled state, patching settings and LLM."""
        from app.services.chat.graph import generate_answer  # noqa: PLC0415

        mock_llm = MagicMock(name="mock_llm_instance")
        mock_get_llm = MagicMock(return_value=mock_llm)

        state: Dict[str, Any] = {
            "query": "what is RAG?",
            "retrieved": retrieved,
            "web_results": web_results,
            "chat_history": [],
            "timings": {},
        }

        with patch("app.services.chat.graph.get_settings", return_value=_make_mock_settings(min_chunk_score)):
            with patch("app.services.chat.graph.get_llm", mock_get_llm):
                result = generate_answer(state)

        return result, mock_llm

    def test_no_retrieved_no_web_returns_abstention(self):
        result, mock_llm = self._call_generate_answer_with_state(
            retrieved=[], web_results=[]
        )
        assert result["answer"] == ABSTENTION_ANSWER
        assert result["insufficient_context"] is True
        mock_llm.invoke.assert_not_called()

    def test_all_chunks_below_floor_no_web_returns_abstention(self):
        weak_chunks = [_make_chunk(0.1), _make_chunk(0.05)]
        result, mock_llm = self._call_generate_answer_with_state(
            retrieved=weak_chunks, web_results=[], min_chunk_score=0.25
        )
        assert result["answer"] == ABSTENTION_ANSWER
        assert result["insufficient_context"] is True
        mock_llm.invoke.assert_not_called()

    def test_abstention_does_not_call_get_llm_at_all(self):
        """get_llm itself (factory) must not be called on the abstention path."""
        from app.services.chat.graph import generate_answer  # noqa: PLC0415

        state: Dict[str, Any] = {
            "query": "test",
            "retrieved": [_make_chunk(0.05)],
            "web_results": [],
            "chat_history": [],
            "timings": {},
        }
        mock_get_llm = MagicMock(name="mock_get_llm_factory")

        with patch("app.services.chat.graph.get_settings", return_value=_make_mock_settings(0.25)):
            with patch("app.services.chat.graph.get_llm", mock_get_llm):
                result = generate_answer(state)

        mock_get_llm.assert_not_called()
        assert result["insufficient_context"] is True

    def test_abstention_generate_ms_is_zero(self):
        result, _ = self._call_generate_answer_with_state(
            retrieved=[], web_results=[]
        )
        assert result["timings"].get("generate_ms", 0.0) == 0.0

    def test_abstention_state_retrieved_updated_to_empty(self):
        """After filtering, state['retrieved'] must be the filtered (empty) list."""
        weak_chunks = [_make_chunk(0.1)]
        result, _ = self._call_generate_answer_with_state(
            retrieved=weak_chunks, web_results=[], min_chunk_score=0.25
        )
        assert result["retrieved"] == []


# ---------------------------------------------------------------------------
# 3. generate_answer — normal path (Groq IS called when context exists)
# ---------------------------------------------------------------------------

class TestNormalPathWithContext:
    def _make_fake_llm_response(self, text: str = "Here is my answer.") -> MagicMock:
        response = MagicMock()
        response.content = text
        return response

    def test_groq_called_when_chunks_pass_filter(self):
        from app.services.chat.graph import generate_answer  # noqa: PLC0415

        good_chunk = _make_chunk(0.8, text="relevant content")
        state: Dict[str, Any] = {
            "query": "what is RAG?",
            "retrieved": [good_chunk],
            "web_results": [],
            "chat_history": [],
            "timings": {},
        }

        fake_response = self._make_fake_llm_response("RAG is a technique.")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response

        with patch("app.services.chat.graph.get_settings", return_value=_make_mock_settings(0.25)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm):
                result = generate_answer(state)

        mock_llm.invoke.assert_called_once()
        assert result["answer"] == "RAG is a technique."
        assert result["insufficient_context"] is False

    def test_web_results_bypass_filter(self):
        """Web results (no cosine score) reach the LLM even when Pinecone is empty."""
        from app.services.chat.graph import generate_answer  # noqa: PLC0415

        web_result = {"source": "web", "title": "Web page", "url": "https://x.com",
                      "score": 0.0, "chunk_text": "web content"}
        state: Dict[str, Any] = {
            "query": "test",
            "retrieved": [],        # no Pinecone chunks
            "web_results": [web_result],
            "chat_history": [],
            "timings": {},
        }

        fake_response = self._make_fake_llm_response("Web-based answer.")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response

        with patch("app.services.chat.graph.get_settings", return_value=_make_mock_settings(0.25)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm):
                result = generate_answer(state)

        mock_llm.invoke.assert_called_once()
        assert result["insufficient_context"] is False
        assert result["answer"] == "Web-based answer."

    def test_mixed_chunk_scores_only_passing_chunks_reach_llm(self):
        """Chunks below the floor are filtered; LLM is called with only the good ones."""
        from app.services.chat.graph import generate_answer  # noqa: PLC0415
        from app.services.prompts.rag_prompt import build_rag_messages  # noqa: PLC0415

        good = _make_chunk(0.9, text="good chunk")
        bad = _make_chunk(0.05, text="bad chunk")
        state: Dict[str, Any] = {
            "query": "test",
            "retrieved": [good, bad],
            "web_results": [],
            "chat_history": [],
            "timings": {},
        }

        captured_sources: list = []

        def fake_build_rag_messages(chat_history, question, sources):
            captured_sources.extend(sources)
            return build_rag_messages(chat_history, question, sources)

        fake_response = self._make_fake_llm_response("Filtered answer.")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response

        with patch("app.services.chat.graph.get_settings", return_value=_make_mock_settings(0.25)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm):
                with patch("app.services.chat.graph.build_rag_messages", side_effect=fake_build_rag_messages):
                    result = generate_answer(state)

        # Only the good chunk should have reached build_rag_messages
        assert len(captured_sources) == 1
        assert captured_sources[0]["chunk_text"] == "good chunk"
        assert result["retrieved"] == [good]
        assert result["insufficient_context"] is False
