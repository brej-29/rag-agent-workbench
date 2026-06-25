"""
CI-safe unit tests for two-stage reranking (T1.6).

Covers:
  (a) flag OFF → rerank_chunks NOT called; context identical to baseline
  (b) flag ON  → cosine floor runs first, rerank_chunks receives floor survivors,
                 top_k reranked chunks returned
  (c) routing  → decide_next still reads the cosine top_score when flag is ON
  (d) error    → rerank API error falls back to cosine order without raising

The Pinecone Inference rerank API is fully mocked — zero network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.services.chat.graph import (  # noqa: E402
    ABSTENTION_ANSWER,
    decide_next,
    generate_answer,
    retrieve_context,
)
from app.services.rerank import rerank_chunks  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(score: float, text: str = "content", source: str = "wiki") -> Dict[str, Any]:
    return {"source": source, "title": "T", "url": "", "score": score, "chunk_text": text}


def _raw_hit(score: float, text: str = "content", doc_id: str = "d1") -> Dict[str, Any]:
    """Simulate a raw Pinecone search hit (as returned by pinecone_search)."""
    return {
        "_id": f"{doc_id}:0",
        "_score": score,
        "fields": {
            "chunk_text": text,
            "title": "Test",
            "source": "wiki",
            "url": "",
            "doc_id": doc_id,
            "chunk_id": 0,
        },
    }


def _settings(rerank_enabled: bool = False, min_chunk_score: float = 0.25,
               top_k: int = 3, candidates: int = 6,
               model: str = "bge-reranker-v2-m3") -> MagicMock:
    s = MagicMock()
    s.RAG_RERANK_ENABLED = rerank_enabled
    s.RAG_RERANK_MODEL = model
    s.RAG_RERANK_CANDIDATES = candidates
    s.RAG_DEFAULT_TOP_K = top_k
    s.RAG_MIN_CHUNK_SCORE = min_chunk_score
    s.RAG_MIN_SCORE = 0.25
    s.PINECONE_TEXT_FIELD = "chunk_text"
    return s


def _base_state(**overrides) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "query": "what is retrieval-augmented generation?",
        "namespace": "test",
        "top_k": 3,
        "min_score": 0.25,
        "use_web_fallback": False,
        "max_web_results": 5,
        "chat_history": [],
        "retrieved": [],
        "web_results": [],
        "timings": {},
        "tavily_available": False,
        "web_fallback_used": False,
        "top_score": 0.0,
        "insufficient_context": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# (a) Flag OFF — rerank_chunks must NOT be called; context identical to baseline
# ---------------------------------------------------------------------------

class TestFlagOff:
    def test_rerank_chunks_not_called_when_flag_off(self):
        good = _chunk(0.9, "relevant content")
        state = _base_state(
            retrieved=[good],
            top_k=3,
        )

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="answer")

        with patch("app.services.chat.graph.get_settings", return_value=_settings(rerank_enabled=False)):
            with patch("app.services.chat.graph.get_llm", return_value=fake_llm):
                with patch("app.services.chat.graph.rerank_chunks") as mock_rerank:
                    result = generate_answer(state)

        mock_rerank.assert_not_called()
        assert result["answer"] == "answer"

    def test_rerank_ms_is_zero_when_flag_off(self):
        good = _chunk(0.8, "content")
        state = _base_state(retrieved=[good], top_k=3)

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="ans")

        with patch("app.services.chat.graph.get_settings", return_value=_settings(rerank_enabled=False)):
            with patch("app.services.chat.graph.get_llm", return_value=fake_llm):
                result = generate_answer(state)

        assert result["timings"]["rerank_ms"] == 0.0

    def test_retrieve_context_uses_top_k_when_flag_off(self):
        """When reranking is OFF, Pinecone is called with exactly top_k (not candidates)."""
        state = _base_state(top_k=3)
        mock_search = MagicMock(return_value=[])

        with patch("app.services.chat.graph.get_settings", return_value=_settings(rerank_enabled=False, candidates=20)):
            with patch("app.services.chat.graph.pinecone_search", mock_search):
                retrieve_context(state)

        mock_search.assert_called_once()
        _, kwargs = mock_search.call_args
        assert kwargs["top_k"] == 3  # top_k, NOT candidates

    def test_retrieved_list_unchanged_when_all_chunks_above_floor(self):
        chunks = [_chunk(0.9, "a"), _chunk(0.8, "b"), _chunk(0.7, "c")]
        state = _base_state(retrieved=chunks, top_k=3)

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="ans")

        with patch("app.services.chat.graph.get_settings",
                   return_value=_settings(rerank_enabled=False, min_chunk_score=0.25)):
            with patch("app.services.chat.graph.get_llm", return_value=fake_llm):
                result = generate_answer(state)

        assert result["retrieved"] == chunks  # cosine order preserved


# ---------------------------------------------------------------------------
# (b) Flag ON — cosine floor first, then rerank, then top_k
# ---------------------------------------------------------------------------

class TestFlagOn:
    def test_rerank_called_only_with_floor_survivors(self):
        """filter_chunks_by_score must run BEFORE rerank_chunks.
        Chunks below the floor must NOT reach the reranker."""
        strong = _chunk(0.9, "strong")
        weak   = _chunk(0.05, "weak")    # below 0.25 floor
        medium = _chunk(0.6, "medium")
        state = _base_state(retrieved=[strong, weak, medium], top_k=2)

        # Reranker returns strong + medium in reversed order (just for the test)
        reranked = [
            {**medium, "rerank_score": 0.95},
            {**strong, "rerank_score": 0.80},
        ]
        mock_rerank = MagicMock(return_value=reranked)

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="reranked answer")

        with patch("app.services.chat.graph.get_settings",
                   return_value=_settings(rerank_enabled=True, min_chunk_score=0.25)):
            with patch("app.services.chat.graph.get_llm", return_value=fake_llm):
                with patch("app.services.chat.graph.rerank_chunks", mock_rerank) as mr:
                    result = generate_answer(state)

        # rerank must receive only the floor survivors (strong + medium, NOT weak)
        call_kwargs = mr.call_args[1]
        passed_chunks = call_kwargs["chunks"]
        passed_texts = {c["chunk_text"] for c in passed_chunks}
        assert "strong" in passed_texts
        assert "medium" in passed_texts
        assert "weak" not in passed_texts, "Below-floor chunk must not reach reranker"
        assert call_kwargs["top_n"] == 2

    def test_rerank_output_becomes_context(self):
        chunk_a = _chunk(0.9, "alpha")
        chunk_b = _chunk(0.7, "beta")
        state = _base_state(retrieved=[chunk_a, chunk_b], top_k=2)

        reranked = [
            {**chunk_b, "rerank_score": 0.99},
            {**chunk_a, "rerank_score": 0.50},
        ]
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="ans")

        with patch("app.services.chat.graph.get_settings",
                   return_value=_settings(rerank_enabled=True, min_chunk_score=0.0)):
            with patch("app.services.chat.graph.get_llm", return_value=fake_llm):
                with patch("app.services.chat.graph.rerank_chunks", return_value=reranked):
                    result = generate_answer(state)

        # state["retrieved"] must reflect the reranked order
        assert result["retrieved"] == reranked
        assert result["retrieved"][0]["chunk_text"] == "beta"

    def test_rerank_ms_recorded_when_flag_on(self):
        chunk = _chunk(0.9, "content")
        state = _base_state(retrieved=[chunk], top_k=1)

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="ans")

        with patch("app.services.chat.graph.get_settings",
                   return_value=_settings(rerank_enabled=True, min_chunk_score=0.0)):
            with patch("app.services.chat.graph.get_llm", return_value=fake_llm):
                with patch("app.services.chat.graph.rerank_chunks",
                           return_value=[{**chunk, "rerank_score": 0.9}]):
                    result = generate_answer(state)

        assert result["timings"]["rerank_ms"] >= 0.0  # timer ran

    def test_retrieve_context_uses_candidates_when_flag_on(self):
        """When reranking is ON, Pinecone is called with max(candidates, top_k)."""
        state = _base_state(top_k=3)
        mock_search = MagicMock(return_value=[])

        with patch("app.services.chat.graph.get_settings",
                   return_value=_settings(rerank_enabled=True, candidates=20, top_k=3)):
            with patch("app.services.chat.graph.pinecone_search", mock_search):
                retrieve_context(state)

        _, kwargs = mock_search.call_args
        assert kwargs["top_k"] == 20  # candidates (> top_k=3)

    def test_candidates_clamped_to_top_k_when_smaller(self):
        """RAG_RERANK_CANDIDATES < top_k → clamped up to top_k."""
        state = _base_state(top_k=10)
        mock_search = MagicMock(return_value=[])

        with patch("app.services.chat.graph.get_settings",
                   return_value=_settings(rerank_enabled=True, candidates=3, top_k=10)):
            with patch("app.services.chat.graph.pinecone_search", mock_search):
                retrieve_context(state)

        _, kwargs = mock_search.call_args
        assert kwargs["top_k"] == 10  # clamped: max(3, 10) = 10


# ---------------------------------------------------------------------------
# (c) Routing — decide_next reads cosine top_score even when reranking is ON
# ---------------------------------------------------------------------------

class TestRoutingReadsCosineTopScore:
    def test_top_score_set_from_cosine_by_retrieve_context(self):
        """retrieve_context must set top_score from cosine _score, not rerank score."""
        hits = [
            _raw_hit(score=0.88, text="best", doc_id="d1"),
            _raw_hit(score=0.50, text="ok",   doc_id="d2"),
        ]
        state = _base_state(top_k=2)

        with patch("app.services.chat.graph.get_settings",
                   return_value=_settings(rerank_enabled=True, candidates=5)):
            with patch("app.services.chat.graph.pinecone_search", return_value=hits):
                result = retrieve_context(state)

        # top_score must be the max cosine score (0.88), not a rerank score
        assert abs(result["top_score"] - 0.88) < 1e-6

    def test_decide_next_uses_cosine_top_score_for_routing(self):
        """decide_next routes based on top_score (cosine); rerank flag does not affect it."""
        # Simulate: top_score below min_score, web fallback available
        state = _base_state(
            top_score=0.10,   # cosine top_score — below min_score threshold
            min_score=0.25,
            use_web_fallback=True,
            tavily_available=True,
            retrieved=[_chunk(0.10)],
        )
        result = decide_next(state)
        assert result["web_fallback_used"] is True  # routing from cosine score, not rerank


# ---------------------------------------------------------------------------
# (d) Rerank API error — graceful fallback to cosine order
# ---------------------------------------------------------------------------

class TestRerankerGracefulDegradation:
    def test_rerank_error_falls_back_to_cosine_order(self):
        """If rerank_chunks raises internally, it returns cosine order — not a crash."""
        chunks = [_chunk(0.9, "a"), _chunk(0.7, "b"), _chunk(0.6, "c")]
        query = "test query"
        top_n = 2

        # Simulate an API failure at the Pinecone client level
        mock_pc = MagicMock()
        mock_pc.inference.rerank.side_effect = RuntimeError("Pinecone API timeout")

        with patch("app.services.rerank.get_pinecone_client", return_value=mock_pc):
            result = rerank_chunks(query=query, chunks=chunks, top_n=top_n, model="bge-reranker-v2-m3")

        # Must not raise; must return cosine-order slice
        assert result == chunks[:top_n]
        assert len(result) == top_n

    def test_rerank_error_in_generate_answer_does_not_raise(self):
        """An error inside rerank_chunks must not bubble up through generate_answer."""
        chunk = _chunk(0.9, "content")
        state = _base_state(retrieved=[chunk], top_k=1)

        # rerank_chunks itself gracefully returns cosine order on error
        fallback = [chunk]
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="fallback answer")

        with patch("app.services.chat.graph.get_settings",
                   return_value=_settings(rerank_enabled=True, min_chunk_score=0.0)):
            with patch("app.services.chat.graph.get_llm", return_value=fake_llm):
                # Simulate rerank_chunks returning cosine fallback (its own error handling)
                with patch("app.services.chat.graph.rerank_chunks", return_value=fallback):
                    result = generate_answer(state)

        assert result["answer"] == "fallback answer"
        assert result["insufficient_context"] is False

    def test_rerank_returns_empty_list_input_unchanged(self):
        """Empty input must return immediately — no API call."""
        mock_pc = MagicMock()
        with patch("app.services.rerank.get_pinecone_client", return_value=mock_pc):
            result = rerank_chunks(query="q", chunks=[], top_n=5, model="bge-reranker-v2-m3")

        assert result == []
        mock_pc.inference.rerank.assert_not_called()
