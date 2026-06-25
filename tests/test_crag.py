"""
Tests for the CRAG corrective retrieval module (T2.4).

All tests are CI-safe: zero network calls, no Pinecone/Groq/Tavily credentials.
LLM and search dependencies are mocked via unittest.mock.

Critical invariant covered (test_max_iters_guard_always_terminates):
  The CRAG loop MUST terminate after RAG_CRAG_MAX_ITERS iterations regardless
  of the grade outcome.  This is the mandatory guard against runaway loops.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.crag import grade_retrieval, rewrite_query
from app.services.chat.graph import corrective_retrieve


# ---------------------------------------------------------------------------
# grade_retrieval — pure function, no mocking needed
# ---------------------------------------------------------------------------

class TestGradeRetrieval:
    def test_good_above_threshold(self):
        assert grade_retrieval(0.5, 0.45) == "good"

    def test_good_at_exact_threshold(self):
        assert grade_retrieval(0.45, 0.45) == "good"

    def test_weak_below_threshold(self):
        assert grade_retrieval(0.3, 0.45) == "weak"

    def test_weak_at_zero(self):
        assert grade_retrieval(0.0, 0.45) == "weak"

    def test_weak_when_threshold_is_one(self):
        assert grade_retrieval(0.99, 1.0) == "weak"


# ---------------------------------------------------------------------------
# rewrite_query — pure function with a mock LLM
# ---------------------------------------------------------------------------

class TestRewriteQuery:
    def test_returns_rewritten_text(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="better query about vectors")
        result = rewrite_query("what is retrieval", mock_llm)
        assert result == "better query about vectors"

    def test_strips_whitespace(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="  trimmed query  ")
        assert rewrite_query("original", mock_llm) == "trimmed query"

    def test_fallback_on_empty_content(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="   ")
        assert rewrite_query("original query", mock_llm) == "original query"

    def test_fallback_on_llm_exception(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        assert rewrite_query("original query", mock_llm) == "original query"

    def test_fallback_when_response_has_no_content_attr(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ""   # no .content attribute, empty string
        assert rewrite_query("original query", mock_llm) == "original query"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(top_score: float = 0.0, retrieved=None):
    return {
        "query": "test query",
        "namespace": "ns",
        "top_k": 5,
        "top_score": top_score,
        "retrieved": retrieved or [],
        "timings": {},
    }


def _mock_settings(enabled=True, max_iters=2, good_score=0.45):
    s = MagicMock()
    s.RAG_CRAG_ENABLED = enabled
    s.RAG_CRAG_MAX_ITERS = max_iters
    s.RAG_CRAG_GOOD_SCORE = good_score
    s.PINECONE_TEXT_FIELD = "chunk_text"
    s.RAG_DEFAULT_TOP_K = 5
    return s


def _weak_hit(score=0.3):
    return {
        "_score": score,
        "fields": {"chunk_text": "weak content", "source": "s", "title": "t", "url": ""},
    }


# ---------------------------------------------------------------------------
# corrective_retrieve — graph node
# ---------------------------------------------------------------------------

class TestCorrectiveRetrieve:

    def test_flag_off_is_exact_passthrough(self):
        """When CRAG is OFF, state is returned unchanged with no CRAG keys added."""
        state = _base_state(top_score=0.1)
        with patch("app.services.chat.graph.get_settings",
                   return_value=_mock_settings(enabled=False)):
            result = corrective_retrieve(state)
        # The function must NOT add crag_iterations or corrective_action
        assert "crag_iterations" not in result
        assert "corrective_action" not in result
        # State is otherwise identical
        assert result["top_score"] == 0.1
        assert result["retrieved"] == []

    def test_good_grade_no_corrective_action(self):
        """CRAG ON, top_score above threshold: no LLM call, no re-retrieval, crag_iterations=0."""
        state = _base_state(top_score=0.6)
        with patch("app.services.chat.graph.get_settings",
                   return_value=_mock_settings(good_score=0.45)):
            with patch("app.services.chat.graph.get_llm") as mock_get_llm:
                with patch("app.services.chat.graph.pinecone_search") as mock_search:
                    result = corrective_retrieve(state)
        assert result["crag_iterations"] == 0
        assert result["corrective_action"] is None
        mock_get_llm.assert_not_called()
        mock_search.assert_not_called()

    def test_weak_grade_fires_correction_and_updates_state(self):
        """CRAG ON, top_score below threshold: query rewritten, state updated from re-retrieval."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="rewritten query")
        good_hit = {
            "_score": 0.7,
            "fields": {"chunk_text": "relevant content", "source": "src", "title": "t", "url": ""},
        }
        state = _base_state(top_score=0.2)
        with patch("app.services.chat.graph.get_settings",
                   return_value=_mock_settings(good_score=0.45, max_iters=2)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm_instance):
                with patch("app.services.chat.graph.pinecone_search", return_value=[good_hit]):
                    result = corrective_retrieve(state)
        assert result["crag_iterations"] == 1
        assert result["corrective_action"] == "rewrite"
        assert result["top_score"] == 0.7
        assert len(result["retrieved"]) == 1
        assert result["retrieved"][0]["chunk_text"] == "relevant content"

    def test_max_iters_guard_always_terminates(self):
        """CRITICAL: loop terminates after exactly RAG_CRAG_MAX_ITERS iterations.

        The threshold is set to 0.99 (unreachably high) so retrieval is always graded
        weak.  The loop MUST still stop after max_iters.  This is the mandatory guard
        against runaway loops identified in the audit.
        """
        max_iters = 2
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="rewritten")
        state = _base_state(top_score=0.1)
        with patch("app.services.chat.graph.get_settings",
                   return_value=_mock_settings(max_iters=max_iters, good_score=0.99)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm_instance):
                with patch("app.services.chat.graph.pinecone_search",
                           return_value=[_weak_hit(0.3)]) as mock_ps:
                    result = corrective_retrieve(state)
        # Loop ran exactly max_iters times — no more
        assert result["crag_iterations"] == max_iters
        assert mock_ps.call_count == max_iters
        # LLM was called once per corrective iteration
        assert mock_llm_instance.invoke.call_count == max_iters

    def test_exhausted_iterations_leave_empty_context_for_abstention(self):
        """After max_iters with no results, retrieved is empty.

        The empty-context guard in generate_answer will then set
        insufficient_context=True and return the deterministic abstention.
        corrective_retrieve must NOT mask this by filling retrieved with stale data.
        """
        state = _base_state(top_score=0.0)
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="rewritten")
        with patch("app.services.chat.graph.get_settings",
                   return_value=_mock_settings(max_iters=1, good_score=0.99)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm_instance):
                with patch("app.services.chat.graph.pinecone_search", return_value=[]):
                    result = corrective_retrieve(state)
        assert result["retrieved"] == []
        assert result["top_score"] == 0.0
        assert result["crag_iterations"] == 1

    def test_max_iters_one_terminates_after_single_attempt(self):
        """Edge case: max_iters=1 means exactly one corrective attempt, then stop."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="rewritten")
        state = _base_state(top_score=0.1)
        with patch("app.services.chat.graph.get_settings",
                   return_value=_mock_settings(max_iters=1, good_score=0.99)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm_instance):
                with patch("app.services.chat.graph.pinecone_search",
                           return_value=[_weak_hit(0.2)]) as mock_ps:
                    result = corrective_retrieve(state)
        assert result["crag_iterations"] == 1
        assert mock_ps.call_count == 1

    def test_second_iteration_uses_rewritten_query(self):
        """When the first correction is still weak, the second iteration re-rewrites."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = [
            MagicMock(content="first rewrite"),
            MagicMock(content="second rewrite"),
        ]
        # First re-retrieval: still weak; second: good
        good_hit = {
            "_score": 0.8,
            "fields": {"chunk_text": "great content", "source": "s", "title": "t", "url": ""},
        }
        state = _base_state(top_score=0.1)
        with patch("app.services.chat.graph.get_settings",
                   return_value=_mock_settings(max_iters=3, good_score=0.45)):
            with patch("app.services.chat.graph.get_llm", return_value=mock_llm_instance):
                with patch("app.services.chat.graph.pinecone_search",
                           side_effect=[[_weak_hit(0.2)], [good_hit]]) as mock_ps:
                    result = corrective_retrieve(state)
        assert result["crag_iterations"] == 2
        assert result["top_score"] == 0.8
        assert mock_ps.call_count == 2
        # Both iterations rewrote the query
        assert mock_llm_instance.invoke.call_count == 2
