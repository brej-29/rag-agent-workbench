"""
CI-safe unit tests for T2.9 — true generation-token streaming.

Coverage (T9.5):
  (a) True-streaming path yields multiple token events (not one post-split blob),
      then a final 'done' metadata event.
  (b) Cached-hit path emits the cached answer + 'done' with cached=True,
      and does NOT call the streaming LLM.
  (c) Abstention path emits the abstention text + 'done' with
      insufficient_context=True; the streaming LLM is NOT called.
  (d) The final 'done' event contains grounding (T2.3) + usage/cost (T2.7)
      + CRAG/contextualize (T2.4/T2.5) + timing (T2.6) fields.
  (e) Generation error emits an 'error' event without hanging; no 'done' follows.

All tests make ZERO network calls — the LLM, Pinecone, and pipeline nodes
are fully mocked.  asyncio.run() is used to drive the async generator without
requiring pytest-asyncio.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.services.chat.streaming import (
    ABSTENTION_ANSWER,
    _build_done_payload,
    _sse,
    stream_chat_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect(gen_coro) -> List[str]:
    """Collect all SSE frames from the async generator in a fresh event loop."""
    async def _run():
        return [frame async for frame in gen_coro]
    return asyncio.run(_run())


def _parse(frame: str) -> Tuple[str, Any]:
    """Parse 'event: X\\ndata: Y\\n\\n' into (event, data)."""
    event: str = ""
    data: Any = None
    for line in frame.strip().split("\n"):
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: "):
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                data = line[6:]
    return event, data


def _initial_state() -> Dict:
    return {
        "query": "What is RAG?",
        "namespace": "dev",
        "top_k": 5,
        "use_web_fallback": False,
        "min_score": 0.25,
        "max_web_results": 5,
        "chat_history": [],
    }


def _pre_gen_state() -> Dict:
    """Simulated state returned by _run_pre_generation_pipeline."""
    return {
        "query": "What is RAG?",
        "namespace": "dev",
        "top_k": 5,
        "retrieved": [
            {"source": "wiki", "title": "RAG", "url": "", "score": 0.85, "chunk_text": "RAG is..."}
        ],
        "web_results": [],
        "timings": {"retrieve_ms": 80.0, "rerank_ms": 0.0, "web_ms": 0.0},
        "tavily_available": False,
        "web_fallback_used": False,
        "top_score": 0.85,
        "token_usage_by_call": {},
        "insufficient_context": False,
        "crag_iterations": 0,
        "corrective_action": None,
        "contextualized_query": None,
        "use_web_fallback": False,
        "min_score": 0.25,
        "max_web_results": 5,
        "chat_history": [],
    }


def _mock_post_gen(state: Dict) -> Dict:
    """Simulate format_response: sets grounding fields."""
    state["grounded"] = True
    state["faithfulness_score"] = 0.92
    state["unverified_citations"] = []
    timings = state.get("timings") or {}
    timings["faithfulness_ms"] = 30.0
    state["timings"] = timings
    return state


def _mock_post_gen_abstention(state: Dict) -> Dict:
    state["grounded"] = None
    state["faithfulness_score"] = None
    state["unverified_citations"] = []
    timings = state.get("timings") or {}
    timings["faithfulness_ms"] = 0.0
    state["timings"] = timings
    return state


# Async generator mock for the streaming LLM.
# Returns 3 distinct chunks so we can verify multiple token events.
async def _astream_three_chunks(*args, **kwargs):
    for token in ["Hello", " world", "!"]:
        chunk = MagicMock()
        chunk.content = token
        yield chunk


# Async generator that raises immediately — simulates LLM error.
async def _astream_raise(*args, **kwargs):
    raise RuntimeError("Groq connection error")
    yield  # makes this an async generator (never reached)


def _mock_llm(astream_fn=None):
    llm = MagicMock()
    llm.astream = astream_fn or _astream_three_chunks
    return llm


def _mock_cached_response():
    cached = MagicMock()
    cached.answer = "Cached answer"
    cached.sources = []
    cached.timings = MagicMock()
    cached.timings.model_dump.return_value = {
        "retrieve_ms": 100.0, "rerank_ms": 0.0, "web_ms": 0.0,
        "generate_ms": 200.0, "faithfulness_ms": 10.0, "total_ms": 310.0,
    }
    cached.trace = MagicMock()
    cached.trace.model_dump.return_value = {"trace_enabled": False, "langsmith_project": None}
    cached.insufficient_context = False
    cached.grounded = True
    cached.faithfulness_score = 0.88
    cached.unverified_citations = []
    cached.crag_iterations = 0
    cached.corrective_action = None
    cached.contextualized_query = None
    cached.usage = None
    return cached


# ---------------------------------------------------------------------------
# (a) True-streaming path: multiple token events, then done
# ---------------------------------------------------------------------------


class TestTrueStreamingTokenEvents:

    def test_multiple_token_events_not_one_blob(self):
        """Verify 3 separate token events (one per chunk), not one post-split blob."""
        pre_gen = _pre_gen_state()
        mock_messages = [MagicMock()]

        with (
            patch("app.services.chat.streaming._run_pre_generation_pipeline", return_value=pre_gen),
            patch("app.services.chat.streaming._prepare_generation_inputs",
                  return_value=(False, pre_gen["retrieved"], mock_messages)),
            patch("app.services.chat.streaming.get_llm", return_value=_mock_llm()),
            patch("app.services.chat.streaming._run_post_generation", side_effect=_mock_post_gen),
        ):
            frames = _collect(stream_chat_response(_initial_state(), {}))

        token_events = [_parse(f) for f in frames if _parse(f)[0] == "token"]
        assert len(token_events) == 3, (
            f"Expected 3 token events (one per chunk), got {len(token_events)}"
        )
        texts = [ev[1]["text"] for ev in token_events]
        assert texts == ["Hello", " world", "!"]

    def test_final_event_is_done(self):
        pre_gen = _pre_gen_state()
        mock_messages = [MagicMock()]

        with (
            patch("app.services.chat.streaming._run_pre_generation_pipeline", return_value=pre_gen),
            patch("app.services.chat.streaming._prepare_generation_inputs",
                  return_value=(False, pre_gen["retrieved"], mock_messages)),
            patch("app.services.chat.streaming.get_llm", return_value=_mock_llm()),
            patch("app.services.chat.streaming._run_post_generation", side_effect=_mock_post_gen),
        ):
            frames = _collect(stream_chat_response(_initial_state(), {}))

        last_event, _ = _parse(frames[-1])
        assert last_event == "done"

    def test_assembled_answer_in_done_event(self):
        pre_gen = _pre_gen_state()
        mock_messages = [MagicMock()]

        with (
            patch("app.services.chat.streaming._run_pre_generation_pipeline", return_value=pre_gen),
            patch("app.services.chat.streaming._prepare_generation_inputs",
                  return_value=(False, pre_gen["retrieved"], mock_messages)),
            patch("app.services.chat.streaming.get_llm", return_value=_mock_llm()),
            patch("app.services.chat.streaming._run_post_generation", side_effect=_mock_post_gen),
        ):
            frames = _collect(stream_chat_response(_initial_state(), {}))

        _, done_data = _parse(frames[-1])
        assert done_data["answer"] == "Hello world!"

    def test_token_events_precede_done_event(self):
        pre_gen = _pre_gen_state()
        mock_messages = [MagicMock()]

        with (
            patch("app.services.chat.streaming._run_pre_generation_pipeline", return_value=pre_gen),
            patch("app.services.chat.streaming._prepare_generation_inputs",
                  return_value=(False, pre_gen["retrieved"], mock_messages)),
            patch("app.services.chat.streaming.get_llm", return_value=_mock_llm()),
            patch("app.services.chat.streaming._run_post_generation", side_effect=_mock_post_gen),
        ):
            frames = _collect(stream_chat_response(_initial_state(), {}))

        events = [_parse(f)[0] for f in frames]
        done_idx = events.index("done")
        assert all(e == "token" for e in events[:done_idx])


# ---------------------------------------------------------------------------
# (b) Cached response path: no LLM call, cached=True in done
# ---------------------------------------------------------------------------


class TestCachedResponsePath:

    def test_cached_path_no_llm_call(self):
        """Cache hit must NOT call the streaming LLM."""
        cached = _mock_cached_response()

        with patch("app.services.chat.streaming.get_llm") as mock_get_llm:
            frames = _collect(
                stream_chat_response(
                    _initial_state(), {}, use_cache=True, cached_response=cached
                )
            )

        mock_get_llm.assert_not_called()

    def test_cached_path_emits_cached_answer(self):
        cached = _mock_cached_response()

        frames = _collect(
            stream_chat_response(
                _initial_state(), {}, use_cache=True, cached_response=cached
            )
        )

        _, token_data = _parse(frames[0])
        assert token_data["text"] == "Cached answer"

    def test_cached_path_done_event_cached_true(self):
        cached = _mock_cached_response()

        frames = _collect(
            stream_chat_response(
                _initial_state(), {}, use_cache=True, cached_response=cached
            )
        )

        _, done_data = _parse(frames[-1])
        assert done_data["cached"] is True

    def test_cached_path_exactly_two_events(self):
        """Cached path: exactly one token event and one done event."""
        cached = _mock_cached_response()

        frames = _collect(
            stream_chat_response(
                _initial_state(), {}, use_cache=True, cached_response=cached
            )
        )

        events = [_parse(f)[0] for f in frames]
        assert events == ["token", "done"]


# ---------------------------------------------------------------------------
# (c) Abstention path: no LLM call, insufficient_context=True
# ---------------------------------------------------------------------------


class TestAbstentionPath:

    def test_abstention_no_llm_call(self):
        """On abstention, the streaming LLM must NOT be called."""
        pre_gen = _pre_gen_state()

        with (
            patch("app.services.chat.streaming._run_pre_generation_pipeline", return_value=pre_gen),
            patch("app.services.chat.streaming._prepare_generation_inputs",
                  return_value=(True, [], [])),
            patch("app.services.chat.streaming.get_llm") as mock_get_llm,
            patch("app.services.chat.streaming._run_post_generation",
                  side_effect=_mock_post_gen_abstention),
        ):
            frames = _collect(stream_chat_response(_initial_state(), {}))

        mock_get_llm.assert_not_called()

    def test_abstention_token_event_contains_abstention_text(self):
        pre_gen = _pre_gen_state()

        with (
            patch("app.services.chat.streaming._run_pre_generation_pipeline", return_value=pre_gen),
            patch("app.services.chat.streaming._prepare_generation_inputs",
                  return_value=(True, [], [])),
            patch("app.services.chat.streaming.get_llm"),
            patch("app.services.chat.streaming._run_post_generation",
                  side_effect=_mock_post_gen_abstention),
        ):
            frames = _collect(stream_chat_response(_initial_state(), {}))

        _, token_data = _parse(frames[0])
        # state has no "answer" key → falls back to ABSTENTION_ANSWER constant
        assert ABSTENTION_ANSWER in token_data["text"]

    def test_abstention_done_insufficient_context_true(self):
        pre_gen = _pre_gen_state()

        def _post_gen_sets_insufficient(state):
            state["insufficient_context"] = True
            state["grounded"] = None
            state["faithfulness_score"] = None
            state["unverified_citations"] = []
            timings = state.get("timings") or {}
            timings["faithfulness_ms"] = 0.0
            state["timings"] = timings
            return state

        with (
            patch("app.services.chat.streaming._run_pre_generation_pipeline", return_value=pre_gen),
            patch("app.services.chat.streaming._prepare_generation_inputs",
                  return_value=(True, [], [])),
            patch("app.services.chat.streaming.get_llm"),
            patch("app.services.chat.streaming._run_post_generation",
                  side_effect=_post_gen_sets_insufficient),
        ):
            frames = _collect(stream_chat_response(_initial_state(), {}))

        _, done_data = _parse(frames[-1])
        assert done_data["insufficient_context"] is True


# ---------------------------------------------------------------------------
# (d) Final 'done' event contains grounding + usage + timing fields
# ---------------------------------------------------------------------------


class TestDoneEventFields:

    def _run_normal_path(self, post_gen_fn=None):
        pre_gen = _pre_gen_state()
        pre_gen["token_usage_by_call"] = {}
        mock_messages = [MagicMock()]

        def _default_post_gen(state):
            state["grounded"] = True
            state["faithfulness_score"] = 0.95
            state["unverified_citations"] = []
            state["crag_iterations"] = 1
            state["corrective_action"] = "rewrite"
            state["contextualized_query"] = "What is retrieval-augmented generation?"
            state["token_usage_by_call"] = {
                "generation": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
                "judge": {"prompt_tokens": 60, "completion_tokens": 20, "total_tokens": 80},
            }
            timings = state.get("timings") or {}
            timings["faithfulness_ms"] = 45.0
            state["timings"] = timings
            return state

        fn = post_gen_fn or _default_post_gen

        with (
            patch("app.services.chat.streaming._run_pre_generation_pipeline", return_value=pre_gen),
            patch("app.services.chat.streaming._prepare_generation_inputs",
                  return_value=(False, pre_gen["retrieved"], mock_messages)),
            patch("app.services.chat.streaming.get_llm", return_value=_mock_llm()),
            patch("app.services.chat.streaming._run_post_generation", side_effect=fn),
        ):
            frames = _collect(stream_chat_response(_initial_state(), {}))

        _, done_data = _parse(frames[-1])
        return done_data

    def test_done_has_grounded_field(self):
        done = self._run_normal_path()
        assert "grounded" in done
        assert done["grounded"] is True

    def test_done_has_faithfulness_score(self):
        done = self._run_normal_path()
        assert "faithfulness_score" in done
        assert done["faithfulness_score"] == pytest.approx(0.95)

    def test_done_has_usage_with_totals(self):
        done = self._run_normal_path()
        assert done["usage"] is not None
        assert done["usage"]["total_tokens"] == 220  # 140 + 80
        assert done["usage"]["prompt_tokens"] == 160
        assert done["usage"]["completion_tokens"] == 60

    def test_done_has_crag_fields(self):
        done = self._run_normal_path()
        assert done["crag_iterations"] == 1
        assert done["corrective_action"] == "rewrite"

    def test_done_has_contextualized_query(self):
        done = self._run_normal_path()
        assert done["contextualized_query"] == "What is retrieval-augmented generation?"

    def test_done_has_timings(self):
        done = self._run_normal_path()
        timings = done["timings"]
        assert "retrieve_ms" in timings
        assert "generate_ms" in timings
        assert "faithfulness_ms" in timings
        assert "total_ms" in timings
        assert timings["faithfulness_ms"] == pytest.approx(45.0)

    def test_done_cached_false_on_normal_path(self):
        done = self._run_normal_path()
        assert done["cached"] is False

    def test_done_has_unverified_citations(self):
        done = self._run_normal_path()
        assert "unverified_citations" in done
        assert done["unverified_citations"] == []


# ---------------------------------------------------------------------------
# (e) Generation error: error event emitted, no done, no hang
# ---------------------------------------------------------------------------


class TestGenerationError:

    def test_generation_error_emits_error_event(self):
        pre_gen = _pre_gen_state()
        mock_messages = [MagicMock()]
        mock_llm = _mock_llm(astream_fn=_astream_raise)

        with (
            patch("app.services.chat.streaming._run_pre_generation_pipeline", return_value=pre_gen),
            patch("app.services.chat.streaming._prepare_generation_inputs",
                  return_value=(False, pre_gen["retrieved"], mock_messages)),
            patch("app.services.chat.streaming.get_llm", return_value=mock_llm),
        ):
            frames = _collect(stream_chat_response(_initial_state(), {}))

        assert len(frames) == 1
        event, data = _parse(frames[0])
        assert event == "error"
        assert "message" in data

    def test_generation_error_no_done_event(self):
        pre_gen = _pre_gen_state()
        mock_messages = [MagicMock()]
        mock_llm = _mock_llm(astream_fn=_astream_raise)

        with (
            patch("app.services.chat.streaming._run_pre_generation_pipeline", return_value=pre_gen),
            patch("app.services.chat.streaming._prepare_generation_inputs",
                  return_value=(False, pre_gen["retrieved"], mock_messages)),
            patch("app.services.chat.streaming.get_llm", return_value=mock_llm),
        ):
            frames = _collect(stream_chat_response(_initial_state(), {}))

        events = [_parse(f)[0] for f in frames]
        assert "done" not in events

    def test_pre_generation_error_emits_error_event(self):
        """If the pre-generation pipeline raises, an error event is emitted."""
        with patch(
            "app.services.chat.streaming._run_pre_generation_pipeline",
            side_effect=RuntimeError("Pinecone down"),
        ):
            frames = _collect(stream_chat_response(_initial_state(), {}))

        assert len(frames) == 1
        event, data = _parse(frames[0])
        assert event == "error"
        assert "Pinecone down" in data["message"]
