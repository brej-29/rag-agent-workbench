"""
Integration tests: real FastAPI app through TestClient, only external
service boundaries mocked (Pinecone, Groq, Tavily).

WHY THESE ARE INTEGRATION TESTS, NOT UNIT TESTS
The 321 existing tests call functions in isolation with mocked collaborators.
These tests drive the REAL app via HTTP: a real POST hits the real router →
auth dependency → rate-limit middleware → LangGraph pipeline (graph.invoke)
→ prompt builders → real response schema.  Wiring bugs, schema mismatches,
and middleware ordering errors that unit tests structurally cannot catch are
caught here.

WHAT IS MOCKED (external network only)
  - pinecone_search  — real hit dict shape that graph.retrieve_context reads
  - get_llm()        — returns a mock LLM whose .invoke() / .astream() shapes
                       match what graph.generate_answer and streaming.py read
  - is_tavily_configured — False (web fallback disabled in all test requests)

WHAT RUNS FOR REAL (everything internal)
  HTTP routing, require_api_key auth, LangGraph graph.invoke(), all graph
  nodes (normalize_input, retrieve_context, generate_answer, format_response,
  …), build_rag_messages, filter_chunks_by_score, verify_citations,
  ChatResponse schema, and the SSE frame encoder.

RUNNING
  `pytest tests/integration/ -v`  — zero network, zero creds needed.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.integration.conftest import INTEGRATION_API_KEY

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

_AUTH_HEADERS = {"X-API-Key": INTEGRATION_API_KEY}

# Realistic Pinecone hit shape — matches what graph.retrieve_context reads:
#   hit_score = float(hit.get("_score") or hit.get("score") or 0.0)
#   fields = hit.get("fields") or {}
#   raw_text = fields.get(text_field, "")
_FAKE_CHUNK = {
    "_score": 0.92,
    "fields": {
        "chunk_text": "RAG combines retrieval with generation to answer questions accurately.",
        "title": "Retrieval-Augmented Generation",
        "source": "wiki",
        "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    },
}

# Below default RAG_MIN_CHUNK_SCORE (0.20) → triggers empty-context abstention
_FAKE_CHUNK_LOW_SCORE = {
    "_score": 0.04,
    "fields": {
        "chunk_text": "Unrelated text with very low relevance score.",
        "title": "Unrelated Document",
        "source": "wiki",
        "url": "",
    },
}

_CHAT_PAYLOAD: dict = {
    "query": "What is retrieval-augmented generation?",
    "namespace": "integ-test",
    "top_k": 3,
    "use_web_fallback": False,
    "min_score": 0.20,
}

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


async def _fake_astream(messages, **kwargs):
    """Async generator matching llm.astream() — yields MagicMock chunks with .content."""
    for token in ["RAG ", "combines ", "retrieval ", "with generation. [1]"]:
        chunk = MagicMock()
        chunk.content = token
        yield chunk


def _make_llm_response(
    answer: str = "RAG combines retrieval with generation. [1]",
    prompt_tokens: int = 150,
    completion_tokens: int = 30,
) -> MagicMock:
    """Realistic Groq/LangChain AIMessage shape that extract_token_usage reads."""
    resp = MagicMock()
    resp.content = answer
    # Path 1 in extract_token_usage: response.usage_metadata with input_tokens/output_tokens
    resp.usage_metadata = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    resp.response_metadata = {}
    return resp


def _make_mock_llm(
    answer: str = "RAG combines retrieval with generation. [1]",
    prompt_tokens: int = 150,
    completion_tokens: int = 30,
) -> MagicMock:
    """Return a mock LLM with realistic invoke() and astream() shapes."""
    mock = MagicMock()
    mock.invoke.return_value = _make_llm_response(answer, prompt_tokens, completion_tokens)
    mock.astream = _fake_astream
    return mock


def _parse_sse(body: str) -> list[tuple[str, Any]]:
    """Parse an SSE response body into [(event, data), ...].

    SSE format: 'event: <name>\\ndata: <json>\\n\\n' blocks.
    """
    frames = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event, data = "", None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    data = line[6:]
        if event:
            frames.append((event, data))
    return frames


# ---------------------------------------------------------------------------
# P1.2 — Health and metrics (no external calls needed)
# ---------------------------------------------------------------------------


class TestHealthAndMetrics:
    def test_health_returns_200_with_status_ok(self, integration_client):
        """/health is public and requires no auth."""
        resp = integration_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "service" in body
        assert "version" in body

    def test_metrics_endpoint_returns_200_with_valid_auth(self, integration_client):
        """/metrics is auth-gated; the real require_api_key dependency runs."""
        resp = integration_client.get("/metrics", headers=_AUTH_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# P1.2 — Auth: real require_api_key dependency (not mocked)
# ---------------------------------------------------------------------------


class TestAuth:
    def test_missing_api_key_header_returns_403(self, integration_client):
        """No X-API-Key header → real auth dependency → 403."""
        resp = integration_client.post("/chat", json=_CHAT_PAYLOAD)
        assert resp.status_code == 403

    def test_invalid_api_key_returns_403(self, integration_client):
        """Wrong key value → real auth dependency → 403."""
        resp = integration_client.post(
            "/chat",
            json=_CHAT_PAYLOAD,
            headers={"X-API-Key": "wrong-key-value"},
        )
        assert resp.status_code == 403

    def test_valid_api_key_passes_auth(self, integration_client):
        """Correct key → auth passes; pipeline runs; 200 returned."""
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# P1.2 — /chat happy path: real router → real graph → real schema
# ---------------------------------------------------------------------------


class TestChatHappyPath:
    """Drive POST /chat through the real pipeline with mocked externals."""

    def test_returns_200(self, integration_client):
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)
        assert resp.status_code == 200

    def test_response_schema_has_all_required_fields(self, integration_client):
        """Every ChatResponse field is present — wiring from graph state to schema."""
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)

        body = resp.json()
        # Core fields
        for field in ("answer", "sources", "timings", "trace", "insufficient_context"):
            assert field in body, f"missing field: {field}"
        # Observability fields (T2.x)
        for field in (
            "grounded", "faithfulness_score", "unverified_citations",
            "crag_iterations", "corrective_action", "contextualized_query", "usage",
        ):
            assert field in body, f"missing observability field: {field}"

    def test_answer_is_the_mocked_llm_content(self, integration_client):
        """Answer flows from mock_llm.invoke().content through graph state to schema."""
        mock_llm = _make_mock_llm(answer="Mocked integration answer. [1]")
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)

        assert resp.json()["answer"] == "Mocked integration answer. [1]"

    def test_sources_populated_from_mocked_pinecone_hit(self, integration_client):
        """Retrieved chunks flow through graph state → SourceHit schema."""
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)

        sources = resp.json()["sources"]
        assert len(sources) >= 1
        src = sources[0]
        assert src["source"] == "wiki"
        assert "Retrieval-Augmented Generation" in src["title"]
        assert "chunk_text" in src

    def test_timings_include_retrieve_ms_and_total_ms(self, integration_client):
        """Timing dict is built from real perf_counter calls in the graph nodes."""
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)

        timings = resp.json()["timings"]
        assert timings["retrieve_ms"] >= 0.0
        assert timings["generate_ms"] >= 0.0
        assert timings["total_ms"] >= 0.0

    def test_usage_carries_token_counts_from_mocked_llm(self, integration_client):
        """Token counts extracted from mock_llm.invoke().usage_metadata reach ChatTokenUsage."""
        mock_llm = _make_mock_llm(prompt_tokens=150, completion_tokens=30)
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)

        usage = resp.json()["usage"]
        assert usage is not None
        assert usage["prompt_tokens"] == 150
        assert usage["completion_tokens"] == 30
        assert usage["total_tokens"] == 180
        assert "by_call_type" in usage

    def test_insufficient_context_false_on_happy_path(self, integration_client):
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)

        assert resp.json()["insufficient_context"] is False


# ---------------------------------------------------------------------------
# P1.2 — Empty-context abstention: real pipeline, no LLM call
# ---------------------------------------------------------------------------


class TestAbstention:
    """Chunks below cosine floor → real pipeline routes to abstention without Groq."""

    def test_low_score_chunks_yield_insufficient_context_true(self, integration_client):
        """filter_chunks_by_score runs for real: 0.04 < 0.20 floor → abstention."""
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK_LOW_SCORE]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)

        assert resp.status_code == 200
        assert resp.json()["insufficient_context"] is True

    def test_abstention_answer_matches_graph_constant(self, integration_client):
        """The answer is the exact ABSTENTION_ANSWER constant — no hallucination."""
        from app.services.chat.graph import ABSTENTION_ANSWER

        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK_LOW_SCORE]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)

        assert resp.json()["answer"] == ABSTENTION_ANSWER

    def test_llm_is_not_called_on_abstention_path(self, integration_client):
        """Primary guard: mocked LLM.invoke must NOT be called on abstention."""
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK_LOW_SCORE]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)

        mock_llm.invoke.assert_not_called()

    def test_empty_retrieval_also_triggers_abstention(self, integration_client):
        """No chunks returned → empty context → abstention, no LLM call."""
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)

        assert resp.json()["insufficient_context"] is True
        mock_llm.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# P1.2 — /chat/stream: SSE event sequence end-to-end
# ---------------------------------------------------------------------------


class TestChatStream:
    """POST /chat/stream through the real SSE pipeline."""

    def test_returns_200_with_event_stream_content_type(self, integration_client):
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.streaming.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post(
                "/chat/stream", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_event_sequence_is_one_or_more_tokens_then_exactly_one_done(
        self, integration_client
    ):
        """Real SSE protocol: token* → done (exactly one), done is last."""
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.streaming.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post(
                "/chat/stream", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS
            )

        frames = _parse_sse(resp.text)
        event_types = [e for e, _ in frames]
        assert "token" in event_types, "expected at least one token event"
        assert event_types.count("done") == 1, "expected exactly one done event"
        assert event_types[-1] == "done", "done must be the last event"

    def test_token_events_carry_text_field(self, integration_client):
        """Each token event data has a 'text' key (T2.9 SSE protocol)."""
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.streaming.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post(
                "/chat/stream", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS
            )

        token_frames = [(e, d) for e, d in _parse_sse(resp.text) if e == "token"]
        assert len(token_frames) >= 1
        for _, data in token_frames:
            assert "text" in data, "token event missing 'text' field"

    def test_done_event_carries_all_observability_fields(self, integration_client):
        """done payload carries the full observability schema (T2.9 protocol)."""
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.streaming.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post(
                "/chat/stream", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS
            )

        done_data = next(d for e, d in _parse_sse(resp.text) if e == "done")
        for field in (
            "answer", "sources", "timings", "insufficient_context",
            "grounded", "faithfulness_score", "unverified_citations",
            "crag_iterations", "cached", "top_score",
        ):
            assert field in done_data, f"done event missing field: {field}"

    def test_stream_abstention_emits_token_then_done_with_insufficient_context_true(
        self, integration_client
    ):
        """Abstention path: token(abstention text) → done(insufficient_context=True)."""
        mock_llm = _make_mock_llm()
        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK_LOW_SCORE]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.streaming.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post(
                "/chat/stream", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS
            )

        frames = _parse_sse(resp.text)
        event_types = [e for e, _ in frames]
        done_data = next(d for e, d in frames if e == "done")
        assert "token" in event_types
        assert done_data["insufficient_context"] is True


# ---------------------------------------------------------------------------
# P1.2 — Faithfulness flag end-to-end: grounded + faithfulness_score populated
# ---------------------------------------------------------------------------


class TestFaithfulnessFlag:
    """With RAG_FAITHFULNESS_ENABLED=True, the real format_response node
    calls judge_faithfulness_with_usage → mock LLM → parses the verdict
    → populates grounded + faithfulness_score in ChatResponse."""

    def test_grounded_and_faithfulness_score_populated(self, integration_client):
        # First invoke: main generation answer
        fake_answer = _make_llm_response(
            answer="RAG is retrieval-augmented generation. [1]",
            prompt_tokens=100,
            completion_tokens=20,
        )
        # Second invoke: faithfulness judge returns JSON verdict
        fake_judgment = _make_llm_response(
            answer='{"grounded": true, "score": 0.85, "rationale": "All claims supported."}',
            prompt_tokens=50,
            completion_tokens=10,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [fake_answer, fake_judgment]
        mock_llm.astream = _fake_astream

        # Build settings with faithfulness enabled, using real settings as base
        from app.core.config import get_settings as _real_gs
        real_s = _real_gs()
        faith_settings = SimpleNamespace(**real_s.model_dump())
        faith_settings.RAG_FAITHFULNESS_ENABLED = True
        faith_settings.RAG_FAITHFULNESS_THRESHOLD = 0.5

        with (
            patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
            patch("app.services.chat.graph.get_llm", return_value=mock_llm),
            patch("app.services.chat.graph.get_settings", return_value=faith_settings),
            patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        ):
            resp = integration_client.post("/chat", json=_CHAT_PAYLOAD, headers=_AUTH_HEADERS)

        assert resp.status_code == 200
        body = resp.json()
        assert body["grounded"] is True
        assert body["faithfulness_score"] == pytest.approx(0.85)
        # Two LLM calls: generation + faithfulness judge
        assert mock_llm.invoke.call_count == 2
