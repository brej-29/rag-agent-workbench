"""
True generation-token streaming for /chat/stream (T2.9).

Architecture
------------
The SSE stream is broken into three phases:

  Phase 1 — Pre-generation (synchronous, in threadpool):
    normalize_input → contextualize_query → retrieve_context →
    corrective_retrieve → decide_next → [web_search if routed]
    All existing pipeline logic (CRAG, contextualize, cosine floor, web
    fallback routing) runs unchanged via direct node function calls.

  Phase 2 — Token streaming (async):
    For usable context: llm.astream(messages) → "token" SSE events in real
    time as the LLM produces them.  Real TTFT improvement over the old
    word-split approach.
    Non-streamable paths (abstention, cache hit) emit a single "token" event
    for the answer text then the "done" event — the LLM is NOT called.

  Phase 3 — Post-generation (synchronous, in threadpool):
    format_response (deterministic citation check + optional faithfulness
    judge), token accounting, Prometheus recording.  The final "done" SSE
    event is emitted AFTER these steps complete, carrying all observability
    fields (grounding, cost, timings, CRAG/contextualize state).

SSE event protocol (T2.9)
-------------------------
  event: token   data: {"text": "<text>"}                    # zero or more
  event: done    data: {<ChatResponse-like fields> + "cached": bool}  # exactly one
  event: error   data: {"message": "<human-readable error>"} # exactly one, no done follows

See docs/CONTEXT.md § "Streaming (T2.9)" for the full protocol and payload shapes.

Design rationale: full-graph astream_events is out of scope — only the final
generation LLM call benefits from streaming; earlier nodes (retrieval, CRAG,
contextualize) are I/O-bound and complete synchronously before the first token
is useful.  Running them in a single threadpool call keeps the code simple and
the existing test coverage unchanged.
"""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.cost_accounting import estimate_cost_usd, extract_token_usage
from app.core.logging import get_logger
from app.core.metrics import record_chat_timings
from app.core.prometheus_metrics import record_chat_timings_prometheus, record_token_usage
from app.core.tracing import get_tracing_response_metadata
from app.services.chat.graph import (
    ABSTENTION_ANSWER,
    _accumulate_token_usage,
    _prepare_generation_inputs,
    contextualize_query,
    corrective_retrieve,
    decide_next,
    format_response,
    normalize_input,
    retrieve_context,
    web_search,
)
from app.services.llm.groq_llm import get_llm

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: Any) -> str:
    """Format a single SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _build_done_payload(state: Dict, total_ms: float, *, cached: bool = False) -> Dict:
    """Assemble the 'done' SSE event payload from pipeline state.

    Carries the same observability fields as ChatResponse (grounding, cost,
    timings, CRAG/contextualize state) plus 'cached' and 'top_score'.
    """
    timings_raw = dict(state.get("timings") or {})

    sources_raw = (state.get("retrieved") or []) + (state.get("web_results") or [])
    sources = [
        {
            "source": str(s.get("source") or "unknown"),
            "title": str(s.get("title") or ""),
            "url": str(s.get("url") or ""),
            "score": float(s.get("score") or 0.0),
            "chunk_text": str(s.get("chunk_text") or ""),
        }
        for s in sources_raw
    ]

    # Token usage: same aggregation logic as routers/chat.py _build_chat_response.
    by_call: Dict = dict(state.get("token_usage_by_call") or {})
    by_call = {k: v for k, v in by_call.items() if int((v or {}).get("total_tokens") or 0) > 0}
    total_prompt = sum(int((v or {}).get("prompt_tokens") or 0) for v in by_call.values())
    total_completion = sum(int((v or {}).get("completion_tokens") or 0) for v in by_call.values())
    total_tokens = total_prompt + total_completion

    settings = get_settings()
    usage: Optional[Dict] = None
    if total_tokens > 0 or by_call:
        cost = estimate_cost_usd(total_prompt, total_completion, settings.GROQ_MODEL)
        usage = {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "estimated_cost_usd": cost,
            "by_call_type": by_call,
        }

    return {
        "answer": str(state.get("answer") or ""),
        "sources": sources,
        "timings": {
            "retrieve_ms": float(timings_raw.get("retrieve_ms") or 0.0),
            "rerank_ms": float(timings_raw.get("rerank_ms") or 0.0),
            "web_ms": float(timings_raw.get("web_ms") or 0.0),
            "generate_ms": float(timings_raw.get("generate_ms") or 0.0),
            "faithfulness_ms": float(timings_raw.get("faithfulness_ms") or 0.0),
            "total_ms": total_ms,
        },
        "trace": get_tracing_response_metadata(),
        "insufficient_context": bool(state.get("insufficient_context") or False),
        "grounded": state.get("grounded"),
        "faithfulness_score": state.get("faithfulness_score"),
        "unverified_citations": list(state.get("unverified_citations") or []),
        "crag_iterations": int(state.get("crag_iterations") or 0),
        "corrective_action": state.get("corrective_action"),
        "contextualized_query": state.get("contextualized_query"),
        "usage": usage,
        "cached": cached,
        "top_score": float(state.get("top_score") or 0.0),
        "web_fallback_used": bool(state.get("web_fallback_used") or False),
    }


# ---------------------------------------------------------------------------
# Synchronous pipeline helpers — called via run_in_threadpool
# ---------------------------------------------------------------------------


def _run_pre_generation_pipeline(state: Dict, config: Dict) -> Dict:
    """Run pipeline nodes up to (but not including) generation.

    Calls node functions directly — identical logic to the compiled LangGraph.
    Synchronous; intended to be called via run_in_threadpool.

    Node order:
      normalize_input → contextualize_query → retrieve_context →
      corrective_retrieve → decide_next → [web_search if routed]
    """
    state = normalize_input(state)
    state = contextualize_query(state)
    state = retrieve_context(state)
    state = corrective_retrieve(state)
    state = decide_next(state)
    if state.get("web_fallback_used"):
        state = web_search(state, config or {})
    return state


def _run_post_generation(state: Dict) -> Dict:
    """Run format_response: citation check + optional faithfulness judge.

    Called after the token stream completes, before the final metadata event.
    Synchronous; intended to be called via run_in_threadpool.
    """
    return format_response(state)


def _record_metrics(state: Dict, total_ms: float) -> None:
    """Emit legacy + Prometheus timing metrics and per-call token counters."""
    timings = dict(state.get("timings") or {})
    timings["total_ms"] = total_ms
    record_chat_timings(
        {
            "retrieve_ms": float(timings.get("retrieve_ms") or 0.0),
            "web_ms": float(timings.get("web_ms") or 0.0),
            "generate_ms": float(timings.get("generate_ms") or 0.0),
            "total_ms": total_ms,
        }
    )
    record_chat_timings_prometheus(timings)
    record_token_usage(state.get("token_usage_by_call") or {})


# ---------------------------------------------------------------------------
# Main streaming generator
# ---------------------------------------------------------------------------


async def stream_chat_response(
    initial_state: Dict,
    config: Dict,
    *,
    use_cache: bool = False,
    cached_response: Any = None,
) -> AsyncGenerator[str, None]:
    """Async generator: run the pipeline and yield SSE frames.

    SSE protocol (T2.9):
      event: token  — one per LLM-generated text chunk; data: {"text": "..."}
      event: done   — exactly one after stream completes; data: full observability payload
      event: error  — on pipeline failure; no done event follows

    Non-streamable paths (cache hit, abstention) emit one "token" event for
    the answer text then the "done" event — the streaming LLM is NOT called.

    Faithfulness + token accounting run AFTER the token stream, before the
    "done" event, so grounding and cost fields are accurate in the metadata.
    """
    start_total = perf_counter()

    # -------------------------------------------------------------------------
    # Path A: Cache hit — emit answer as single token event; no LLM call
    # -------------------------------------------------------------------------
    if use_cache and cached_response is not None:
        logger.info("Streaming: serving cached response.")
        yield _sse("token", {"text": cached_response.answer})
        done_payload = {
            "answer": cached_response.answer,
            "sources": [s.model_dump() for s in cached_response.sources],
            "timings": cached_response.timings.model_dump(),
            "trace": cached_response.trace.model_dump(),
            "insufficient_context": cached_response.insufficient_context,
            "grounded": cached_response.grounded,
            "faithfulness_score": cached_response.faithfulness_score,
            "unverified_citations": cached_response.unverified_citations,
            "crag_iterations": cached_response.crag_iterations,
            "corrective_action": cached_response.corrective_action,
            "contextualized_query": cached_response.contextualized_query,
            "usage": cached_response.usage.model_dump() if cached_response.usage else None,
            "cached": True,
            "top_score": 0.0,
            "web_fallback_used": False,
        }
        yield _sse("done", done_payload)
        return

    # -------------------------------------------------------------------------
    # Phase 1: Pre-generation pipeline (synchronous, in threadpool)
    # normalize_input → contextualize_query → retrieve_context →
    # corrective_retrieve → decide_next → [web_search]
    # -------------------------------------------------------------------------
    try:
        state: Dict = await run_in_threadpool(
            _run_pre_generation_pipeline, initial_state, config
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Streaming: pre-generation pipeline failed: %s", exc)
        yield _sse("error", {"message": str(exc)})
        return

    # -------------------------------------------------------------------------
    # Determine generation inputs: apply cosine floor, rerank, abstention check
    # -------------------------------------------------------------------------
    try:
        should_abstain, _usable_sources, messages = await run_in_threadpool(
            _prepare_generation_inputs, state
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Streaming: generation preparation failed: %s", exc)
        yield _sse("error", {"message": "Generation preparation failed. Please try again."})
        return

    # -------------------------------------------------------------------------
    # Path B: Empty-context abstention (T1.3) — no LLM call
    # -------------------------------------------------------------------------
    if should_abstain:
        logger.info("Streaming: abstention path (no usable context).")
        yield _sse("token", {"text": str(state.get("answer") or ABSTENTION_ANSWER)})
        try:
            state = await run_in_threadpool(_run_post_generation, state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Streaming: post-generation failed on abstention path: %s", exc)
        total_ms = (perf_counter() - start_total) * 1000.0
        timings = state.get("timings") or {}
        timings["total_ms"] = total_ms
        state["timings"] = timings
        _record_metrics(state, total_ms)
        yield _sse("done", _build_done_payload(state, total_ms, cached=False))
        return

    # -------------------------------------------------------------------------
    # Phase 2: Real token streaming — LLM produces tokens as they are generated
    # -------------------------------------------------------------------------
    llm = get_llm()
    answer_chunks: list[str] = []
    last_chunk: Any = None
    gen_start = perf_counter()

    try:
        async for chunk in llm.astream(messages, config=config or {}):
            last_chunk = chunk
            text = str(getattr(chunk, "content", None) or "")
            if text:
                answer_chunks.append(text)
                yield _sse("token", {"text": text})
    except Exception as exc:  # noqa: BLE001
        logger.error("Streaming: LLM generation failed: %s", exc)
        yield _sse("error", {"message": "LLM generation failed. Please try again later."})
        return

    generate_ms = (perf_counter() - gen_start) * 1000.0
    state["answer"] = "".join(answer_chunks)
    state["insufficient_context"] = False
    timings = state.get("timings") or {}
    timings["generate_ms"] = generate_ms
    state["timings"] = timings

    # Capture generation token usage from the final streaming chunk.
    # The Groq streaming API may include usage in the last chunk.
    # Falls back to zeros (via extract_token_usage guard) if not provided.
    if last_chunk is not None:
        _accumulate_token_usage(state, "generation", extract_token_usage(last_chunk))

    # -------------------------------------------------------------------------
    # Phase 3: Post-generation (faithfulness, citations, token accounting)
    # Runs AFTER the token stream, before the final metadata event.
    # -------------------------------------------------------------------------
    try:
        state = await run_in_threadpool(_run_post_generation, state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Streaming: post-generation step failed: %s", exc)

    total_ms = (perf_counter() - start_total) * 1000.0
    timings = state.get("timings") or {}
    timings["total_ms"] = total_ms
    state["timings"] = timings
    _record_metrics(state, total_ms)
    yield _sse("done", _build_done_payload(state, total_ms, cached=False))
