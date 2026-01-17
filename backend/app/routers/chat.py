import json
from time import perf_counter
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.core.cache import cache_enabled, get_chat_cached, set_chat_cached
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import record_chat_timings
from app.core.rate_limit import limiter
from app.core.tracing import (
    get_tracing_callbacks,
    get_tracing_response_metadata,
)
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatTimings,
    ChatTraceMetadata,
    SourceHit,
)
from app.services.chat.graph import get_chat_graph

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])


def _build_chat_response(state: Dict) -> ChatResponse:
    """Convert graph state into a ChatResponse model."""
    timings_raw = state.get("timings") or {}
    timings = ChatTimings(
        retrieve_ms=float(timings_raw.get("retrieve_ms") or 0.0),
        web_ms=float(timings_raw.get("web_ms") or 0.0),
        generate_ms=float(timings_raw.get("generate_ms") or 0.0),
        total_ms=float(timings_raw.get("total_ms") or 0.0),
    )

    sources_raw: List[Dict] = (state.get("retrieved") or []) + (
        state.get("web_results") or []
    )
    sources: List[SourceHit] = [
        SourceHit(
            source=str(src.get("source") or "unknown"),
            title=str(src.get("title") or ""),
            url=str(src.get("url") or ""),
            score=float(src.get("score") or 0.0),
            chunk_text=str(src.get("chunk_text") or ""),
        )
        for src in sources_raw
    ]

    trace_meta = ChatTraceMetadata(**get_tracing_response_metadata())

    return ChatResponse(
        answer=str(state.get("answer") or ""),
        sources=sources,
        timings=timings,
        trace=trace_meta,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Production-style RAG chat endpoint",
    description=(
        "Runs an agentic RAG flow using Pinecone retrieval, optional Tavily web "
        "fallback, and a Groq-backed LLM to generate an answer. "
        "Returns the answer, source snippets, timings, and LangSmith trace metadata."
    ),
)
@limiter.limit("30/minute")
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:  # noqa: ARG001
    settings = get_settings()
    namespace = payload.namespace or settings.PINECONE_NAMESPACE

    logger.info(
        "Received /chat request namespace='%s' top_k=%d use_web_fallback=%s",
        namespace,
        payload.top_k,
        payload.use_web_fallback,
    )

    use_cache = cache_enabled() and not payload.chat_history
    cached_response: Optional[ChatResponse] = None
    if use_cache:
        cached = get_chat_cached(
            namespace=namespace,
            query=payload.query,
            top_k=payload.top_k,
            min_score=payload.min_score,
            use_web_fallback=payload.use_web_fallback,
        )
        if cached is not None:
            logger.info(
                "Serving /chat response from cache namespace='%s' query='%s'",
                namespace,
                payload.query,
            )
            cached_response = cached

    if cached_response is not None:
        # Still record timings and metrics based on the cached response.
        record_chat_timings(
            {
                "retrieve_ms": cached_response.timings.retrieve_ms,
                "web_ms": cached_response.timings.web_ms,
                "generate_ms": cached_response.timings.generate_ms,
                "total_ms": cached_response.timings.total_ms,
            }
        )
        return cached_response

    graph = get_chat_graph()
    callbacks = get_tracing_callbacks()
    config: Dict = {}
    if callbacks:
        config["callbacks"] = callbacks

    initial_state = {
        "query": payload.query,
        "namespace": namespace,
        "top_k": payload.top_k,
        "use_web_fallback": payload.use_web_fallback,
        "min_score": payload.min_score,
        "max_web_results": payload.max_web_results,
        "chat_history": [
            {"role": msg.role, "content": msg.content}
            for msg in (payload.chat_history or [])
        ],
    }

    start_total = perf_counter()

    def _invoke_graph() -> Dict:
        return graph.invoke(initial_state, config=config)

    # Exceptions (including UpstreamServiceError) are handled by global handlers.
    state = await run_in_threadpool(_invoke_graph)

    total_ms = (perf_counter() - start_total) * 1000.0
    timings = state.get("timings") or {}
    timings["total_ms"] = total_ms
    state["timings"] = timings

    web_used = bool(state.get("web_fallback_used"))
    top_score = float(state.get("top_score") or 0.0)
    logger.info(
        "Chat request completed namespace='%s' web_fallback_used=%s "
        "retrieve_ms=%.2f web_ms=%.2f generate_ms=%.2f total_ms=%.2f top_score=%.4f",
        namespace,
        web_used,
        float(timings.get("retrieve_ms") or 0.0),
        float(timings.get("web_ms") or 0.0),
        float(timings.get("generate_ms") or 0.0),
        float(timings.get("total_ms") or 0.0),
        top_score,
    )

    response_model = _build_chat_response(state)

    # Record metrics based on this response.
    record_chat_timings(
        {
            "retrieve_ms": response_model.timings.retrieve_ms,
            "web_ms": response_model.timings.web_ms,
            "generate_ms": response_model.timings.generate_ms,
            "total_ms": response_model.timings.total_ms,
        }
    )

    # Cache only when chat_history is empty.
    if use_cache:
        set_chat_cached(
            namespace=namespace,
            query=payload.query,
            top_k=payload.top_k,
            min_score=payload.min_score,
            use_web_fallback=payload.use_web_fallback,
            value=response_model,
        )

    return response_model


@router.post(
    "/chat/stream",
    summary="Streaming RAG chat endpoint (SSE)",
    description=(
        "Same behaviour as /chat but streams the answer over Server-Sent Events "
        "(SSE). The final event includes the full JSON payload with answer, sources, "
        "timings, and trace metadata."
    ),
)
@limiter.limit("30/minute")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:  # noqa: ARG001
    settings = get_settings()
    namespace = payload.namespace or settings.PINECONE_NAMESPACE

    logger.info(
        "Received /chat/stream request namespace='%s' top_k=%d use_web_fallback=%s",
        namespace,
        payload.top_k,
        payload.use_web_fallback,
    )

    graph = get_chat_graph()
    callbacks = get_tracing_callbacks()
    config: Dict = {}
    if callbacks:
        config["callbacks"] = callbacks

    initial_state = {
        "query": payload.query,
        "namespace": namespace,
        "top_k": payload.top_k,
        "use_web_fallback": payload.use_web_fallback,
        "min_score": payload.min_score,
        "max_web_results": payload.max_web_results,
        "chat_history": [
            {"role": msg.role, "content": msg.content}
            for msg in (payload.chat_history or [])
        ],
    }

    start_total = perf_counter()

    def _invoke_graph() -> Dict:
        return graph.invoke(initial_state, config=config)

    # Exceptions (including UpstreamServiceError) are handled by global handlers.
    state = await run_in_threadpool(_invoke_graph)

    total_ms = (perf_counter() - start_total) * 1000.0
    timings = state.get("timings") or {}
    timings["total_ms"] = total_ms
    state["timings"] = timings

    web_used = bool(state.get("web_fallback_used"))
    top_score = float(state.get("top_score") or 0.0)
    logger.info(
        "Streaming chat completed namespace='%s' web_fallback_used=%s "
        "retrieve_ms=%.2f web_ms=%.2f generate_ms=%.2f total_ms=%.2f top_score=%.4f",
        namespace,
        web_used,
        float(timings.get("retrieve_ms") or 0.0),
        float(timings.get("web_ms") or 0.0),
        float(timings.get("generate_ms") or 0.0),
        float(timings.get("total_ms") or 0.0),
        top_score,
    )

    response_model = _build_chat_response(state)
    answer_text = response_model.answer

    # Record metrics based on this response as well.
    record_chat_timings(
        {
            "retrieve_ms": response_model.timings.retrieve_ms,
            "web_ms": response_model.timings.web_ms,
            "generate_ms": response_model.timings.generate_ms,
            "total_ms": response_model.timings.total_ms,
        }
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        # Stream the answer token-by-token (space-delimited) as simple SSE events.
        for token in answer_text.split():
            yield f"data: {token}\n\n"

        # Send a final event containing the full JSON payload for clients that
        # want metadata and sources.
        final_payload = response_model.model_dump()
        yield f"event: end\ndata: {json.dumps(final_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")