from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, StateGraph

from app.core.config import get_settings
from app.core.cost_accounting import estimate_cost_usd, extract_token_usage
from app.core.errors import UpstreamServiceError
from app.core.logging import get_logger
from app.schemas.chat import ChatRequest
from app.services.contextualize import contextualize_followup
from app.services.crag import grade_retrieval, rewrite_query, rewrite_query_with_usage
from app.services.faithfulness import (
    FaithfulnessVerdict,
    judge_faithfulness,
    judge_faithfulness_with_usage,
    verify_citations,
)
from app.services.llm.groq_llm import get_llm
from app.services.prompts.faithfulness_prompt import build_faithfulness_judge_messages  # noqa: F401 (imported for test-patchability)
from app.services.prompts.rag_prompt import (
    build_context_string,
    build_rag_messages,
    filter_chunks_by_score,
)
from app.services.pinecone_store import search as pinecone_search
from app.services.rerank import RERANK_CANDIDATES_MAX, rerank_chunks
from app.services.tools.tavily_tool import get_tavily_tool, is_tavily_configured

logger = get_logger(__name__)

# Returned verbatim when no usable context survives retrieval + chunk filtering.
# Defined once so tests can assert against a known constant rather than a
# fragile substring match.  The Groq LLM is NOT called on this path.
ABSTENTION_ANSWER = (
    "I was unable to find sufficient information in the knowledge base to answer "
    "your question. No retrieved chunks met the minimum relevance score threshold. "
    "Try enabling the web search fallback, broadening your query, or ingesting "
    "additional documents."
)


class ChatState(TypedDict, total=False):
    query: str
    namespace: str
    top_k: int
    min_score: float
    use_web_fallback: bool
    max_web_results: int
    chat_history: List[Dict[str, str]]

    retrieved: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]

    answer: str
    timings: Dict[str, float]

    tavily_available: bool
    web_fallback_used: bool
    top_score: float
    insufficient_context: bool

    # Grounding fields populated by format_response (T2.2)
    grounded: Optional[bool]
    faithfulness_score: Optional[float]
    unverified_citations: List[int]

    # CRAG observability fields (T2.4) — set by corrective_retrieve when CRAG is ON
    crag_iterations: int
    corrective_action: Optional[str]

    # T2.5 contextualization — set by contextualize_query
    contextualized_query: Optional[str]

    # T2.7 token accounting — accumulated across ALL LLM calls in the request
    token_usage_by_call: Dict[str, Dict[str, int]]


def _ensure_timings(state: ChatState) -> Dict[str, float]:
    timings = state.get("timings") or {}
    if not isinstance(timings, dict):
        timings = {}
    state["timings"] = timings
    return timings  # type: ignore[return-value]


def _accumulate_token_usage(
    state: ChatState,
    call_type: str,
    usage: Dict[str, int],
) -> None:
    """Sum token counts from an LLM call into state["token_usage_by_call"].

    call_type values: "generation", "judge", "crag_rewrite", "contextualize".
    These are the bounded labels also used as Prometheus counter labels (T2.7).
    Accumulates across multiple iterations (e.g. CRAG loops).
    """
    by_call: Dict[str, Dict[str, int]] = state.get("token_usage_by_call") or {}
    prev = by_call.get(call_type, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    by_call[call_type] = {
        "prompt_tokens": prev["prompt_tokens"] + int(usage.get("prompt_tokens") or 0),
        "completion_tokens": prev["completion_tokens"] + int(usage.get("completion_tokens") or 0),
        "total_tokens": prev["total_tokens"] + int(usage.get("total_tokens") or 0),
    }
    state["token_usage_by_call"] = by_call


def normalize_input(state: ChatState, _config: RunnableConfig | None = None) -> ChatState:
    """Normalise input state with default values from settings."""
    settings = get_settings()

    namespace = state.get("namespace") or settings.PINECONE_NAMESPACE
    top_k = int(state.get("top_k") or settings.RAG_DEFAULT_TOP_K)
    min_score = float(state.get("min_score") or settings.RAG_MIN_SCORE)
    max_web_results = int(state.get("max_web_results") or settings.RAG_MAX_WEB_RESULTS)

    chat_history = state.get("chat_history") or []
    # Normalise chat_history into a list of {role, content} dicts
    normalized_history: List[Dict[str, str]] = []
    for item in chat_history:
        role = item.get("role", "user")
        content = item.get("content", "")
        if content:
            normalized_history.append({"role": role, "content": content})

    new_state: ChatState = {
        **state,
        "namespace": namespace,
        "top_k": top_k,
        "min_score": min_score,
        "max_web_results": max_web_results,
        "chat_history": normalized_history,
        "retrieved": [],
        "web_results": [],
        "timings": state.get("timings") or {},
        "tavily_available": is_tavily_configured(),
        "web_fallback_used": False,
    }

    logger.info(
        "Chat graph input normalised namespace='%s' top_k=%d min_score=%.3f "
        "use_web_fallback=%s max_web_results=%d tavily_available=%s",
        new_state["namespace"],
        new_state["top_k"],
        new_state["min_score"],
        bool(new_state["use_web_fallback"]),
        new_state["max_web_results"],
        new_state["tavily_available"],
    )
    return new_state


def contextualize_query(state: ChatState, config: RunnableConfig | None = None) -> ChatState:  # noqa: ARG001
    """Rewrite a follow-up question into a standalone question (T2.5).

    When OFF (RAG_CONTEXTUALIZE_ENABLED=False, the default): exact pass-through —
    state is returned unchanged, byte-for-byte identical to the node not existing.

    When ON + no chat_history: pass-through — no LLM call (nothing to contextualize).

    When ON + history present: calls the Groq LLM via contextualize_followup().
    If the rewrite succeeds, state["query"] is replaced with the standalone form
    and state["contextualized_query"] is set to the rewritten query for observability.
    If the rewrite fails (exception, empty response), falls back to the original query;
    state["contextualized_query"] is set to None.

    DISTINCT from corrective_retrieve (CRAG / T2.4):
      - This node runs BEFORE retrieval and uses conversation history as input.
      - CRAG runs AFTER retrieval and corrects for low retrieval quality.
    """
    settings = get_settings()
    if not settings.RAG_CONTEXTUALIZE_ENABLED:
        return state  # exact pass-through — no keys added

    history = state.get("chat_history") or []
    if not history:
        return state  # first-turn request — nothing to contextualize, no LLM call

    original_query: str = state["query"]
    llm = get_llm()
    rewritten, usage = contextualize_followup(
        original_query=original_query,
        chat_history=history,
        llm=llm,
    )

    if rewritten and rewritten != original_query:
        state["query"] = rewritten
        state["contextualized_query"] = rewritten
    else:
        state["contextualized_query"] = None

    _accumulate_token_usage(state, "contextualize", usage)
    return state


def retrieve_context(state: ChatState, _config: RunnableConfig | None = None) -> ChatState:
    """Retrieve relevant document chunks from Pinecone.

    When RAG_RERANK_ENABLED is True, fetches RAG_RERANK_CANDIDATES candidates
    (clamped to >= top_k) to give the reranker a wider pool.  When False the
    behavior is identical to the pre-reranking baseline — no extra call, no
    reordering.

    top_score is always the MAX COSINE score of the retrieved hits so that
    decide_next's web-fallback routing (which compares against the cosine-
    calibrated RAG_MIN_SCORE) is unaffected by the reranking flag.
    """
    settings = get_settings()
    timings = _ensure_timings(state)

    # Stage-1 retrieval: when reranking is enabled, fetch a wider candidate pool.
    # When disabled, fetch exactly top_k — byte-for-byte identical to baseline.
    if settings.RAG_RERANK_ENABLED:
        # Lower clamp: candidates must cover at least top_k.
        # Upper clamp: bge-reranker-v2-m3 caps at RERANK_CANDIDATES_MAX docs
        # per call; exceeding it triggers an API error (even if caught by
        # graceful degradation, the call latency is already wasted).
        n_candidates = min(max(settings.RAG_RERANK_CANDIDATES, state["top_k"]), RERANK_CANDIDATES_MAX)
    else:
        n_candidates = state["top_k"]

    start = perf_counter()
    raw_hits: List[Dict[str, Any]] = pinecone_search(
        namespace=state["namespace"],
        query_text=state["query"],
        top_k=n_candidates,
        filters=None,
        fields=None,
    )
    elapsed_ms = (perf_counter() - start) * 1000.0
    timings["retrieve_ms"] = elapsed_ms
    state["timings"] = timings

    text_field = settings.PINECONE_TEXT_FIELD
    retrieved: List[Dict[str, Any]] = []
    top_score = 0.0

    for hit in raw_hits:
        hit_score = float(hit.get("_score") or hit.get("score") or 0.0)
        fields: Dict[str, Any] = hit.get("fields") or {}
        raw_text = fields.get(text_field, "") or ""

        # Map the configured text field into a stable chunk_text key
        chunk_text = str(raw_text)
        title = str(fields.get("title") or "")
        source = str(fields.get("source") or "unknown")
        url = str(fields.get("url") or "")

        retrieved.append(
            {
                "source": source,
                "title": title,
                "url": url,
                "score": hit_score,
                "chunk_text": chunk_text,
            }
        )
        top_score = max(top_score, hit_score)

    state["retrieved"] = retrieved
    state["top_score"] = top_score

    logger.info(
        "Pinecone retrieval completed namespace='%s' top_k=%d hits=%d top_score=%.4f",
        state["namespace"],
        state["top_k"],
        len(retrieved),
        top_score,
    )
    return state


def corrective_retrieve(state: ChatState, config: RunnableConfig | None = None) -> ChatState:  # noqa: ARG001
    """CRAG corrective retrieval loop (gated by RAG_CRAG_ENABLED).

    When OFF (the default): exact pass-through — state is returned unchanged,
    byte-for-byte identical to the node never being wired.

    When ON: grades the initial retrieval using the top cosine score already in state
    (no re-embedding — avoids circular validation).  If graded 'weak', rewrites the
    query via the Groq LLM and re-retrieves from Pinecone, up to RAG_CRAG_MAX_ITERS
    times.  The loop ALWAYS terminates after max_iters regardless of the grade outcome.

    Composes with existing nodes unchanged: the cosine floor (generate_answer), the
    Tavily web fallback (decide_next -> web_search), the empty-context abstention
    (generate_answer), and the faithfulness check (format_response) all run after this
    node and are unmodified.
    """
    settings = get_settings()
    if not settings.RAG_CRAG_ENABLED:
        return state

    _ensure_timings(state)
    state["crag_iterations"] = 0
    state["corrective_action"] = None

    current_query: str = state["query"]
    max_iters: int = settings.RAG_CRAG_MAX_ITERS

    for iteration in range(max_iters):
        top_score = float(state.get("top_score") or 0.0)
        grade = grade_retrieval(top_score, settings.RAG_CRAG_GOOD_SCORE)

        if grade == "good":
            logger.info(
                "CRAG grade=good top_score=%.4f >= threshold=%.4f (iteration %d) -- no correction",
                top_score,
                settings.RAG_CRAG_GOOD_SCORE,
                iteration,
            )
            break

        state["crag_iterations"] = iteration + 1
        logger.info(
            "CRAG iteration=%d grade=weak top_score=%.4f threshold=%.4f -- rewriting query",
            iteration + 1,
            top_score,
            settings.RAG_CRAG_GOOD_SCORE,
        )

        # Rewrite query using existing Groq client — no new LLM, no new dependency
        llm = get_llm()
        rewritten, rewrite_usage = rewrite_query_with_usage(current_query, llm)
        _accumulate_token_usage(state, "crag_rewrite", rewrite_usage)
        if rewritten != current_query:
            current_query = rewritten
            state["corrective_action"] = "rewrite"

        # Re-retrieve from Pinecone with the (possibly rewritten) query
        n_candidates = int(state.get("top_k") or settings.RAG_DEFAULT_TOP_K)
        raw_hits: List[Dict[str, Any]] = pinecone_search(
            namespace=state["namespace"],
            query_text=current_query,
            top_k=n_candidates,
            filters=None,
            fields=None,
        )

        text_field = settings.PINECONE_TEXT_FIELD
        retrieved: List[Dict[str, Any]] = []
        new_top_score = 0.0
        for hit in raw_hits:
            hit_score = float(hit.get("_score") or hit.get("score") or 0.0)
            fields: Dict[str, Any] = hit.get("fields") or {}
            chunk_text = str(fields.get(text_field, "") or "")
            retrieved.append({
                "source": str(fields.get("source") or "unknown"),
                "title": str(fields.get("title") or ""),
                "url": str(fields.get("url") or ""),
                "score": hit_score,
                "chunk_text": chunk_text,
            })
            new_top_score = max(new_top_score, hit_score)

        state["retrieved"] = retrieved
        state["top_score"] = new_top_score
        logger.info(
            "CRAG re-retrieval iteration=%d hits=%d top_score=%.4f",
            iteration + 1,
            len(retrieved),
            new_top_score,
        )

    return state


def decide_next(state: ChatState, _config: RunnableConfig | None = None) -> ChatState:
    """Decide whether to proceed with web search or answer generation."""
    use_web = bool(state.get("use_web_fallback"))
    tavily_available = bool(state.get("tavily_available"))
    retrieved = state.get("retrieved") or []
    min_score = float(state.get("min_score") or 0.0)
    top_score = float(state.get("top_score") or 0.0)

    should_use_web = False
    if use_web and tavily_available:
        if not retrieved:
            should_use_web = True
        elif top_score < min_score:
            should_use_web = True

    state["web_fallback_used"] = should_use_web

    logger.info(
        "Chat routing decision use_web=%s tavily_available=%s "
        "retrieved=%d top_score=%.4f min_score=%.4f",
        should_use_web,
        tavily_available,
        len(retrieved),
        top_score,
        min_score,
    )
    return state


def _route_after_decide_next(state: ChatState) -> str:
    """Conditional routing function for LangGraph."""
    if state.get("web_fallback_used"):
        return "web_search"
    return "generate_answer"


def web_search(state: ChatState, config: RunnableConfig | None = None) -> ChatState:
    """Perform Tavily web search and convert results into pseudo-doc chunks."""
    timings = _ensure_timings(state)
    max_results = int(state.get("max_web_results") or 5)

    tool = get_tavily_tool(max_results=max_results)
    if tool is None:
        logger.warning("Tavily tool unavailable; skipping web search.")
        timings.setdefault("web_ms", 0.0)
        state["timings"] = timings
        state["web_results"] = []
        return state

    start = perf_counter()
    try:
        # The TavilySearchResults tool is a Runnable, so we can pass config for tracing.
        results: Any = tool.invoke({"query": state["query"]}, config=config or {})
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (perf_counter() - start) * 1000.0
        timings["web_ms"] = elapsed_ms
        state["timings"] = timings
        logger.error("Tavily web search failed: %s", exc)
        raise UpstreamServiceError(
            service="Tavily",
            message="Upstream Tavily web search failed. Try again later or disable web fallback.",
        ) from exc

    elapsed_ms = (perf_counter() - start) * 1000.0
    timings["web_ms"] = elapsed_ms
    state["timings"] = timings

    web_hits: List[Dict[str, Any]] = []
    # TavilySearchResults returns a list of dicts by default.
    if isinstance(results, list):
        iterable = results
    else:
        iterable = getattr(results, "data", []) or []

    for item in iterable:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")
        title = str(item.get("title") or "") or url
        content = str(item.get("content") or item.get("snippet") or "")

        web_hits.append(
            {
                "source": "web",
                "title": title,
                "url": url,
                "score": 0.0,
                "chunk_text": content,
            }
        )

    logger.info(
        "Tavily web search completed results=%d elapsed_ms=%.2f",
        len(web_hits),
        elapsed_ms,
    )
    state["web_results"] = web_hits
    return state


def _prepare_generation_inputs(
    state: ChatState,
) -> tuple[bool, list, list]:
    """Apply cosine floor, optional rerank, and check for empty context.

    Extracted from generate_answer so the streaming path can call it without
    invoking the LLM — the streaming generator calls this synchronously, then
    drives the LLM via llm.astream() itself.

    Mutates state["retrieved"] and state["timings"] in place.

    Returns:
        (should_abstain, usable_sources, messages)
        If should_abstain=True, state["answer"] and state["insufficient_context"]
        are already set; usable_sources and messages are empty lists.
    """
    settings = get_settings()
    timings = _ensure_timings(state)

    # Stage 1: cosine floor (Pinecone chunks only).
    # Routing in decide_next already read top_score from the full retrieved
    # list, so filtering here is safe.  Web results bypass filtering (no score).
    filtered_pinecone = filter_chunks_by_score(
        state.get("retrieved") or [], settings.RAG_MIN_CHUNK_SCORE
    )
    state["retrieved"] = filtered_pinecone

    # Stage 2: hosted rerank (gated by RAG_RERANK_ENABLED).
    top_k = state.get("top_k") or settings.RAG_DEFAULT_TOP_K
    if settings.RAG_RERANK_ENABLED and filtered_pinecone:
        rerank_start = perf_counter()
        filtered_pinecone = rerank_chunks(
            query=state["query"],
            chunks=filtered_pinecone,
            top_n=top_k,
            model=settings.RAG_RERANK_MODEL,
        )
        timings["rerank_ms"] = (perf_counter() - rerank_start) * 1000.0
        state["retrieved"] = filtered_pinecone
    else:
        timings["rerank_ms"] = 0.0
    state["timings"] = timings

    web_results = state.get("web_results") or []
    usable_sources = filtered_pinecone + web_results

    if not usable_sources:
        state["answer"] = ABSTENTION_ANSWER
        state["insufficient_context"] = True
        timings.setdefault("generate_ms", 0.0)
        state["timings"] = timings
        logger.info(
            "Empty context after chunk score filtering — returning deterministic "
            "abstention (Groq NOT called). "
            "filtered_pinecone=%d web_results=%d min_chunk_score=%.3f",
            len(filtered_pinecone),
            len(web_results),
            settings.RAG_MIN_CHUNK_SCORE,
        )
        return True, [], []

    messages = build_rag_messages(
        chat_history=state.get("chat_history") or [],
        question=state["query"],
        sources=usable_sources,
    )
    return False, usable_sources, messages


def generate_answer(state: ChatState, config: RunnableConfig | None = None) -> ChatState:
    """Generate an answer using the Groq-backed chat model.

    Two behaviours:
    1. Per-chunk score floor + optional rerank (delegated to _prepare_generation_inputs).
    2. Empty-context guard — if no usable context survives, a deterministic
       abstention is returned WITHOUT calling the LLM.
    """
    should_abstain, _usable_sources, messages = _prepare_generation_inputs(state)
    if should_abstain:
        return state

    llm = get_llm()
    timings = _ensure_timings(state)
    start = perf_counter()
    try:
        response = llm.invoke(messages, config=config or {})
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (perf_counter() - start) * 1000.0
        timings["generate_ms"] = elapsed_ms
        state["timings"] = timings
        logger.error("Groq chat completion failed: %s", exc)
        raise UpstreamServiceError(
            service="Groq",
            message="Upstream Groq chat completion failed. Please try again later.",
        ) from exc

    elapsed_ms = (perf_counter() - start) * 1000.0
    timings["generate_ms"] = elapsed_ms
    state["timings"] = timings

    answer_text: str
    try:
        answer_text = str(getattr(response, "content", "") or response)
    except Exception:  # noqa: BLE001
        answer_text = str(response)

    state["answer"] = answer_text
    state["insufficient_context"] = False
    _accumulate_token_usage(state, "generation", extract_token_usage(response))
    logger.info("Answer generation completed elapsed_ms=%.2f", elapsed_ms)
    return state


def format_response(state: ChatState, _config: RunnableConfig | None = None) -> ChatState:
    """Grounding checks run after generation.

    Layer 1 (always, free): deterministic citation-marker verification.
    Layer 2 (gated by RAG_FAITHFULNESS_ENABLED): LLM-judge faithfulness check.

    The abstention path (insufficient_context=True) bypasses the judge entirely —
    there is no model-generated answer to evaluate and the LLM must not be called
    a second time.  Does NOT alter the answer text.
    """
    settings = get_settings()
    timings = _ensure_timings(state)

    answer_text: str = state.get("answer") or ""
    retrieved: List[Dict[str, Any]] = state.get("retrieved") or []
    web_results: List[Dict[str, Any]] = state.get("web_results") or []
    is_abstention: bool = bool(state.get("insufficient_context"))

    # Layer 1: deterministic citation-marker check (always runs, zero model calls).
    all_sources = retrieved + web_results
    unverified = verify_citations(answer_text, all_sources)
    state["unverified_citations"] = unverified

    # Defaults — overwritten below when the judge runs successfully.
    state["grounded"] = None
    state["faithfulness_score"] = None
    timings["faithfulness_ms"] = 0.0

    # Layer 2: LLM-judge (gated).  Skip on abstention path — no answer to judge.
    if settings.RAG_FAITHFULNESS_ENABLED and not is_abstention and answer_text:
        context_string = build_context_string(all_sources)
        llm = get_llm()
        t0 = perf_counter()
        verdict, judge_usage = judge_faithfulness_with_usage(answer_text, context_string, llm)
        timings["faithfulness_ms"] = (perf_counter() - t0) * 1000.0
        _accumulate_token_usage(state, "judge", judge_usage)

        # Resolve grounded using the score threshold (more objective than raw bool).
        if verdict.faithfulness_score is not None:
            state["grounded"] = (
                verdict.faithfulness_score >= settings.RAG_FAITHFULNESS_THRESHOLD
            )
        elif verdict.grounded is not None:
            state["grounded"] = verdict.grounded
        # else: parse failure → grounded stays None ("unknown")

        state["faithfulness_score"] = verdict.faithfulness_score
        logger.info(
            "Faithfulness check grounded=%s score=%s unverified_citations=%s ms=%.2f",
            state["grounded"],
            state["faithfulness_score"],
            unverified,
            timings["faithfulness_ms"],
        )

    state["timings"] = timings
    return state


_graph: Optional[Any] = None


def get_chat_graph() -> Any:
    """Return the compiled LangGraph chat graph (lazy singleton)."""
    global _graph
    if _graph is not None:
        return _graph

    workflow: StateGraph = StateGraph(ChatState)

    workflow.add_node("normalize_input", normalize_input)
    workflow.add_node("contextualize_query", contextualize_query)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("corrective_retrieve", corrective_retrieve)
    workflow.add_node("decide_next", decide_next)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("format_response", format_response)

    workflow.set_entry_point("normalize_input")
    workflow.add_edge("normalize_input", "contextualize_query")
    workflow.add_edge("contextualize_query", "retrieve_context")
    workflow.add_edge("retrieve_context", "corrective_retrieve")
    workflow.add_edge("corrective_retrieve", "decide_next")
    workflow.add_conditional_edges(
        "decide_next",
        _route_after_decide_next,
        {
            "web_search": "web_search",
            "generate_answer": "generate_answer",
        },
    )
    workflow.add_edge("web_search", "generate_answer")
    workflow.add_edge("generate_answer", "format_response")
    workflow.add_edge("format_response", END)

    _graph = workflow.compile()
    logger.info("Chat LangGraph compiled and initialised.")
    return _graph