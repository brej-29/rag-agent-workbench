from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, StateGraph

from app.core.config import get_settings
from app.core.errors import UpstreamServiceError
from app.core.logging import get_logger
from app.schemas.chat import ChatRequest
from app.services.llm.groq_llm import get_llm
from app.services.prompts.rag_prompt import build_rag_messages
from app.services.pinecone_store import search as pinecone_search
from app.services.tools.tavily_tool import get_tavily_tool, is_tavily_configured

logger = get_logger(__name__)


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


def _ensure_timings(state: ChatState) -> Dict[str, float]:
    timings = state.get("timings") or {}
    if not isinstance(timings, dict):
        timings = {}
    state["timings"] = timings
    return timings  # type: ignore[return-value]


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


def retrieve_context(state: ChatState, _config: RunnableConfig | None = None) -> ChatState:
    """Retrieve relevant document chunks from Pinecone."""
    settings = get_settings()
    timings = _ensure_timings(state)

    start = perf_counter()
    raw_hits: List[Dict[str, Any]] = pinecone_search(
        namespace=state["namespace"],
        query_text=state["query"],
        top_k=state["top_k"],
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


def generate_answer(state: ChatState, config: RunnableConfig | None = None) -> ChatState:
    """Generate an answer using the Groq-backed chat model."""
    timings = _ensure_timings(state)
    messages = build_rag_messages(
        chat_history=state.get("chat_history") or [],
        question=state["query"],
        sources=(state.get("retrieved") or []) + (state.get("web_results") or []),
    )

    llm = get_llm()
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
    logger.info("Answer generation completed elapsed_ms=%.2f", elapsed_ms)
    return state


def format_response(state: ChatState, _config: RunnableConfig | None = None) -> ChatState:
    """No-op node reserved for future formatting; currently returns state."""
    # This node exists mainly to keep the graph structure explicit and ready
    # for future formatting steps (e.g. re-ranking or response post-processing).
    return state


_graph: Optional[Any] = None


def get_chat_graph() -> Any:
    """Return the compiled LangGraph chat graph (lazy singleton)."""
    global _graph
    if _graph is not None:
        return _graph

    workflow: StateGraph = StateGraph(ChatState)

    workflow.add_node("normalize_input", normalize_input)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("decide_next", decide_next)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("format_response", format_response)

    workflow.set_entry_point("normalize_input")
    workflow.add_edge("normalize_input", "retrieve_context")
    workflow.add_edge("retrieve_context", "decide_next")
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