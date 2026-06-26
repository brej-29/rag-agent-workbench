from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(
        ...,
        description="Role of the message author (user or assistant).",
    )
    content: str = Field(..., description="Message text content.")


class ChatTokenUsage(BaseModel):
    """Per-request token accounting across ALL LLM calls (T2.7).

    Token counts are ACTUAL values from the Groq API response — not tokenizer
    estimates.  Cost is an ESTIMATE from an as-of-date pricing table
    (backend/app/core/cost_accounting.py) that must be updated when provider
    pricing changes.
    """

    prompt_tokens: int = Field(
        default=0,
        description="Total prompt (input) tokens across all LLM calls in this request.",
    )
    completion_tokens: int = Field(
        default=0,
        description="Total completion (output) tokens across all LLM calls in this request.",
    )
    total_tokens: int = Field(
        default=0,
        description="Total tokens (prompt + completion) across all LLM calls in this request.",
    )
    estimated_cost_usd: Optional[float] = Field(
        default=None,
        description=(
            "ESTIMATE of total LLM cost in USD. Derived from an as-of-date pricing "
            "table in backend/app/core/cost_accounting.py — MUST be manually updated "
            "when provider pricing changes.  None when the model is not in the pricing "
            "table.  Does not account for free-tier credits, discounts, or batch pricing."
        ),
    )
    by_call_type: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-call-type token breakdown. Keys: 'generation' (main answer LLM), "
            "'judge' (faithfulness check, when RAG_FAITHFULNESS_ENABLED=True), "
            "'crag_rewrite' (CRAG corrective rewrite, when RAG_CRAG_ENABLED=True), "
            "'contextualize' (history-aware rewrite, when RAG_CONTEXTUALIZE_ENABLED=True). "
            "Each value is a dict with prompt_tokens / completion_tokens / total_tokens. "
            "Call types with zero tokens are omitted."
        ),
    )


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query to be answered.")
    namespace: Optional[str] = Field(
        default=None,
        description=(
            "Target Pinecone namespace. Defaults to the configured "
            "PINECONE_NAMESPACE when omitted."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of retrieved document chunks.",
    )
    use_web_fallback: bool = Field(
        default=True,
        description=(
            "Whether to fall back to web search when retrieval is weak. "
            "Requires a configured Tavily API key."
        ),
    )
    min_score: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=(
            "If the top retrieval score is below this threshold and "
            "use_web_fallback is true, a web search will be attempted."
        ),
    )
    max_web_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of web search results to fetch when enabled.",
    )
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description=(
            "Optional prior conversation history. "
            "Messages with role='user' or 'assistant' are supported."
        ),
    )


class SourceHit(BaseModel):
    source: str = Field(
        ...,
        description="Origin of the snippet (e.g. wiki, openalex, arxiv, web).",
    )
    title: str = Field(
        ...,
        description="Title of the underlying document or web page.",
    )
    url: str = Field(
        "",
        description="URL associated with the source, when available.",
    )
    score: float = Field(
        0.0,
        description=(
            "Relevance score from the vector store or a synthetic score for web search."
        ),
    )
    chunk_text: str = Field(
        ...,
        description="Text content of the retrieved chunk or web snippet.",
    )


class ChatTimings(BaseModel):
    retrieve_ms: float = Field(
        0.0,
        description="Time spent retrieving from Pinecone, in milliseconds.",
    )
    rerank_ms: float = Field(
        0.0,
        description=(
            "Time spent calling the Pinecone hosted reranker, in milliseconds. "
            "Zero when RAG_RERANK_ENABLED is False (the default)."
        ),
    )
    web_ms: float = Field(
        0.0,
        description="Time spent calling web search tools, in milliseconds.",
    )
    generate_ms: float = Field(
        0.0,
        description="Time spent generating the answer with the LLM, in milliseconds.",
    )
    faithfulness_ms: float = Field(
        0.0,
        description=(
            "Time spent on the LLM-judge faithfulness check, in milliseconds. "
            "Zero when RAG_FAITHFULNESS_ENABLED is False (the default)."
        ),
    )
    total_ms: float = Field(
        0.0,
        description="End-to-end time from request receipt to response, in milliseconds.",
    )


class ChatTraceMetadata(BaseModel):
    langsmith_project: Optional[str] = Field(
        default=None,
        description="LangSmith project name associated with this trace, if any.",
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether LangSmith / LangChain tracing was enabled for this call.",
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Generated answer text.")
    sources: List[SourceHit] = Field(
        default_factory=list,
        description="List of document or web snippets used as context.",
    )
    timings: ChatTimings = Field(
        default_factory=ChatTimings,
        description="Timing information for key phases of the pipeline.",
    )
    trace: ChatTraceMetadata = Field(
        ...,
        description="Tracing configuration metadata for observability.",
    )
    insufficient_context: bool = Field(
        default=False,
        description=(
            "True when no usable context survived retrieval and per-chunk score "
            "filtering.  The answer is the deterministic abstention message; the "
            "Groq LLM was NOT called.  Callers can use this flag to distinguish a "
            "genuine model-generated answer from an abstention without parsing the "
            "answer text."
        ),
    )
    grounded: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the generated answer is grounded in the retrieved context, "
            "as determined by the LLM judge (RAG_FAITHFULNESS_ENABLED=True). "
            "None when the judge was not called (flag OFF or abstention path). "
            "Distinct from insufficient_context: insufficient_context means the "
            "LLM was never called; grounded=False means the LLM answered but "
            "the answer is not well-supported by the context."
        ),
    )
    faithfulness_score: Optional[float] = Field(
        default=None,
        description=(
            "Score 0.0-1.0 from the faithfulness judge indicating the proportion "
            "of answer claims supported by context.  None when the judge was not "
            "called or JSON parsing of the judge response failed."
        ),
    )
    unverified_citations: List[int] = Field(
        default_factory=list,
        description=(
            "Citation numbers [n] found in the answer that reference chunks "
            "outside the valid range [1, len(sources)].  Set by the deterministic "
            "citation check (always runs, even when RAG_FAITHFULNESS_ENABLED=False). "
            "Empty list means all citations are in range or the answer has none."
        ),
    )
    crag_iterations: int = Field(
        default=0,
        description=(
            "Number of CRAG corrective iterations performed before reaching decide_next.  "
            "0 when RAG_CRAG_ENABLED is False, or when the initial retrieval was graded "
            "as good (top cosine score >= RAG_CRAG_GOOD_SCORE).  Maximum value is "
            "RAG_CRAG_MAX_ITERS (the hard loop bound)."
        ),
    )
    corrective_action: Optional[str] = Field(
        default=None,
        description=(
            "Action taken by the CRAG loop, or None when no correction occurred.  "
            "Current value: 'rewrite' (query was rewritten via LLM and Pinecone was "
            "re-queried).  None when RAG_CRAG_ENABLED is False or retrieval was graded good."
        ),
    )
    contextualized_query: Optional[str] = Field(
        default=None,
        description=(
            "The rewritten standalone query used for retrieval when T2.5 history-aware "
            "contextualization fired (RAG_CONTEXTUALIZE_ENABLED=True AND prior chat_history "
            "was present).  None when the feature is OFF, no history was present (first "
            "turn), or the rewrite fell back to the original due to an error.  "
            "Distinct from corrective_action (CRAG): contextualized_query reflects a "
            "pre-retrieval rewrite using history; CRAG rewrites post-weak-retrieval."
        ),
    )
    usage: Optional[ChatTokenUsage] = Field(
        default=None,
        description=(
            "Per-request token usage and estimated cost across ALL LLM calls "
            "(generation, faithfulness judge, CRAG rewrite, contextualize rewrite).  "
            "None on cached responses.  Token counts are ACTUAL values from the Groq "
            "API response.  Cost is an ESTIMATE — see ChatTokenUsage.estimated_cost_usd."
        ),
    )