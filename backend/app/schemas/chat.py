from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(
        ...,
        description="Role of the message author (user or assistant).",
    )
    content: str = Field(..., description="Message text content.")


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
    web_ms: float = Field(
        0.0,
        description="Time spent calling web search tools, in milliseconds.",
    )
    generate_ms: float = Field(
        0.0,
        description="Time spent generating the answer with the LLM, in milliseconds.",
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