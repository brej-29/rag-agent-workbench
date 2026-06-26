import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from a local .env file if present
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    APP_NAME: str = Field(default="rag-agent-workbench")
    APP_VERSION: str = Field(default="0.1.0")

    # Pinecone
    PINECONE_API_KEY: str = Field(..., description="Pinecone API key")
    PINECONE_INDEX_NAME: str = Field(
        ..., description="Name of the Pinecone index (used for configuration checks)"
    )
    PINECONE_HOST: str = Field(
        ..., description="Pinecone index host URL for data-plane operations"
    )
    PINECONE_NAMESPACE: str = Field(
        default="dev", description="Default Pinecone namespace"
    )
    PINECONE_TEXT_FIELD: str = Field(
        default="chunk_text",
        description=(
            "Text field name used by the Pinecone integrated embedding index. "
            "For example, set to 'content' if your index field_map uses that name."
        ),
    )
    # Embedding model and dimension used to CREATE the index (scripts/create_index.py)
    # and VERIFIED at startup (pinecone_store.init_pinecone logs the live values from
    # pc.describe_index — T1.1).  Changing either requires recreating the index.
    PINECONE_EMBED_MODEL: str = Field(
        default="llama-text-embed-v2",
        description=(
            "Pinecone integrated embedding model.  Must match the model the index "
            "was created with.  Used by scripts/create_index.py and logged at "
            "startup for drift detection."
        ),
    )
    PINECONE_EMBED_DIMENSION: int = Field(
        default=1024,
        description=(
            "Vector dimension of the Pinecone integrated embedding index.  "
            "llama-text-embed-v2 supports 384–2048; 1024 is the default and "
            "the value in use.  Changing this requires full index recreation."
        ),
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Application log level")

    # HTTP client defaults
    HTTP_TIMEOUT_SECONDS: float = Field(
        default=10.0, description="Default timeout for outbound HTTP requests"
    )
    HTTP_MAX_RETRIES: int = Field(
        default=3, description="Max retries for outbound HTTP requests"
    )

    # Groq / LLM
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        description="Groq API key (required for /chat endpoints)",
    )
    GROQ_BASE_URL: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Groq OpenAI-compatible base URL",
    )
    GROQ_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Default Groq chat model used for /chat",
    )

    # Web search / Tavily
    TAVILY_API_KEY: Optional[str] = Field(
        default=None,
        description="Tavily API key for web search fallback (optional)",
    )

    # RAG defaults
    RAG_DEFAULT_TOP_K: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Default number of documents to retrieve for RAG",
    )
    RAG_MIN_SCORE: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Default minimum relevance score to trust retrieval without web fallback",
    )
    RAG_MIN_CHUNK_SCORE: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description=(
            "Per-chunk cosine score floor for Pinecone vector chunks.  Chunks below "
            "this threshold are excluded from the context window before generation.  "
            "Distinct from RAG_MIN_SCORE (the web-fallback routing threshold, line 77): "
            "RAG_MIN_SCORE decides whether to invoke Tavily web search; "
            "RAG_MIN_CHUNK_SCORE decides which individual Pinecone chunks are usable "
            "context.  Set to 0.20 as a SAFETY BOUND derived from the T2.2 top-k sweep "
            "(eval/run_sweep.py, n=30 queries): the minimum cosine score of any retrieved "
            "chunk belonging to a golden-relevant doc was 0.2368 across all queries.  "
            "Setting the floor below 0.2368 guarantees no known-relevant chunk is dropped "
            "from the eval set.  0.20 is NOT an empirically-tuned optimum; sharp floor "
            "tuning requires a precision-oriented eval, graded relevance labels at chunk "
            "level, or a larger corpus where low-quality noise chunks are retrievable."
        ),
    )
    RAG_MAX_WEB_RESULTS: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of web search results to fetch when using Tavily",
    )

    # Two-stage reranking (disabled by default; enable only after A/B eval justifies it)
    # Relationship to existing retrieval settings:
    #   RAG_DEFAULT_TOP_K  — final number of chunks delivered to the LLM (unchanged)
    #   RAG_MIN_SCORE      — cosine routing threshold for web-fallback (unchanged, cosine-only)
    #   RAG_MIN_CHUNK_SCORE — cosine per-chunk floor applied BEFORE rerank (unchanged, cosine-only)
    #   RAG_RERANK_CANDIDATES — wider first-stage pool; must be >= RAG_DEFAULT_TOP_K
    # Rerank scores are a DIFFERENT scale/distribution from cosine — the cosine thresholds
    # above are never applied to rerank scores.
    RAG_RERANK_ENABLED: bool = Field(
        default=False,
        description=(
            "Enable two-stage retrieval: dense-retrieve RAG_RERANK_CANDIDATES candidates "
            "then rerank with the Pinecone hosted reranker.  Default OFF.  Enable only "
            "after an A/B run (`make eval-ab`) justifies it."
        ),
    )
    RAG_RERANK_MODEL: str = Field(
        default="bge-reranker-v2-m3",
        description=(
            "Pinecone Inference reranker model.  Eval-tunable. "
            "Operator must confirm this model is available on their Pinecone plan — "
            "plan availability varies.  'pinecone-rerank-v0' is a development-tier "
            "alternative with lower throughput.  See https://docs.pinecone.io/models/overview"
        ),
    )
    RAG_RERANK_CANDIDATES: int = Field(
        default=20,
        ge=1,
        le=200,
        description=(
            "Number of candidates to retrieve in the first (dense) stage when reranking "
            "is enabled.  Eval-tunable: larger pools give the reranker more to work with "
            "at the cost of higher rerank inference latency.  Must be >= RAG_DEFAULT_TOP_K "
            "(the final top_k delivered to the LLM); values below top_k are clamped up. "
            "Ignored when RAG_RERANK_ENABLED is False."
        ),
    )

    # Faithfulness / grounding check (T2.2)
    RAG_FAITHFULNESS_ENABLED: bool = Field(
        default=False,
        description=(
            "Enable the LLM-judge faithfulness check in format_response.  Defaults to "
            "False to avoid extra LLM call cost on every request.  When False, the "
            "cheap deterministic citation-marker check still runs (verify_citations); "
            "only the judge call is skipped.  Set to True to populate grounded and "
            "faithfulness_score in ChatResponse."
        ),
    )
    RAG_FAITHFULNESS_THRESHOLD: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum faithfulness_score (from the LLM judge) required to set "
            "grounded=True in ChatResponse.  PLACEHOLDER — this value is NOT "
            "evidence-backed.  Calibrate against an answer-level eval set (answer + "
            "context + human grounded label) before treating it as justified.  Setting "
            "it too high yields false 'ungrounded' flags; too low defeats the check."
        ),
    )

    # Corrective RAG (CRAG) — disabled by default; OFF path byte-for-byte identical to baseline
    RAG_CRAG_ENABLED: bool = Field(
        default=False,
        description=(
            "Enable the CRAG corrective retrieval loop.  When ON, initial retrieval is "
            "graded by top cosine score; if weak, the query is rewritten by the Groq LLM "
            "and Pinecone is re-queried up to RAG_CRAG_MAX_ITERS times.  "
            "Default OFF — the retrieve->decide_next path is byte-for-byte identical to "
            "the baseline when this flag is False.  Composed with the existing cosine "
            "floor, web fallback, abstention, and faithfulness checks."
        ),
    )
    RAG_CRAG_MAX_ITERS: int = Field(
        default=2,
        ge=1,
        le=10,
        description=(
            "Hard maximum number of corrective iterations for the CRAG loop.  The loop "
            "ALWAYS terminates after this many iterations regardless of the grade outcome.  "
            "Non-negotiable safety guard — prevents runaway loops.  Ignored when "
            "RAG_CRAG_ENABLED is False."
        ),
    )
    RAG_CRAG_GOOD_SCORE: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine score threshold for grading retrieval in the CRAG loop.  If the top "
            "retrieved chunk's cosine score is >= this value, retrieval is graded 'good' "
            "and no corrective action is taken.  PLACEHOLDER — not empirically tuned.  "
            "Calibrate against an answer-quality eval before treating it as justified.  "
            "Ignored when RAG_CRAG_ENABLED is False."
        ),
    )

    # History-aware query contextualization (T2.5)
    # Distinct from CRAG rewrite (T2.4): this triggers BEFORE retrieval using
    # conversation history; CRAG triggers AFTER weak retrieval on the current query.
    RAG_CONTEXTUALIZE_ENABLED: bool = Field(
        default=False,
        description=(
            "Enable history-aware query contextualization before retrieval.  When ON "
            "and the request includes prior chat_history, the follow-up query is rewritten "
            "into a standalone question by the Groq LLM before Pinecone retrieval runs.  "
            "When OFF (the default), retrieval runs on the raw current message — identical "
            "to the baseline behavior.  When ON but no history is present (first turn), "
            "the LLM is NOT called and retrieval runs on the raw message unchanged.  "
            "Distinct from RAG_CRAG_ENABLED (T2.4): CRAG corrects weak retrieval on a "
            "single turn; this corrects context-free fragments across turns."
        ),
    )

    # Operational toggles
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable SlowAPI rate limiting middleware when true.",
    )
    CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable in-memory TTL caching for /search and /chat when true.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()  # type: ignore[call-arg]


def get_env_bool(name: str, default: bool = False) -> bool:
    """Utility to parse boolean flags from environment variables."""
    raw: Optional[str] = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}