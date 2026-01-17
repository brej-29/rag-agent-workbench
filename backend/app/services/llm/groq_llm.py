from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return a configured Groq-backed ChatOpenAI model.

    This uses Groq's OpenAI-compatible API so we avoid additional heavy SDKs.
    The function is cached so the client is created once per process.
    """
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        # Configuration error rather than an upstream service outage.
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set GROQ_API_KEY in your environment "
            "to enable the /chat endpoints."
        )

    logger.info(
        "Initialising Groq ChatOpenAI client base_url=%s model=%s",
        settings.GROQ_BASE_URL,
        settings.GROQ_MODEL,
    )

    return ChatOpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url=settings.GROQ_BASE_URL,
        model=settings.GROQ_MODEL,
        temperature=0.2,
        timeout=settings.HTTP_TIMEOUT_SECONDS,
        max_retries=settings.HTTP_MAX_RETRIES,
    )