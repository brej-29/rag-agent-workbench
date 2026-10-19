from functools import lru_cache
from typing import List

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def configured_models() -> List[str]:
    """Primary model followed by fallbacks, in call order, de-duplicated."""
    settings = get_settings()
    models = [settings.GROQ_MODEL] + [
        m.strip() for m in settings.LLM_FALLBACK_MODELS.split(",") if m.strip()
    ]
    return list(dict.fromkeys(models))


def _chat_model(model: str) -> ChatOpenAI:
    settings = get_settings()
    extra = {"reasoning_effort": settings.LLM_REASONING_EFFORT} if settings.LLM_REASONING_EFFORT else {}
    return ChatOpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url=settings.GROQ_BASE_URL,
        model=model,
        temperature=0.2,
        timeout=settings.HTTP_TIMEOUT_SECONDS,
        max_retries=settings.HTTP_MAX_RETRIES,
        **extra,
    )


@lru_cache(maxsize=1)
def get_llm() -> Runnable:
    """Return the chat model chain: primary model with ordered fallbacks.

    Uses the OpenAI-compatible API (Groq by default; any compatible provider
    via GROQ_BASE_URL) so we avoid additional heavy SDKs.  If the primary call
    raises (retired model, rate limit, timeout) the next model is tried; for
    astream this applies until the first chunk is emitted.  Cached once per process.
    """
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        # Configuration error rather than an upstream service outage.
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set GROQ_API_KEY in your environment "
            "to enable the /chat endpoints."
        )

    models = configured_models()
    logger.info(
        "Initialising chat model chain base_url=%s models=%s reasoning_effort=%s",
        settings.GROQ_BASE_URL,
        models,
        settings.LLM_REASONING_EFFORT or "-",
    )
    primary, *fallbacks = [_chat_model(m) for m in models]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
