from functools import lru_cache
from typing import List, NamedTuple, Optional
from urllib.parse import urlparse

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class Provider(NamedTuple):
    name: str
    base_url: str
    api_key: Optional[str]
    models: List[str]
    max_retries: int


def _split(primary: str, fallbacks: str) -> List[str]:
    return list(dict.fromkeys([primary] + [m.strip() for m in fallbacks.split(",") if m.strip()]))


def provider() -> Provider:
    """OmniRoute when OMNIROUTE_API_KEY is set, else Groq.  Never logs or returns the key in errors."""
    s = get_settings()
    if s.OMNIROUTE_API_KEY:
        url = (s.OMNIROUTE_BASE_URL or "").strip()
        parsed = urlparse(url)
        if not (parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS)):
            raise RuntimeError(
                "OMNIROUTE_BASE_URL must be an https:// URL (plain http only for localhost), "
                "so the OmniRoute key is never sent unencrypted."
            )
        # The gateway already falls back between providers; client retries would multiply quota use.
        return Provider("omniroute", url, s.OMNIROUTE_API_KEY,
                        _split(s.OMNIROUTE_CHAT_MODEL, s.OMNIROUTE_FALLBACK_MODELS), 1)
    return Provider("groq", s.GROQ_BASE_URL, s.GROQ_API_KEY,
                    _split(s.GROQ_MODEL, s.LLM_FALLBACK_MODELS), s.HTTP_MAX_RETRIES)


def configured_models() -> List[str]:
    """Primary model followed by fallbacks, in call order, de-duplicated."""
    return provider().models


def _chat_model(model: str) -> ChatOpenAI:
    settings, p = get_settings(), provider()
    extra = {"reasoning_effort": settings.LLM_REASONING_EFFORT} if settings.LLM_REASONING_EFFORT else {}
    return ChatOpenAI(
        api_key=p.api_key,
        base_url=p.base_url,
        model=model,
        temperature=0.2,
        # ponytail: free gateway combos answer in 1-5 s but long judge JSON can take longer.
        timeout=max(settings.HTTP_TIMEOUT_SECONDS, 60.0) if p.name == "omniroute" else settings.HTTP_TIMEOUT_SECONDS,
        max_retries=p.max_retries,
        **extra,
    )


@lru_cache(maxsize=1)
def get_llm() -> Runnable:
    """Return the chat model chain: primary model with ordered fallbacks.

    OpenAI-compatible API: OmniRoute when OMNIROUTE_API_KEY is set, else Groq.
    If the primary call raises (retired model, rate limit, timeout) the next model
    is tried; for astream this applies until the first chunk is emitted.
    """
    p = provider()
    if not p.api_key:
        raise RuntimeError(
            "No LLM key is configured. Set OMNIROUTE_API_KEY and OMNIROUTE_BASE_URL "
            "(or GROQ_API_KEY) in the environment to enable the /chat endpoints."
        )
    settings = get_settings()
    logger.info(
        "Initialising chat model chain provider=%s host=%s models=%s reasoning_effort=%s",
        p.name, urlparse(p.base_url).hostname, p.models, settings.LLM_REASONING_EFFORT or "-",
    )
    primary, *fallbacks = [_chat_model(m) for m in p.models]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
