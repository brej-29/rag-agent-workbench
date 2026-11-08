from unittest.mock import patch

import pytest

from app.core.config import Settings
from app.core.cost_accounting import estimate_cost_usd
from app.services.llm import groq_llm


def _settings(**kw):
    base = dict(PINECONE_API_KEY="x", PINECONE_INDEX_NAME="x", PINECONE_HOST="https://x.invalid", GROQ_API_KEY="x", OMNIROUTE_API_KEY=None, OMNIROUTE_BASE_URL=None)
    return Settings(**{**base, **kw})


def _build(**kw):
    groq_llm.get_llm.cache_clear()
    with patch.object(groq_llm, "get_settings", return_value=_settings(**kw)):
        try:
            return groq_llm.get_llm()
        finally:
            groq_llm.get_llm.cache_clear()


def test_default_chain_is_gpt_oss_with_fallback_and_low_reasoning():
    llm = _build()
    assert llm.runnable.model_name == "openai/gpt-oss-20b"
    assert [f.model_name for f in llm.fallbacks] == ["openai/gpt-oss-120b"]
    assert llm.runnable.reasoning_effort == "low"


def test_fallbacks_are_ordered_deduplicated_and_optional():
    llm = _build(GROQ_MODEL="a", LLM_FALLBACK_MODELS=" b, a ,c,, ")
    assert [f.model_name for f in llm.fallbacks] == ["b", "c"]
    single = _build(GROQ_MODEL="a", LLM_FALLBACK_MODELS="", LLM_REASONING_EFFORT="")
    assert single.model_name == "a" and single.reasoning_effort is None


def test_default_models_are_priced():
    for model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
        assert estimate_cost_usd(1_000_000, 1_000_000, model) is not None


OMNI = dict(OMNIROUTE_API_KEY="omni-secret", OMNIROUTE_BASE_URL="https://omni.example/v1")


def test_omniroute_replaces_groq_when_its_key_is_set():
    llm = _build(**OMNI)
    assert llm.runnable.model_name == "free-normal"
    assert [f.model_name for f in llm.fallbacks] == ["free-advanced"]
    assert llm.runnable.openai_api_base == "https://omni.example/v1"
    assert llm.runnable.openai_api_key.get_secret_value() == "omni-secret"
    assert llm.runnable.max_retries == 1  # the gateway already falls back; retries multiply quota use
    assert "omni-secret" not in repr(llm)


@pytest.mark.parametrize("url", [None, "http://omni.example/v1", "ftp://x"])
def test_omniroute_refuses_a_missing_or_plain_http_url(url):
    with pytest.raises(RuntimeError, match="OMNIROUTE_BASE_URL") as e:
        _build(OMNIROUTE_API_KEY="omni-secret", OMNIROUTE_BASE_URL=url)
    assert "omni-secret" not in str(e.value)


def test_omniroute_allows_plain_http_only_on_localhost():
    assert _build(OMNIROUTE_API_KEY="k", OMNIROUTE_BASE_URL="http://127.0.0.1:20128/v1").runnable.model_name == "free-normal"


def test_generation_failures_are_classified_without_echoing_the_error():
    from app.services.chat.streaming import _FAILURE_MESSAGES, generation_failure

    class RateLimitError(Exception):
        status_code = 429

    class APITimeoutError(Exception):
        pass

    assert generation_failure(RateLimitError("key=sk-123")) == "busy"
    assert generation_failure(APITimeoutError()) == "timeout"
    assert generation_failure(ValueError("boom")) == "unavailable"
    assert all("sk-" not in m for m in _FAILURE_MESSAGES.values())
