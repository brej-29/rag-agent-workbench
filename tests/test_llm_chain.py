from unittest.mock import patch

from app.core.config import Settings
from app.core.cost_accounting import estimate_cost_usd
from app.services.llm import groq_llm


def _settings(**kw):
    base = dict(PINECONE_API_KEY="x", PINECONE_INDEX_NAME="x", PINECONE_HOST="https://x.invalid", GROQ_API_KEY="x")
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
