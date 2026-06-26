"""
Token count and estimated cost accounting for Groq LLM calls (T2.7).

PRICING TABLE
-------------
Source: https://console.groq.com/docs/models (Groq pricing page)
As-of date: 2026-06-25
IMPORTANT: MUST BE MANUALLY UPDATED when provider pricing changes.
            The estimates become stale the moment Groq adjusts prices.
            Check https://console.groq.com/docs/models before relying on
            cost figures for billing or budgeting decisions.

Token counts are ACTUAL values from the Groq API response (usage object),
not tokenizer estimates.  Cost figures derived from those actual counts using
this pricing table are ESTIMATES — they assume no discounts, promotions, or
plan-specific rates.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Pricing table — ONE place; update date when prices change
# ---------------------------------------------------------------------------
#
# Format: { model_id: { "input": $/1M tokens, "output": $/1M tokens } }
# The IDs must match the GROQ_MODEL values operators configure.
#
# As-of 2026-06-25. UPDATE THIS TABLE when Groq changes pricing.
_GROQ_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "llama-3.1-8b-instant": {
        "input": 0.05,   # $0.05 / 1M input tokens
        "output": 0.08,  # $0.08 / 1M output tokens
    },
    "llama-3.3-70b-versatile": {
        "input": 0.59,   # $0.59 / 1M input tokens
        "output": 0.79,  # $0.79 / 1M output tokens
    },
    "llama-3.1-70b-versatile": {
        "input": 0.59,
        "output": 0.79,
    },
    "llama-3.1-405b-reasoning": {
        "input": 3.00,
        "output": 3.00,
    },
    "mixtral-8x7b-32768": {
        "input": 0.24,
        "output": 0.24,
    },
    "gemma2-9b-it": {
        "input": 0.20,
        "output": 0.20,
    },
    # Add new models here when they are added to Groq.
    # Source: https://console.groq.com/docs/models
    # As-of 2026-06-25. UPDATE DATE WHEN PRICES CHANGE.
}

# Sentinel used when cost is unknown (model not in table or zero tokens).
_PRICING_AS_OF = "2026-06-25"


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> float | None:
    """Estimate total cost in USD for a single model call.

    Returns None when the model is not in the pricing table (so callers can
    distinguish "cost unknown" from "cost = $0.00").

    IMPORTANT: This is an ESTIMATE based on the as-of-date pricing table.
    It does not account for discounts, free-tier credits, or batch pricing.
    Always verify current pricing at https://console.groq.com/docs/models.
    """
    if model not in _GROQ_PRICING_USD_PER_1M:
        return None
    rates = _GROQ_PRICING_USD_PER_1M[model]
    cost = (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000
    return cost


def extract_token_usage(response: Any) -> dict[str, int]:
    """Extract token usage from a LangChain / Groq LLM response.

    Tries two standard LangChain attribute paths:
      1. response.usage_metadata (langchain-core 0.2+ AIMessage attribute)
         Keys: input_tokens, output_tokens, total_tokens
      2. response.response_metadata["token_usage"] (OpenAI-compatible fallback)
         Keys: prompt_tokens, completion_tokens, total_tokens

    Returns a dict with keys prompt_tokens / completion_tokens / total_tokens.
    Returns zeros when usage is unavailable (e.g. mock responses in tests).
    Never raises.
    """
    _empty = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Path 1: usage_metadata dict (langchain-core 0.2+ AIMessage)
    try:
        usage_meta = getattr(response, "usage_metadata", None)
        if isinstance(usage_meta, dict) and "total_tokens" in usage_meta:
            return {
                "prompt_tokens": int(usage_meta.get("input_tokens") or 0),
                "completion_tokens": int(usage_meta.get("output_tokens") or 0),
                "total_tokens": int(usage_meta.get("total_tokens") or 0),
            }
    except Exception:  # noqa: BLE001
        pass

    # Path 2: response_metadata["token_usage"] (OpenAI-compatible)
    try:
        resp_meta = getattr(response, "response_metadata", None)
        if isinstance(resp_meta, dict):
            token_usage = resp_meta.get("token_usage", {})
            if isinstance(token_usage, dict) and "total_tokens" in token_usage:
                return {
                    "prompt_tokens": int(token_usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(token_usage.get("completion_tokens") or 0),
                    "total_tokens": int(token_usage.get("total_tokens") or 0),
                }
    except Exception:  # noqa: BLE001
        pass

    return _empty
