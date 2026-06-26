"""
History-aware query contextualization (T2.5).

Rewrites a follow-up question into a standalone question using conversation
history, so Pinecone retrieval doesn't run on a context-free fragment.

DISTINCT from CRAG query rewrite (crag.py / T2.4):
  - T2.5 (this): triggers BEFORE retrieval; input = current message + history.
    Fixes the multi-turn retrieval problem.
  - T2.4 (CRAG): triggers AFTER weak retrieval; input = current query alone.
    Fixes the retrieval-quality problem on a single turn.

Falls back to the original query on LLM error — a failed contextualization
must never break the request.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.core.cost_accounting import extract_token_usage
from app.core.logging import get_logger

logger = get_logger(__name__)

_EMPTY_USAGE: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def contextualize_followup(
    original_query: str,
    chat_history: List[Dict[str, str]],
    llm: Any,
) -> tuple[str, Dict[str, int]]:
    """Rewrite a follow-up query into a standalone query using conversation history.

    Args:
        original_query: The current user message (may be a fragment like "what about the second one?").
        chat_history:   Prior conversation turns as list of {role, content} dicts.
        llm:            Existing Groq LLM client — no new client created.

    Returns:
        (rewritten_query, usage_dict) where:
          - rewritten_query is the standalone form (falls back to original_query on error).
          - usage_dict has keys prompt_tokens, completion_tokens, total_tokens from
            the ACTUAL API response (zeros on error/fallback — never estimated).

    The caller is responsible for checking whether rewritten_query != original_query
    to determine if a rewrite actually occurred.
    """
    from app.services.prompts.contextualize_prompt import build_contextualize_messages  # noqa: PLC0415

    if not chat_history:
        return original_query, dict(_EMPTY_USAGE)

    messages = build_contextualize_messages(original_query, chat_history)
    try:
        response = llm.invoke(messages)
        text = str(getattr(response, "content", None) or "").strip()
        usage = extract_token_usage(response)
        if text:
            logger.info(
                "T2.5 contextualize: '%s' -> '%s'",
                original_query[:80],
                text[:80],
            )
            return text, usage
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "T2.5 contextualize failed (%s); falling back to original query.",
            exc,
        )

    return original_query, dict(_EMPTY_USAGE)
