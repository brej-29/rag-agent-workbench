"""
CRAG (Corrective RAG) helpers: retrieval grader and query rewriter.

Design constraints
------------------
- grade_retrieval uses cosine scores ALREADY computed by retrieve_context.
  It does NOT re-embed with the retrieval model — that would be circular validation.
- rewrite_query uses the existing Groq LLM client (passed in by the caller).
  No new LLM client, no new dependency.
- Both functions are pure / side-effect-free from the caller's perspective,
  making them trivially unit-testable without network calls.
"""

from __future__ import annotations

from typing import Any, Literal

from app.core.logging import get_logger

logger = get_logger(__name__)


def grade_retrieval(top_score: float, good_score: float) -> Literal["good", "weak"]:
    """Grade retrieval quality from the top cosine score already in state.

    Returns 'good' when top_score >= good_score, 'weak' otherwise.
    Uses scores from retrieve_context — no re-embedding, no circular validation.
    """
    return "good" if top_score >= good_score else "weak"


def rewrite_query(original_query: str, llm: Any) -> str:
    """Rewrite a query via the existing Groq LLM to improve retrieval on the next attempt.

    Falls back to the original query if the LLM returns empty text or raises.
    The llm argument is the caller's existing client — no new client is created.
    """
    from app.services.prompts.query_rewrite_prompt import build_query_rewrite_messages  # noqa: PLC0415

    messages = build_query_rewrite_messages(original_query)
    try:
        response = llm.invoke(messages)
        text = str(getattr(response, "content", "") or response).strip()
        if text:
            logger.info("CRAG query rewrite: '%s' -> '%s'", original_query[:80], text[:80])
            return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("CRAG query rewrite failed (%s); using original query", exc)
    return original_query
