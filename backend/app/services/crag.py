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

import re
from typing import Any, Dict, Literal, Optional

from app.core.cost_accounting import extract_token_usage
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rewrite guard — deterministic, embedder-independent.  The LLM rewriter may only
# produce a *search query*; anything else keeps the original query (fails safe).
# ---------------------------------------------------------------------------

# Instruction-override / exfiltration phrasing.  A match only disables the optional
# LLM rewrite (the query is still answered or abstained on normally), so a false
# positive costs one skipped correction, never a wrong answer.
_OVERRIDE_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\b[^.?!]{0,40}?"
    r"\b(?:previous|prior|above|earlier|all|your|the)\b[^.?!]{0,20}?"
    r"\b(?:instructions?|prompts?|rules|directions|guidelines|messages?)\b"
    r"|\b(?:reveal|print|show|repeat|output|leak|display|dump|tell\s+me)\b[^.?!]{0,30}?"
    r"\b(?:system|developer|hidden|initial|secret)\s+(?:prompt|instructions?|message|rules)\b"
    r"|\byou\s+are\s+now\b",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"\bi['’]?m sorry\b|\bas an ai\b"
    r"|\bi (?:can['’]?t|cannot|won['’]?t|will not|am unable to|am not able to) "
    r"(?:help|comply|assist|do|provide|share|answer)\b",
    re.IGNORECASE,
)
_SENTENCE_BREAK_RE = re.compile(r"[.!?]\s+[A-Z]")
_NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
_MAX_QUERY_WORDS = 32


def looks_like_instruction_override(query: str) -> bool:
    """True when the user query reads as an attempt to override or exfiltrate instructions.

    Such queries are never handed to the LLM rewriter: observed live, the rewriter
    laundered "Ignore all previous instructions..." into benign questions and the
    pipeline then answered a question the user never asked.
    """
    return bool(_OVERRIDE_RE.search(query or ""))


def validate_rewrite(original_query: str, rewritten: str) -> Optional[str]:
    """Return a rejection reason for a rewrite that is not a plain search query, else None.

    Reasons: 'empty', 'refusal', 'answer' (multi-sentence or longer than a query),
    'new_facts' (numbers absent from the original: the rewriter must not inject
    knowledge, e.g. a hallucinated score).
    """
    text = (rewritten or "").strip()
    if not text:
        return "empty"
    if _REFUSAL_RE.search(text):
        return "refusal"
    if len(text.split()) > _MAX_QUERY_WORDS or _SENTENCE_BREAK_RE.search(text):
        return "answer"
    original_numbers = set(_NUMBER_RE.findall(original_query or ""))
    if any(n not in original_numbers for n in _NUMBER_RE.findall(text)):
        return "new_facts"
    return None


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
    text, _ = rewrite_query_with_usage(original_query, llm)
    return text


def rewrite_query_with_usage(
    original_query: str,
    llm: Any,
) -> tuple[str, Dict[str, int]]:
    """Like rewrite_query but also returns actual token usage from the response.

    Returns (rewritten_text, usage_dict) where usage_dict has keys
    prompt_tokens / completion_tokens / total_tokens from the ACTUAL Groq API
    response.  Returns zeros on error/fallback — never estimated.

    rewrite_query() delegates to this function so both functions share one
    implementation and the CRAG tests (which mock get_llm at graph level) are
    unaffected by the additional return value.
    """
    from app.services.prompts.query_rewrite_prompt import build_query_rewrite_messages  # noqa: PLC0415

    _empty: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    messages = build_query_rewrite_messages(original_query)
    try:
        response = llm.invoke(messages)
        text = str(getattr(response, "content", "") or response).strip()
        usage = extract_token_usage(response)
        if text:
            logger.info("CRAG query rewrite: '%s' -> '%s'", original_query[:80], text[:80])
            return text, usage
    except Exception as exc:  # noqa: BLE001
        logger.warning("CRAG query rewrite failed (%s); using original query", exc)
    return original_query, _empty
