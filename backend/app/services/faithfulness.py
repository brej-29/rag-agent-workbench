"""
Faithfulness and grounding verification for generated answers.

Two independent, separately importable layers:

  1. verify_citations(answer_text, context_sources) — deterministic, zero model calls.
     Parses [n] citation markers in the answer and checks each n is in the range
     [1, len(context_sources)].  Returns out-of-range marker numbers.

  2. judge_faithfulness(answer_text, retrieved_context, llm) — LLM-as-judge.
     Calls the existing Groq LLM (no new provider).  Returns a FaithfulnessVerdict.
     On JSON parse failure degrades to grounded=None / faithfulness_score=None rather
     than raising — the caller must NOT block the response on "unknown".

Design: neither layer uses the retrieval embedder.  The deterministic check is purely
lexical; the judge uses a language model as a semantic reasoner over the already-
retrieved text.  This avoids the circular-validation anti-pattern of re-embedding
the answer in the same vector space the retriever uses, which would encode the
embedder's own biases into the faithfulness signal.

Both functions are importable by eval/faithfulness.py for offline scoring of
golden answer/context pairs without duplicating the implementation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger
from app.services.prompts.faithfulness_prompt import build_faithfulness_judge_messages

logger = get_logger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class FaithfulnessVerdict:
    """Structured output from the LLM-judge faithfulness check.

    grounded:           True/False when the judge produced a parseable verdict;
                        None when the judge was not called or JSON parsing failed.
    faithfulness_score: Float 0.0-1.0 from the judge; None on parse failure.
    rationale:          Judge's explanation, or an error token on failure.
    """

    grounded: Optional[bool]
    faithfulness_score: Optional[float]
    rationale: str


def verify_citations(
    answer_text: str,
    context_sources: List[Dict[str, Any]],
) -> List[int]:
    """Return citation numbers that are dangling (out of range).

    Parses every [n] marker in answer_text and checks n is in [1, len(context_sources)].
    No model call.

    Args:
        answer_text:     The generated answer, potentially containing [n] markers.
        context_sources: All source chunks forwarded to the LLM (Pinecone + web).

    Returns:
        Sorted list of out-of-range citation numbers found in the text.
        Empty list means all citations are valid (or the answer has no citations).
    """
    n_sources = len(context_sources)
    dangling: List[int] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(answer_text):
        n = int(match.group(1))
        if n not in seen:
            seen.add(n)
            if n < 1 or n > n_sources:
                dangling.append(n)
    return sorted(dangling)


def judge_faithfulness(
    answer_text: str,
    retrieved_context: str,
    llm: Any,
) -> FaithfulnessVerdict:
    """LLM-as-judge: assess whether answer claims are grounded in the context.

    Uses the existing Groq LLM — no new provider.  The judge is instructed to
    return a structured JSON verdict.  JSON parse failure degrades gracefully to
    grounded=None / faithfulness_score=None; the caller must treat "unknown" as
    acceptable and must not raise or block the response.

    Args:
        answer_text:       The generated answer to evaluate.
        retrieved_context: The context string that was supplied to the generation LLM.
        llm:               LangChain-compatible LLM instance (e.g. ChatOpenAI / Groq).

    Returns:
        FaithfulnessVerdict with grounded, faithfulness_score, and rationale.
    """
    messages = build_faithfulness_judge_messages(answer_text, retrieved_context)
    try:
        response = llm.invoke(messages)
        raw: str = str(getattr(response, "content", "") or response)
    except Exception as exc:  # noqa: BLE001
        logger.error("Faithfulness judge LLM call failed: %s", exc)
        return FaithfulnessVerdict(
            grounded=None,
            faithfulness_score=None,
            rationale=f"judge_call_error: {exc}",
        )

    return _parse_verdict(raw)


def _parse_verdict(raw: str) -> FaithfulnessVerdict:
    """Parse the judge's text into a FaithfulnessVerdict.

    Handles markdown code-block wrapping and extracts the first JSON object.
    Degrades gracefully on all parse failures.
    """
    text = raw.strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Extract first JSON object between outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.warning("Faithfulness judge returned no JSON object: %r", raw[:200])
        return FaithfulnessVerdict(
            grounded=None,
            faithfulness_score=None,
            rationale="parse_error: no JSON object found",
        )

    json_text = text[start : end + 1]
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Faithfulness judge JSON decode error: %s — raw: %r", exc, raw[:200]
        )
        return FaithfulnessVerdict(
            grounded=None,
            faithfulness_score=None,
            rationale=f"parse_error: {exc}",
        )

    grounded_raw = data.get("grounded")
    score_raw = data.get("score")
    rationale_raw = str(
        data.get("rationale")
        or data.get("reason")
        or data.get("explanation")
        or ""
    )

    try:
        grounded: Optional[bool] = bool(grounded_raw) if grounded_raw is not None else None
    except Exception:  # noqa: BLE001
        grounded = None

    try:
        faithfulness_score: Optional[float] = (
            float(score_raw) if score_raw is not None else None
        )
        if faithfulness_score is not None:
            faithfulness_score = max(0.0, min(1.0, faithfulness_score))
    except Exception:  # noqa: BLE001
        faithfulness_score = None

    return FaithfulnessVerdict(
        grounded=grounded,
        faithfulness_score=faithfulness_score,
        rationale=rationale_raw,
    )
