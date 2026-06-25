"""
History-aware query contextualization prompt (T2.5).

DISTINCT from query_rewrite_prompt.py (T2.4 / CRAG):
  - This prompt triggers BEFORE retrieval, rewriting a follow-up question into a
    standalone question using the conversation history as input.
  - CRAG's rewrite triggers AFTER weak retrieval, rewriting to improve retrieval
    quality on the same turn.  The two rewrites have different triggers, different
    inputs, and serve different purposes.

Output contract: the model must return ONLY the rewritten standalone question with
no preamble, explanation, or quotation marks.  If the follow-up is already
self-contained, return it verbatim.
"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

CONTEXTUALIZE_SYSTEM_PROMPT = """\
You are a helpful assistant that transforms a follow-up question into a fully \
self-contained, standalone question.

You will receive:
  1. A conversation history (list of prior user and assistant messages).
  2. The latest follow-up question from the user.

Your task: rewrite the follow-up question so it is clear and complete on its own, \
without any reference to the conversation history.

Rules:
  - Preserve the intent of the follow-up exactly.
  - Replace pronouns (it, they, this, that, these, those) and ellipses with the \
specific nouns from the conversation history.
  - If the follow-up is already self-contained, return it verbatim.
  - Return ONLY the rewritten question — no explanation, no preamble, no quotation \
marks, no trailing punctuation changes.\
"""


def build_contextualize_messages(
    follow_up: str,
    chat_history: List[Dict[str, str]],
) -> List[BaseMessage]:
    """Build LangChain messages for the contextualize-follow-up call.

    Args:
        follow_up:    The current user message (potentially a fragment follow-up).
        chat_history: Prior messages as dicts with 'role' and 'content' keys.

    Returns:
        A list of BaseMessage objects ready to pass to llm.invoke().
    """
    history_lines: List[str] = []
    for msg in chat_history:
        role = str(msg.get("role") or "user").capitalize()
        content = str(msg.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content}")

    history_str = "\n".join(history_lines) if history_lines else "(no prior history)"

    human_content = (
        f"Conversation history:\n{history_str}\n\n"
        f"Follow-up question: {follow_up}\n\n"
        f"Standalone question:"
    )

    return [
        SystemMessage(content=CONTEXTUALIZE_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]
