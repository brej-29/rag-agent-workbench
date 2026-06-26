"""
Query rewrite prompt for the CRAG corrective retrieval loop.

INTENTIONALLY separate from rag_prompt.py (generation) and faithfulness_prompt.py
(grounding judge) — each prompt serves a distinct role and must not be entangled.
"""

from __future__ import annotations

from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

QUERY_REWRITE_SYSTEM_PROMPT = """\
You are a retrieval query optimizer for a knowledge-base question-answering system.
The previous retrieval attempt did not return sufficiently relevant documents.

Your task is to rewrite the query so that a dense-vector search is more likely to find
relevant passages.  Strategies: use different terminology, be more specific, decompose
a compound question into its most important sub-topic, or remove vague phrasing.

Return ONLY the rewritten query text — no explanation, no preamble, no quotation marks.\
"""

_USER_TEMPLATE = """\
Original query: {original_query}

Rewritten query:\
"""


def build_query_rewrite_messages(original_query: str) -> List[BaseMessage]:
    """Build LangChain messages for the query-rewrite LLM call.

    Args:
        original_query: The query that produced weak retrieval results.

    Returns:
        [SystemMessage, HumanMessage] ready for llm.invoke().
    """
    return [
        SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=_USER_TEMPLATE.format(original_query=original_query)),
    ]
