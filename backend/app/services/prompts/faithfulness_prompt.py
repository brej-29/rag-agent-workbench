"""
Judge prompt builder for the faithfulness/grounding check.

INTENTIONALLY separate from rag_prompt.py — generation prompts and evaluation
prompts must not be entangled.  The judge must assess the answer without being
influenced by the same template that produced it.

The prompt instructs the model to use ONLY the provided context and return a
structured JSON object.  No markdown prose, no outside knowledge.
"""

from __future__ import annotations

from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

FAITHFULNESS_SYSTEM_PROMPT = """\
You are a strict faithfulness evaluator for a retrieval-augmented generation (RAG) system.

Your sole task is to assess whether an AI-generated answer is grounded in the provided context.

Rules:
- A claim is grounded if it can be directly supported or inferred from the provided context.
- A claim is ungrounded if it introduces facts not present in the context, even if true.
- Logical deductions that follow directly from context are acceptable.
- Do NOT use outside knowledge — only the provided context counts as evidence.
- Respond ONLY with a JSON object. No preamble. No explanation outside the JSON.\
"""

_USER_TEMPLATE = """\
Context snippets provided to the AI:
---
{context}
---

AI-generated answer to evaluate:
---
{answer}
---

Assess whether the answer's claims are supported ONLY by the context above.

Respond with a JSON object containing exactly these fields:
{{
  "grounded": <boolean — true if the answer is substantially grounded in the context>,
  "score": <float 0.0 to 1.0 — proportion of the answer's claims supported by the context>,
  "rationale": "<one to three sentences explaining your verdict>"
}}

Respond ONLY with the JSON object.\
"""


def build_faithfulness_judge_messages(
    answer_text: str,
    context_string: str,
) -> List[BaseMessage]:
    """Build the LangChain message list for the faithfulness judge LLM call.

    Args:
        answer_text:    The generated answer to evaluate.
        context_string: The context provided to the generation LLM, formatted
                        as a single string (use build_context_string from rag_prompt.py).

    Returns:
        [SystemMessage, HumanMessage] ready for llm.invoke().
    """
    user_content = _USER_TEMPLATE.format(
        context=context_string,
        answer=answer_text,
    )
    return [
        SystemMessage(content=FAITHFULNESS_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
