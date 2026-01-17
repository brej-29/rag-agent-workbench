from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

SYSTEM_PROMPT = """You are a focused research assistant for a retrieval-augmented generation (RAG) system.

You MUST:
- Answer the user's question using ONLY the provided context snippets.
- Treat each context snippet as a citation, referenced inline as [1], [2], etc.
- Prefer concise, clear explanations over long essays.
- Never fabricate facts that are not supported by the context.

If the context is insufficient to answer the question:
- Say that you do not know based on the current context.
- Suggest that the caller enable or use web search fallback for a more complete answer.
"""

USER_PROMPT_TEMPLATE = """You are given context snippets retrieved from a vector store and optionally from web search.

Each snippet is numbered like [1], [2], etc. Use these numbers to cite sources inline in your answer.

Context:
{context}

User question:
{question}

Instructions:
- Use the context to answer the question.
- Use inline citations like [1], [2] whenever you rely on a snippet.
- If you cannot answer from the context, say so explicitly and recommend using web search fallback.
"""


def build_context_string(sources: List[Dict[str, Any]]) -> str:
    """Format retrieved and web results into a numbered context block.

    Each source is expected to have keys:
      - source (e.g. wiki, arxiv, openalex, web)
      - title
      - url (optional)
      - chunk_text
    """
    lines: List[str] = []
    for idx, src in enumerate(sources, start=1):
        source_label = src.get("source") or "unknown"
        title = src.get("title") or ""
        url = src.get("url") or ""
        chunk_text = src.get("chunk_text") or ""

        header_parts = [f"[{idx}] ({source_label})"]
        if title:
            header_parts.append(title)
        header = " ".join(header_parts)

        lines.append(header)
        if url:
            lines.append(url)
        if chunk_text:
            lines.append(chunk_text)

    return "\n\n".join(lines)


def build_user_prompt(question: str, context: str) -> str:
    """Render the user-facing prompt given a question and context string."""
    return USER_PROMPT_TEMPLATE.format(question=question, context=context)


def build_rag_messages(
    chat_history: List[Dict[str, str]],
    question: str,
    sources: List[Dict[str, Any]],
) -> List[BaseMessage]:
    """Build a LangChain message list for the RAG chat interaction.

    chat_history is a list of dicts with keys:
      - role: "user" | "assistant"
      - content: str
    """
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    for item in chat_history:
        role = item.get("role")
        content = item.get("content") or ""
        if not content:
            continue
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            # Default to user if role is unknown
            messages.append(HumanMessage(content=content))

    context = build_context_string(sources)
    user_prompt = build_user_prompt(question=question, context=context)
    messages.append(HumanMessage(content=user_prompt))

    return messages