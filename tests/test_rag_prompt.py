"""
CI-safe unit tests for rag_prompt.py pure functions.

Does NOT duplicate the filter_chunks_by_score tests already in test_retrieval_gating.py.

Covers:
  - build_context_string: empty sources, numbered citations, source label,
    title, url inclusion/omission, chunk_text inclusion/omission,
    missing-key fallback, separator
  - build_rag_messages: system prompt first, human prompt last, question
    injected, citation instruction present, chat_history ordering (user →
    HumanMessage, assistant → AIMessage, unknown role → HumanMessage),
    empty-content history items skipped, sources injected into final message

Zero network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

from app.services.prompts.rag_prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    build_context_string,
    build_rag_messages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _src(
    source: str = "wiki",
    title: str = "Test Title",
    url: str = "https://example.com",
    chunk_text: str = "Chunk content.",
) -> Dict[str, Any]:
    return {"source": source, "title": title, "url": url, "chunk_text": chunk_text}


# ---------------------------------------------------------------------------
# build_context_string
# ---------------------------------------------------------------------------

class TestBuildContextString:
    def test_empty_sources_returns_empty_string(self):
        assert build_context_string([]) == ""

    def test_single_source_has_citation_number_one(self):
        ctx = build_context_string([_src()])
        assert "[1]" in ctx

    def test_multiple_sources_numbered_sequentially(self):
        ctx = build_context_string([_src(title="A"), _src(title="B"), _src(title="C")])
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "[3]" in ctx

    def test_source_label_in_header(self):
        ctx = build_context_string([_src(source="arxiv")])
        assert "(arxiv)" in ctx

    def test_title_in_header(self):
        ctx = build_context_string([_src(title="Quantum Entanglement")])
        assert "Quantum Entanglement" in ctx

    def test_url_included_when_present(self):
        ctx = build_context_string([_src(url="https://example.com/paper")])
        assert "https://example.com/paper" in ctx

    def test_url_omitted_when_empty(self):
        ctx = build_context_string([{"source": "wiki", "title": "T", "url": "", "chunk_text": "C"}])
        assert "http" not in ctx

    def test_chunk_text_included_when_present(self):
        ctx = build_context_string([_src(chunk_text="Important finding here.")])
        assert "Important finding here." in ctx

    def test_chunk_text_omitted_when_empty(self):
        src = {"source": "wiki", "title": "T", "url": "https://x.com", "chunk_text": ""}
        ctx = build_context_string([src])
        # output should contain header + url but no empty element from chunk_text
        parts = ctx.split("\n\n")
        assert len(parts) == 2  # [header, url] — no third empty entry

    def test_missing_source_key_falls_back_to_unknown(self):
        src = {"title": "T", "url": "", "chunk_text": "C"}
        ctx = build_context_string([src])
        assert "unknown" in ctx

    def test_separator_between_elements_is_double_newline(self):
        ctx = build_context_string([_src(url="", chunk_text="body")])
        # header and body are separated by \n\n
        assert "\n\n" in ctx

    def test_two_sources_both_appear_in_output(self):
        ctx = build_context_string([
            _src(title="Paper One", chunk_text="finding one"),
            _src(title="Paper Two", chunk_text="finding two"),
        ])
        assert "finding one" in ctx
        assert "finding two" in ctx


# ---------------------------------------------------------------------------
# build_rag_messages
# ---------------------------------------------------------------------------

class TestBuildRagMessages:
    def test_first_message_is_system_message(self):
        msgs = build_rag_messages([], "question?", [])
        assert isinstance(msgs[0], SystemMessage)

    def test_system_prompt_content_matches_constant(self):
        msgs = build_rag_messages([], "question?", [])
        assert msgs[0].content == SYSTEM_PROMPT

    def test_last_message_is_human_message(self):
        msgs = build_rag_messages([], "question?", [])
        assert isinstance(msgs[-1], HumanMessage)

    def test_question_appears_in_last_message(self):
        msgs = build_rag_messages([], "What is RAG?", [])
        assert "What is RAG?" in msgs[-1].content

    def test_citation_instruction_in_last_message(self):
        msgs = build_rag_messages([], "question?", [_src()])
        # USER_PROMPT_TEMPLATE includes "Use inline citations like [1], [2]"
        assert "inline citations" in msgs[-1].content

    def test_empty_history_yields_exactly_two_messages(self):
        msgs = build_rag_messages([], "q", [])
        assert len(msgs) == 2  # SystemMessage + HumanMessage(question)

    def test_user_history_item_becomes_human_message(self):
        msgs = build_rag_messages([{"role": "user", "content": "Previous Q"}], "q", [])
        assert isinstance(msgs[1], HumanMessage)
        assert msgs[1].content == "Previous Q"

    def test_assistant_history_item_becomes_ai_message(self):
        msgs = build_rag_messages([{"role": "assistant", "content": "Previous A"}], "q", [])
        assert isinstance(msgs[1], AIMessage)
        assert msgs[1].content == "Previous A"

    def test_unknown_role_defaults_to_human_message(self):
        msgs = build_rag_messages([{"role": "mystery", "content": "Injected"}], "q", [])
        assert isinstance(msgs[1], HumanMessage)

    def test_empty_content_history_items_are_skipped(self):
        history = [{"role": "user", "content": ""}]
        msgs = build_rag_messages(history, "q", [])
        # empty-content item must not add a message
        assert len(msgs) == 2

    def test_none_content_history_item_is_skipped(self):
        history = [{"role": "user", "content": None}]
        msgs = build_rag_messages(history, "q", [])
        assert len(msgs) == 2

    def test_sources_chunk_text_injected_into_final_message(self):
        sources = [_src(chunk_text="Key context sentence for the test.")]
        msgs = build_rag_messages([], "q", sources)
        assert "Key context sentence for the test." in msgs[-1].content

    def test_full_conversation_order(self):
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]
        msgs = build_rag_messages(history, "Second question", [])
        assert len(msgs) == 4
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], HumanMessage)  # "First question"
        assert msgs[1].content == "First question"
        assert isinstance(msgs[2], AIMessage)       # "First answer"
        assert msgs[2].content == "First answer"
        assert isinstance(msgs[3], HumanMessage)   # "Second question" + context
        assert "Second question" in msgs[3].content
