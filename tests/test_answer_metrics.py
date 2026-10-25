"""Answer-quality / abstention eval: pure metric functions and the claim judge parser.
Offline; the live runner is eval/run_answers.py."""
from unittest.mock import MagicMock

import pytest

from eval.answer_metrics import (
    citation_precision,
    claim_recall,
    cohen_kappa,
    judge_answer,
    summarize,
)

SOURCES = [{"doc_id": "eaf004fe" + "0" * 56}, {"doc_id": "2b529e6a" + "1" * 56}, {"doc_id": "70d250f9" + "2" * 56}]


def test_citation_precision_counts_cited_sources_from_evidence_docs():
    answer = "RAG retrieves [1]. Vector DBs store embeddings [2]. Magnons [3]. Again [1]."
    assert citation_precision(answer, SOURCES, ["eaf004fe", "2b529e6a"]) == pytest.approx(2 / 3)


def test_citation_precision_ignores_dangling_and_is_none_without_citations():
    assert citation_precision("Nothing cited.", SOURCES, ["eaf004fe"]) is None
    assert citation_precision("Bad [9] good [1]", SOURCES, ["eaf004fe"]) == 1.0


def test_claim_recall():
    assert claim_recall([True, False, True, True]) == 0.75
    assert claim_recall([]) is None


def test_cohen_kappa_perfect_chance_and_degenerate():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohen_kappa([True, True], [True, True]) == 1.0  # identical constant labels


def test_summarize_reports_abstention_confusion():
    rows = [
        {"answerable": True, "abstained": False, "claim_recall": 1.0, "faithfulness": 0.9, "citation_precision": 1.0},
        {"answerable": True, "abstained": True, "claim_recall": None, "faithfulness": None, "citation_precision": None},
        {"answerable": False, "abstained": True},
        {"answerable": False, "abstained": False},
    ]
    s = summarize(rows)
    assert s["n_answerable"] == 2 and s["n_unanswerable"] == 2
    assert s["false_abstain_rate"] == 0.5
    assert s["correct_abstain_rate"] == 0.5
    assert s["mean_claim_recall"] == 1.0 and s["mean_faithfulness"] == 0.9


def _llm(content):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


def test_judge_answer_uses_strict_schema_and_returns_per_claim_verdicts():
    llm = _llm('{"declined": false, "claims_supported": [true, false]}')
    verdict = judge_answer(llm, "q", "answer", ["c1", "c2"])
    assert verdict == {"declined": False, "claims_supported": [True, False]}
    fmt = llm.invoke.call_args.kwargs["response_format"]
    assert fmt["json_schema"]["strict"] is True


def test_judge_answer_rejects_wrong_length():
    llm = _llm('{"declined": false, "claims_supported": [true]}')
    with pytest.raises(ValueError):
        judge_answer(llm, "q", "answer", ["c1", "c2"])


def test_calibrate_reads_human_labels(tmp_path, capsys):
    from eval.run_answers import calibrate

    p = tmp_path / "cal.csv"
    p.write_text("query,claim,judge_supported,human_supported,answer\n"
                 "q,c1,true,true,a\nq,c2,false,false,a\nq,c3,true,false,a\nq,c4,true,,a\n", encoding="utf-8")
    assert calibrate(p) == 0
    out = capsys.readouterr().out
    assert "labelled=3" in out and "agreement=0.667" in out


def test_resume_keeps_scored_rows_and_reruns_errors():
    from eval.run_answers import split_resume

    entries = [{"query": "a"}, {"query": "b"}, {"query": "c"}]
    previous = [{"query": "a", "claim_recall": 1.0}, {"query": "b", "error": "429"}]
    kept, todo = split_resume(entries, previous)
    assert kept == {"a": {"query": "a", "claim_recall": 1.0}}
    assert [e["query"] for e in todo] == ["b", "c"]
