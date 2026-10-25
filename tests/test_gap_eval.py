from eval.run_gapcheck import score


def test_score_counts_agreement_false_coverage_and_verification():
    rows = [{"name": "a", "expected": {"x": "missing", "y": "covered", "z": "partial"}, "planted_missing": ["x"],
             "verdicts": {"x": "covered", "y": "covered", "z": "unverified"}}]
    rep = score(rows)
    assert rep["labelled"] == 3 and rep["exact"] == 1 and rep["accuracy"] == 0.333
    assert rep["false_covered"] == ["a:x"] and rep["gate_passed"] is False
    assert rep["verified_rate"] == 0.667 and rep["missing_caught"] == 0
