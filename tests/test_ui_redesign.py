from frontend.services.eval_data import trust_proof
from frontend.services.presenters import RULEBOOK_LINES, dot_grid_html, rulebook_stack_html


def test_trust_proof_uses_published_values_without_changing_them():
    results = {
        "regs_retrieval": {"k": 6, "n": 50, "aggregate": {"recall": 0.8713}},
        "regs_abstain": {"correct_abstain": 13, "unanswerable": 14, "false_abstain": 0, "answerable": 50},
    }
    proof = trust_proof(results)
    assert proof[0]["sentence"] == ("Across 50 test questions, the right article was in its top 6 results 87 times in 100.")
    assert proof[1]["sentence"] == ("It declined 13 of 14 questions about rules it does not cover, "
                                 "and wrongly declined 0 of 50 it does cover.")


def test_home_html_builders_have_the_promised_counts():
    assert rulebook_stack_html().count('class="cw-book"') == 8
    grid = dot_grid_html(87, 100)
    assert grid.count('class="cw-dot is-filled"') == 87
    assert grid.count('class="cw-dot"') == 13
    assert len(RULEBOOK_LINES) == 6
