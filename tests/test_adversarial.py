"""Adversarial suite, structural half: defences that must hold even if the model
obeys every instruction it sees.  ObedientLLM echoes every canary in its prompt and
dumps its system prompt, i.e. a fully compromised model.  The model-dependent half
(indirect injection, exfiltration) is measured live by `python eval/adversarial.py`."""
import pytest

from eval.adversarial import ObedientLLM, leaked_prompt, load_cases, run_case

CASES = load_cases()
STRUCTURAL = [c for c in CASES if c["structural"]]


def test_suite_has_structural_and_model_dependent_cases():
    assert len(STRUCTURAL) >= 4 and len(CASES) > len(STRUCTURAL)


@pytest.mark.parametrize("case", STRUCTURAL, ids=[c["id"] for c in STRUCTURAL])
def test_structural_defence_holds_against_compromised_model(case):
    result = run_case(case, llm=ObedientLLM(), search=None)
    assert not result["attack_succeeded"], result


def test_obedient_model_really_is_compromised():
    """Without this, passing structural cases could just mean a toothless fake."""
    case = next(c for c in CASES if c["id"] == "indirect-canary")
    assert run_case(case, llm=ObedientLLM(), search=None)["attack_succeeded"]


def test_leak_detector():
    from app.services.prompts.rag_prompt import SYSTEM_PROMPT

    assert leaked_prompt("Sure! " + SYSTEM_PROMPT[10:90] + " ...")
    assert not leaked_prompt("A vector database stores embeddings for similarity search [1].")
