"""Faithfulness judge uses strict JSON-schema structured output (Groq gpt-oss supports
`response_format={"type": "json_schema", "strict": true}`), so the verdict cannot be
malformed or mistyped; LLM_STRUCTURED_OUTPUT=False restores plain prompting for
providers without structured outputs."""
from unittest.mock import MagicMock, patch

import pytest

from app.services import faithfulness
from app.services.faithfulness import judge_faithfulness_with_usage


def _llm(content='{"grounded": true, "score": 0.9, "rationale": "ok"}'):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content, usage_metadata=None, response_metadata={})
    return llm


def _settings(on):
    s = MagicMock()
    s.LLM_STRUCTURED_OUTPUT = on
    return s


def test_judge_requests_strict_json_schema_by_default():
    llm = _llm()
    with patch.object(faithfulness, "get_settings", return_value=_settings(True)):
        verdict, _ = judge_faithfulness_with_usage("answer [1]", "context", llm)
    fmt = llm.invoke.call_args.kwargs["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"grounded", "score", "rationale"}
    assert schema["properties"]["grounded"]["type"] == "boolean"
    assert verdict.grounded is True and verdict.faithfulness_score == 0.9


def test_structured_output_can_be_disabled_for_other_providers():
    llm = _llm()
    with patch.object(faithfulness, "get_settings", return_value=_settings(False)):
        judge_faithfulness_with_usage("answer", "context", llm)
    assert "response_format" not in llm.invoke.call_args.kwargs


@pytest.mark.parametrize("raw,expected", [("false", False), ("False", False), ("true", True), ("no", None)])
def test_string_booleans_are_not_truthy(raw, expected):
    # bool("false") is True: a non-strict judge answering "false" was counted as grounded.
    verdict = faithfulness._parse_verdict(f'{{"grounded": "{raw}", "score": 0.1, "rationale": "x"}}')
    assert verdict.grounded is expected
