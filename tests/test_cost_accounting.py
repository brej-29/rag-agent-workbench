"""
CI-safe unit tests for T2.7 — token count extraction and cost accounting.

Coverage:
  (e) Token counts come from the provider response (not estimated).
  (f) Auxiliary call tokens (judge, crag_rewrite, contextualize) are included
      in the per-request total via _accumulate_token_usage.
  (g) Cost arithmetic from the pricing table (_GROQ_PRICING_USD_PER_1M).
  (h) Prometheus LLM_TOKENS_TOTAL counter increments with call_type label.

All tests make ZERO network calls.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.core.cost_accounting import estimate_cost_usd, extract_token_usage


# ---------------------------------------------------------------------------
# (e) Token counts from provider response — NOT estimated
# ---------------------------------------------------------------------------

class TestExtractTokenUsage:
    """extract_token_usage reads ACTUAL counts from the response object."""

    def test_usage_metadata_path_read(self):
        response = MagicMock()
        response.usage_metadata = {
            "input_tokens": 120,
            "output_tokens": 45,
            "total_tokens": 165,
        }
        usage = extract_token_usage(response)
        assert usage["prompt_tokens"] == 120
        assert usage["completion_tokens"] == 45
        assert usage["total_tokens"] == 165

    def test_response_metadata_path_read(self):
        response = MagicMock()
        response.usage_metadata = None  # Path 1 unavailable
        response.response_metadata = {
            "token_usage": {
                "prompt_tokens": 80,
                "completion_tokens": 30,
                "total_tokens": 110,
            }
        }
        usage = extract_token_usage(response)
        assert usage["prompt_tokens"] == 80
        assert usage["completion_tokens"] == 30
        assert usage["total_tokens"] == 110

    def test_mock_response_returns_zeros(self):
        # MagicMock auto-creates attributes; isinstance(MagicMock(), dict) is False,
        # so extract_token_usage returns zeros safely.  This is the guard that
        # prevents int(MagicMock()) failures in CI tests.
        response = MagicMock()
        # Do NOT set usage_metadata or response_metadata to real dicts;
        # let them auto-generate as MagicMock attributes.
        usage = extract_token_usage(response)
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_none_response_returns_zeros(self):
        usage = extract_token_usage(None)
        assert usage["total_tokens"] == 0

    def test_usage_metadata_takes_precedence_over_response_metadata(self):
        response = MagicMock()
        response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        response.response_metadata = {
            "token_usage": {
                "prompt_tokens": 999,
                "completion_tokens": 999,
                "total_tokens": 999,
            }
        }
        usage = extract_token_usage(response)
        # usage_metadata is Path 1 (preferred)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50

    def test_partial_usage_metadata_handled(self):
        response = MagicMock()
        # Missing output_tokens key — should default to 0
        response.usage_metadata = {"input_tokens": 50, "total_tokens": 50}
        usage = extract_token_usage(response)
        assert usage["prompt_tokens"] == 50
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 50


# ---------------------------------------------------------------------------
# (f) Auxiliary call tokens included in total — _accumulate_token_usage
# ---------------------------------------------------------------------------

class TestAccumulateTokenUsage:
    """_accumulate_token_usage from graph.py sums tokens across all call types."""

    def _get_accumulate(self):
        from app.services.chat.graph import _accumulate_token_usage  # noqa: PLC0415
        return _accumulate_token_usage

    def test_accumulates_generation_tokens(self):
        state: Dict[str, Any] = {"token_usage_by_call": {}}
        self._get_accumulate()(state, "generation", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        assert state["token_usage_by_call"]["generation"]["total_tokens"] == 150

    def test_accumulates_judge_tokens(self):
        state: Dict[str, Any] = {"token_usage_by_call": {}}
        self._get_accumulate()(state, "judge", {"prompt_tokens": 60, "completion_tokens": 20, "total_tokens": 80})
        assert state["token_usage_by_call"]["judge"]["total_tokens"] == 80

    def test_accumulates_crag_rewrite_tokens_across_iterations(self):
        state: Dict[str, Any] = {"token_usage_by_call": {}}
        accumulate = self._get_accumulate()
        # Two CRAG iterations
        accumulate(state, "crag_rewrite", {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50})
        accumulate(state, "crag_rewrite", {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50})
        assert state["token_usage_by_call"]["crag_rewrite"]["total_tokens"] == 100
        assert state["token_usage_by_call"]["crag_rewrite"]["prompt_tokens"] == 80

    def test_accumulates_contextualize_tokens(self):
        state: Dict[str, Any] = {"token_usage_by_call": {}}
        self._get_accumulate()(state, "contextualize", {"prompt_tokens": 70, "completion_tokens": 15, "total_tokens": 85})
        assert state["token_usage_by_call"]["contextualize"]["total_tokens"] == 85

    def test_zero_usage_dict_accumulates_zeros(self):
        state: Dict[str, Any] = {"token_usage_by_call": {}}
        self._get_accumulate()(state, "generation", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        assert state["token_usage_by_call"]["generation"]["total_tokens"] == 0

    def test_multiple_call_types_independent(self):
        state: Dict[str, Any] = {"token_usage_by_call": {}}
        accumulate = self._get_accumulate()
        accumulate(state, "generation", {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140})
        accumulate(state, "judge", {"prompt_tokens": 60, "completion_tokens": 20, "total_tokens": 80})
        assert state["token_usage_by_call"]["generation"]["total_tokens"] == 140
        assert state["token_usage_by_call"]["judge"]["total_tokens"] == 80

    def test_empty_token_usage_by_call_initialised(self):
        state: Dict[str, Any] = {}
        self._get_accumulate()(state, "generation", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        assert state["token_usage_by_call"]["generation"]["total_tokens"] == 15


# ---------------------------------------------------------------------------
# (g) Cost arithmetic from pricing table
# ---------------------------------------------------------------------------

class TestEstimateCostUsd:
    """estimate_cost_usd uses the as-of-date pricing table — output is an ESTIMATE."""

    def test_known_model_returns_nonzero(self):
        cost = estimate_cost_usd(1_000_000, 1_000_000, "llama-3.1-8b-instant")
        # At $0.05/M input and $0.08/M output: (1M × 0.05 + 1M × 0.08) / 1M = $0.13
        assert cost == pytest.approx(0.13, rel=0.01)

    def test_unknown_model_returns_none(self):
        cost = estimate_cost_usd(1000, 500, "gpt-99-ultra")
        assert cost is None

    def test_zero_tokens_returns_zero_cost(self):
        cost = estimate_cost_usd(0, 0, "llama-3.1-8b-instant")
        assert cost == pytest.approx(0.0)

    def test_cost_proportional_to_tokens(self):
        cost_half = estimate_cost_usd(500_000, 500_000, "llama-3.1-8b-instant")
        cost_full = estimate_cost_usd(1_000_000, 1_000_000, "llama-3.1-8b-instant")
        assert cost_full == pytest.approx(2 * cost_half, rel=0.001)

    def test_input_and_output_rates_differ(self):
        input_only = estimate_cost_usd(1_000_000, 0, "llama-3.1-8b-instant")
        output_only = estimate_cost_usd(0, 1_000_000, "llama-3.1-8b-instant")
        # $0.05 vs $0.08 — they must differ
        assert input_only != output_only
        assert input_only == pytest.approx(0.05, rel=0.01)
        assert output_only == pytest.approx(0.08, rel=0.01)

    def test_llama70b_has_higher_rate_than_8b(self):
        cost_8b = estimate_cost_usd(1_000_000, 1_000_000, "llama-3.1-8b-instant")
        cost_70b = estimate_cost_usd(1_000_000, 1_000_000, "llama-3.3-70b-versatile")
        assert cost_70b > cost_8b


# ---------------------------------------------------------------------------
# (h) Prometheus LLM_TOKENS_TOTAL counter increments with call_type label
# ---------------------------------------------------------------------------

class TestPrometheusTokenCounter:
    def test_record_token_usage_increments_counter(self):
        from app.core.prometheus_metrics import LLM_TOKENS_TOTAL, record_token_usage  # noqa: PLC0415

        by_call = {
            "generation": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }
        before = LLM_TOKENS_TOTAL.labels(call_type="generation")._value.get()
        record_token_usage(by_call)
        after = LLM_TOKENS_TOTAL.labels(call_type="generation")._value.get()
        assert after == before + 150

    def test_zero_tokens_not_incremented(self):
        from app.core.prometheus_metrics import LLM_TOKENS_TOTAL, record_token_usage  # noqa: PLC0415

        by_call = {
            "contextualize": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        before = LLM_TOKENS_TOTAL.labels(call_type="contextualize")._value.get()
        record_token_usage(by_call)
        after = LLM_TOKENS_TOTAL.labels(call_type="contextualize")._value.get()
        assert after == before  # zero → no increment

    def test_multiple_call_types_each_incremented(self):
        from app.core.prometheus_metrics import LLM_TOKENS_TOTAL, record_token_usage  # noqa: PLC0415

        by_call = {
            "generation": {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110},
            "judge": {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55},
        }
        before_gen = LLM_TOKENS_TOTAL.labels(call_type="generation")._value.get()
        before_judge = LLM_TOKENS_TOTAL.labels(call_type="judge")._value.get()
        record_token_usage(by_call)
        assert LLM_TOKENS_TOTAL.labels(call_type="generation")._value.get() == before_gen + 110
        assert LLM_TOKENS_TOTAL.labels(call_type="judge")._value.get() == before_judge + 55

    def test_empty_by_call_type_is_safe(self):
        from app.core.prometheus_metrics import record_token_usage  # noqa: PLC0415
        record_token_usage({})   # must not raise

    def test_none_by_call_type_is_safe(self):
        from app.core.prometheus_metrics import record_token_usage  # noqa: PLC0415
        record_token_usage(None)  # type: ignore[arg-type] — must not raise
