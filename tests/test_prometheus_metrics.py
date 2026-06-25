"""
Tests for Prometheus instrumentation (T2.6).

All tests are CI-safe: zero network calls, no Pinecone/Groq/Tavily credentials.

Tests cover:
  (a) /metrics/prometheus returns HTTP 200 with the expected metric names in
      the Prometheus text exposition format.
  (b) Recording pipeline timings via record_chat_timings_prometheus() is
      reflected in the Histogram sample count.
  (c) The legacy GET /metrics JSON endpoint is unbroken (structure check on
      get_metrics_snapshot()).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.core.metrics import get_metrics_snapshot
from app.core.prometheus_metrics import (
    RAG_PHASE_DURATION,
    record_chat_timings_prometheus,
    setup_prometheus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_app() -> FastAPI:
    """Minimal FastAPI app with only the Prometheus endpoint wired."""
    app = FastAPI()
    setup_prometheus(app)
    return app


def _sample_timings(total_ms: float = 800.0) -> dict:
    return {
        "retrieve_ms": 320.0,
        "web_ms": 0.0,
        "generate_ms": 450.0,
        "rerank_ms": 0.0,
        "faithfulness_ms": 0.0,
        "total_ms": total_ms,
    }


# ---------------------------------------------------------------------------
# (a) Prometheus endpoint returns 200 and exposes expected metric names
# ---------------------------------------------------------------------------

class TestPrometheusEndpoint:
    def test_endpoint_returns_200(self):
        """GET /metrics/prometheus returns HTTP 200."""
        client = TestClient(_make_test_app())
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200

    def test_content_type_is_prometheus_text_format(self):
        """Response Content-Type is the Prometheus plain-text exposition format."""
        client = TestClient(_make_test_app())
        resp = client.get("/metrics/prometheus")
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_rag_phase_histogram_appears_in_output(self):
        """rag_phase_duration_seconds is present even before any observations."""
        client = TestClient(_make_test_app())
        resp = client.get("/metrics/prometheus")
        assert "rag_phase_duration_seconds" in resp.text

    def test_http_instrumentation_metric_appears_after_request(self):
        """After making a request, an HTTP-level metric is present in the exposition."""
        app = _make_test_app()
        client = TestClient(app)
        # Make a dummy request so the instrumentator has something to report.
        client.get("/nonexistent")
        resp = client.get("/metrics/prometheus")
        body = resp.text
        # The Instrumentator exposes http_requests_total or http_request_duration —
        # at least one HTTP metric must be present.
        assert "http_request" in body


# ---------------------------------------------------------------------------
# (b) Observation reflected in the Histogram
# ---------------------------------------------------------------------------

class TestHistogramObservation:
    def test_record_timings_increments_total_count(self):
        """Observing total_ms > 0 increments the 'total' phase sample count."""
        before = REGISTRY.get_sample_value(
            "rag_phase_duration_seconds_count",
            {"phase": "total"},
        ) or 0.0

        record_chat_timings_prometheus(_sample_timings(total_ms=900.0))

        after = REGISTRY.get_sample_value(
            "rag_phase_duration_seconds_count",
            {"phase": "total"},
        )
        assert after == before + 1.0

    def test_record_timings_increments_retrieve_count(self):
        """Observing retrieve_ms > 0 increments the 'retrieve' phase sample count."""
        before = REGISTRY.get_sample_value(
            "rag_phase_duration_seconds_count",
            {"phase": "retrieve"},
        ) or 0.0

        record_chat_timings_prometheus(_sample_timings())

        after = REGISTRY.get_sample_value(
            "rag_phase_duration_seconds_count",
            {"phase": "retrieve"},
        )
        assert after == before + 1.0

    def test_zero_phases_not_observed(self):
        """Phases with value 0.0 are skipped — their count does not increase."""
        # faithfulness_ms = 0 when RAG_FAITHFULNESS_ENABLED is False (the default)
        before = REGISTRY.get_sample_value(
            "rag_phase_duration_seconds_count",
            {"phase": "faithfulness"},
        ) or 0.0

        record_chat_timings_prometheus(_sample_timings())  # faithfulness_ms=0.0

        after = REGISTRY.get_sample_value(
            "rag_phase_duration_seconds_count",
            {"phase": "faithfulness"},
        ) or 0.0
        assert after == before  # unchanged

    def test_observe_converts_ms_to_seconds(self):
        """500 ms observation should appear in the <= 0.5s bucket (not <= 0.25s)."""
        # Record baseline sum for "total" phase.
        before_sum = REGISTRY.get_sample_value(
            "rag_phase_duration_seconds_sum",
            {"phase": "total"},
        ) or 0.0

        record_chat_timings_prometheus({"total_ms": 500.0})

        after_sum = REGISTRY.get_sample_value(
            "rag_phase_duration_seconds_sum",
            {"phase": "total"},
        )
        # 500 ms == 0.5 s; sum should have increased by exactly 0.5
        assert abs((after_sum - before_sum) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# (c) Legacy JSON /metrics endpoint is preserved
# ---------------------------------------------------------------------------

class TestLegacyMetricsEndpoint:
    def test_get_metrics_snapshot_returns_expected_structure(self):
        """get_metrics_snapshot() still returns all keys the existing endpoint relies on."""
        snapshot = get_metrics_snapshot()
        assert "requests_by_path" in snapshot
        assert "errors_by_path" in snapshot
        assert "timings" in snapshot
        assert "p50_ms" in snapshot["timings"]
        assert "p95_ms" in snapshot["timings"]
        assert "average_ms" in snapshot["timings"]
        assert "cache" in snapshot
        assert "sample_count" in snapshot
        assert "samples" in snapshot

    def test_legacy_snapshot_timings_include_four_fields(self):
        """Legacy timings track retrieve_ms, web_ms, generate_ms, total_ms."""
        from app.core.metrics import _TIMING_FIELDS
        assert set(_TIMING_FIELDS) == {"retrieve_ms", "web_ms", "generate_ms", "total_ms"}
