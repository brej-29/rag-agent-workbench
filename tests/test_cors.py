"""
CI-safe unit tests for CORS configuration (security.py).

Covers:
  - _get_allowed_origins: env-driven resolution, fallback, whitespace stripping
  - configure_security: allow_credentials is always False regardless of origins
  - Prod-like wildcard warning: fires in prod, silent in dev or with explicit origins

Zero network calls.  All env-var and prod-detection side-effects are patched.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.core.security import _get_allowed_origins, configure_security  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_app() -> MagicMock:
    return MagicMock(name="fastapi_app")


def _add_middleware_kwargs(app: MagicMock) -> dict:
    """Extract keyword arguments passed to app.add_middleware(...)."""
    return app.add_middleware.call_args[1]


# ---------------------------------------------------------------------------
# 1. _get_allowed_origins — pure resolution logic
# ---------------------------------------------------------------------------

class TestGetAllowedOrigins:
    def test_returns_wildcard_when_env_unset(self):
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": ""}, clear=False):
            origins = _get_allowed_origins()
        assert origins == ["*"]

    def test_returns_list_when_env_set(self):
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "https://a.com,https://b.com"}):
            origins = _get_allowed_origins()
        assert origins == ["https://a.com", "https://b.com"]

    def test_strips_whitespace_from_entries(self):
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": " https://a.com , https://b.com "}):
            origins = _get_allowed_origins()
        assert origins == ["https://a.com", "https://b.com"]

    def test_blank_env_falls_back_to_wildcard(self):
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "  ,  , "}):
            origins = _get_allowed_origins()
        assert origins == ["*"]

    def test_single_origin_returns_single_element_list(self):
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "https://only.example.com"}):
            origins = _get_allowed_origins()
        assert origins == ["https://only.example.com"]


# ---------------------------------------------------------------------------
# 2. configure_security — allow_credentials must always be False
#
# The WHATWG Fetch Standard forbids wildcard origins + allow_credentials=True;
# browsers reject that combination.  This API uses bearer-token (X-API-Key)
# auth, so credentials mode is never needed.
# ---------------------------------------------------------------------------

class TestCorsAlwaysFalseCredentials:
    def test_credentials_false_when_wildcard_origins(self):
        app = _mock_app()
        with patch("app.core.security._get_allowed_origins", return_value=["*"]):
            with patch("app.core.security._is_production_like", return_value=False):
                configure_security(app)
        kw = _add_middleware_kwargs(app)
        assert kw["allow_origins"] == ["*"]
        assert kw["allow_credentials"] is False

    def test_credentials_false_when_explicit_origins(self):
        app = _mock_app()
        with patch("app.core.security._get_allowed_origins",
                   return_value=["https://a.com", "https://b.com"]):
            configure_security(app)
        kw = _add_middleware_kwargs(app)
        assert kw["allow_origins"] == ["https://a.com", "https://b.com"]
        assert kw["allow_credentials"] is False

    def test_wildcard_plus_credentials_false_is_never_emitted(self):
        """Guard: the spec-invalid wildcard+credentials pair must never appear."""
        app = _mock_app()
        with patch("app.core.security._get_allowed_origins", return_value=["*"]):
            with patch("app.core.security._is_production_like", return_value=True):
                configure_security(app)
        kw = _add_middleware_kwargs(app)
        # If wildcard is present, credentials must be False (spec-valid)
        if "*" in kw["allow_origins"]:
            assert kw["allow_credentials"] is False


# ---------------------------------------------------------------------------
# 3. Prod-like wildcard warning
# ---------------------------------------------------------------------------

class TestProdWildcardWarning:
    def test_warning_fires_when_wildcard_in_prod(self, caplog):
        app = _mock_app()
        with patch("app.core.security._get_allowed_origins", return_value=["*"]):
            with patch("app.core.security._is_production_like", return_value=True):
                with caplog.at_level(logging.WARNING, logger="app.core.security"):
                    configure_security(app)
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_texts, "Expected at least one WARNING but got none"
        combined = " ".join(warning_texts)
        assert "wildcard" in combined.lower()
        assert "ALLOWED_ORIGINS" in combined

    def test_no_warning_when_explicit_origins_in_prod(self, caplog):
        app = _mock_app()
        explicit = ["https://prod.example.com"]
        with patch("app.core.security._get_allowed_origins", return_value=explicit):
            with patch("app.core.security._is_production_like", return_value=True):
                with caplog.at_level(logging.WARNING, logger="app.core.security"):
                    configure_security(app)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings

    def test_no_warning_when_wildcard_in_dev(self, caplog):
        app = _mock_app()
        with patch("app.core.security._get_allowed_origins", return_value=["*"]):
            with patch("app.core.security._is_production_like", return_value=False):
                with caplog.at_level(logging.WARNING, logger="app.core.security"):
                    configure_security(app)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings

    def test_warning_message_advises_allowed_origins_env_var(self, caplog):
        """Warning text must name the env var the operator should set."""
        app = _mock_app()
        with patch("app.core.security._get_allowed_origins", return_value=["*"]):
            with patch("app.core.security._is_production_like", return_value=True):
                with caplog.at_level(logging.WARNING, logger="app.core.security"):
                    configure_security(app)
        combined = " ".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
        assert "ALLOWED_ORIGINS" in combined
