from app.core import auth


def test_open_when_no_key_configured(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    auth._get_configured_api_key.cache_clear()
    assert auth.api_key_matches(None) and auth.api_key_matches("anything")


def test_exact_match_only_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    auth._get_configured_api_key.cache_clear()
    try:
        assert auth.api_key_matches("s3cret")
        assert not auth.api_key_matches(None)
        assert not auth.api_key_matches("s3cre")
        assert not auth.api_key_matches("s3cret ")
    finally:
        auth._get_configured_api_key.cache_clear()
