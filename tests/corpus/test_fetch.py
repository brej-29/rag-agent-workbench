import hashlib

import httpx
import pytest

from corpus.fetch import fetch
from corpus.sources import SOURCES, Source

SRC = Source("t", "t", "T", "Test", "Test act", "https://x.invalid/doc", "eu-xhtml", "lic", "regulation")


def _client(body: bytes, seen: list):
    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=body)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_sends_negotiation_headers_and_caches(tmp_path):
    seen = []
    entry = fetch(SRC, tmp_path, _client(b"<html/>", seen))
    assert seen[0].headers["accept"] == "application/xhtml+xml" and seen[0].headers["accept-language"] == "eng"
    assert entry["sha256"] == hashlib.sha256(b"<html/>").hexdigest() and entry["bytes"] == 7
    fetch(SRC, tmp_path, _client(b"other", seen))
    assert len(seen) == 1  # cached


def test_empty_body_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="empty"):
        fetch(SRC, tmp_path, _client(b"", []))


def test_sources_are_official_and_licensed():
    for s in SOURCES:
        assert s.url.startswith(("http://publications.europa.eu/resource/celex/",
                                 "https://raw.githubusercontent.com/usnistgov/oscal-content/",
                                 "https://airc.nist.gov/docs/"))
        assert s.licence
    assert {s.prefix for s in SOURCES} == {"eu-ai-act", "gdpr", "nis2", "nis2-ir", "dora", "cra", "nist-800-53", "nist-csf",
                                           "nist-ai-rmf"}
