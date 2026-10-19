#!/usr/bin/env python3
"""Fail loudly when a configured chat model is retired, deprecated or missing.

Checks every model in GROQ_MODEL + LLM_FALLBACK_MODELS:
  1. listed by GET {GROQ_BASE_URL}/models
  2. answers a 1-message chat completion (HTTP 200)
  3. (Groq only) not named in a heading on console.groq.com/docs/deprecations,
     which catches models announced for shutdown before they disappear.

Exit 0 = all good, 1 = at least one problem.  Needs GROQ_API_KEY (env or .env).

Usage:  python scripts/check_models.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

DEPRECATIONS_URL = "https://console.groq.com/docs/deprecations"


def deprecated_models(html: str) -> dict[str, str]:
    """Map model id -> heading text (e.g. 'August 16, 2026: ...') for each <h3> on the page.

    Replacements live in tables, not headings, so only retired models match.
    """
    found: dict[str, str] = {}
    for h3 in re.findall(r"<h3\b[^>]*>(.*?)</h3>", html, flags=re.S):
        heading = re.sub(r"<[^>]+>", "", h3).strip()
        for model in re.findall(r"<code\b[^>]*>(.*?)</code>", h3, flags=re.S):
            found[model.strip()] = heading
    return found


def main() -> int:
    from app.core.config import get_settings  # noqa: PLC0415
    from app.services.llm.groq_llm import configured_models  # noqa: PLC0415

    s = get_settings()
    if not s.GROQ_API_KEY:
        print("FAIL: GROQ_API_KEY is not set")
        return 1
    models = configured_models()
    headers = {"Authorization": f"Bearer {s.GROQ_API_KEY}"}
    base = s.GROQ_BASE_URL.rstrip("/")
    problems: list[str] = []

    with httpx.Client(timeout=30, headers=headers) as client:
        r = client.get(f"{base}/models")
        r.raise_for_status()
        listed = {m["id"] for m in r.json()["data"]}

        deprecated: dict[str, str] = {}
        if "groq.com" in base:
            page = httpx.get(DEPRECATIONS_URL, timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
            deprecated = deprecated_models(page.text) if page.status_code == 200 else {}
            if not deprecated:
                problems.append(f"could not parse {DEPRECATIONS_URL} (HTTP {page.status_code}); check it by hand")

        for model in models:
            if model not in listed:
                problems.append(f"{model}: not listed by {base}/models")
                continue
            body = {"model": model, "messages": [{"role": "user", "content": "Reply: ok"}], "max_tokens": 64}
            if s.LLM_REASONING_EFFORT:
                body["reasoning_effort"] = s.LLM_REASONING_EFFORT
            c = client.post(f"{base}/chat/completions", json=body)
            if c.status_code != 200:
                problems.append(f"{model}: chat completion HTTP {c.status_code} {c.text[:160]}")
            if model in deprecated:
                problems.append(f"{model}: on Groq deprecations page -> '{deprecated[model]}'")
            else:
                print(f"OK   {model}")

    for p in problems:
        print(f"FAIL {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
