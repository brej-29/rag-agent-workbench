from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


def post_upload_text(
    base_url: str,
    api_key: Optional[str],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Call backend /documents/upload-text with the given payload.

    Sends X-API-Key when provided and raises for HTTP errors.
    """
    url = f"{base_url.rstrip('/')}/documents/upload-text"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()