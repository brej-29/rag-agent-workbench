import argparse
import json
import sys
from typing import Any, Dict

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the /chat endpoint with Tavily web search fallback. "
            "This expects TAVILY_API_KEY to be configured in the backend environment."
        )
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the running backend (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="dev",
        help="Target Pinecone namespace (default: dev).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Latest news about retrieval-augmented generation benchmarks",
        help="Query that is likely to require fresh web information.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.backend_url
    url = f"{base_url.rstrip('/')}/chat"

    payload: Dict[str, Any] = {
        "query": args.query,
        "namespace": args.namespace,
        "top_k": 3,
        "use_web_fallback": True,
        "min_score": 0.9,  # Intentionally high to encourage web fallback
        "max_web_results": 5,
        "chat_history": [],
    }

    print(f"[smoke_chat_web] POST {url} with payload:", file=sys.stderr)
    print(json.dumps(payload, indent=2), file=sys.stderr)

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"[smoke_chat_web] /chat request failed: {exc}", file=sys.stderr)
            if resp.content:
                print(resp.text, file=sys.stderr)
            return 1

        data = resp.json()
        print("[smoke_chat_web] /chat response:", file=sys.stderr)
        print(json.dumps(data, indent=2), file=sys.stderr)

    answer = data.get("answer", "")
    sources = data.get("sources", [])
    web_sources = [s for s in sources if s.get("source") == "web"]

    print("\nAnswer:\n", answer)
    print(f"\nTotal sources returned: {len(sources)}")
    print(f"Sources from web search: {len(web_sources)}")

    for src in web_sources[:5]:
        print(f"- {src.get('title')} ({src.get('url', '')})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())