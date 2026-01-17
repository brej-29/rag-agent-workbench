import argparse
import json
import sys
from typing import Any, Dict, List

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the /chat endpoint of the RAG backend. "
            "Optionally ingests a couple of Wikipedia pages first."
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
        "--skip-ingest",
        action="store_true",
        help="Skip the initial Wikipedia ingestion step.",
    )
    return parser.parse_args()


def ingest_wiki(client: httpx.Client, base_url: str, namespace: str) -> None:
    url = f"{base_url.rstrip('/')}/ingest/wiki"
    payload: Dict[str, Any] = {
        "titles": [
            "Retrieval-augmented generation",
            "Vector database",
        ],
        "namespace": namespace,
    }

    print(f"[smoke_chat] POST {url} with payload:", file=sys.stderr)
    print(json.dumps(payload, indent=2), file=sys.stderr)

    try:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"[smoke_chat] Wiki ingest failed: {exc}", file=sys.stderr)
        if resp.content:
            print(resp.text, file=sys.stderr)
        return

    print("[smoke_chat] Wiki ingest response:", file=sys.stderr)
    print(json.dumps(resp.json(), indent=2), file=sys.stderr)


def call_chat(client: httpx.Client, base_url: str, namespace: str) -> None:
    url = f"{base_url.rstrip('/')}/chat"
    payload: Dict[str, Any] = {
        "query": "What is retrieval-augmented generation?",
        "namespace": namespace,
        "top_k": 5,
        "use_web_fallback": True,
        "min_score": 0.25,
        "max_web_results": 5,
        "chat_history": [
            {
                "role": "user",
                "content": "You are helping me understand retrieval-augmented generation.",
            }
        ],
    }

    print(f"[smoke_chat] POST {url} with payload:", file=sys.stderr)
    print(json.dumps(payload, indent=2), file=sys.stderr)

    resp = client.post(url, json=payload)
    try:
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"[smoke_chat] /chat request failed: {exc}", file=sys.stderr)
        if resp.content:
            print(resp.text, file=sys.stderr)
        return

    data = resp.json()
    print("[smoke_chat] /chat response:", file=sys.stderr)
    print(json.dumps(data, indent=2), file=sys.stderr)

    answer = data.get("answer", "")
    sources: List[Dict[str, Any]] = data.get("sources", [])[:3]
    print("\nAnswer:\n", answer)
    print("\nFirst up to 3 sources:")
    for src in sources:
        print(
            f"- [{src.get('source')}] {src.get('title')} ({src.get('url', '')}) "
            f"score={src.get('score')}",
        )


def main() -> int:
    args = parse_args()
    base_url = args.backend_url

    with httpx.Client(timeout=60.0) as client:
        if not args.skip_ingest:
            ingest_wiki(client, base_url, args.namespace)
        call_chat(client, base_url, args.namespace)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())