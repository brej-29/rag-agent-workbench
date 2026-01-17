import argparse
import json
import sys

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the /ingest/arxiv endpoint of the RAG backend."
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
        default="retrieval augmented generation",
        help="arXiv search query (default: 'retrieval augmented generation').",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=2,
        help="Maximum number of documents to ingest (default: 2).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = f"{args.backend_url.rstrip('/')}/ingest/arxiv"

    payload = {
        "query": args.query,
        "max_docs": args.max_docs,
        "namespace": args.namespace,
        "category": "smoke-test",
    }

    print(f"POST {url} with payload:", file=sys.stderr)
    print(json.dumps(payload, indent=2), file=sys.stderr)

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"Request failed: {exc}", file=sys.stderr)
            if resp.content:
                print("Response body:", file=sys.stderr)
                print(resp.text, file=sys.stderr)
            return 1

        print("Response JSON:", file=sys.stderr)
        print(json.dumps(resp.json(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())