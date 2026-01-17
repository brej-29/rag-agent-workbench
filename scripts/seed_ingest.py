import argparse
import itertools
import sys
from typing import List

import httpx


TOPICS: List[str] = [
    "retrieval augmented generation",
    "vector databases",
    "semantic search",
    "information retrieval",
    "large language models",
    "transformer architectures",
    "question answering",
    "document similarity",
    "embedding models",
    "knowledge graphs",
    "few-shot learning",
    "self supervised learning",
    "contrastive learning",
    "neural search",
    "dense passage retrieval",
    "sparse retrieval",
    "multi modal retrieval",
    "open domain question answering",
    "context windows",
    "memory in llms",
    "hallucination mitigation",
    "prompt engineering",
    "evaluation of rag",
    "document chunking",
    "vector compression",
    "approximate nearest neighbors",
    "Pinecone vector database",
    "OpenAlex scholarly graph",
    "arXiv preprint search",
    "retrieval pipelines",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the RAG backend with documents from arXiv and OpenAlex."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the backend (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="dev",
        help="Target Pinecone namespace (default: dev)",
    )
    parser.add_argument(
        "--mailto",
        type=str,
        required=True,
        help="Contact email for OpenAlex API (required)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=20,
        help="Max documents per topic per source (capped at 20)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_docs = min(args.max_docs, 20)

    print(
        f"Seeding backend at {args.base_url} into namespace='{args.namespace}' "
        f"with up to {max_docs} docs per topic per source.",
        file=sys.stderr,
    )

    arxiv_url = f"{args.base_url.rstrip('/')}/ingest/arxiv"
    openalex_url = f"{args.base_url.rstrip('/')}/ingest/openalex"

    with httpx.Client(timeout=30.0) as client:
        for idx, topic in enumerate(TOPICS, start=1):
            print(f"[{idx}/{len(TOPICS)}] Topic: {topic}", file=sys.stderr)

            try:
                arxiv_resp = client.post(
                    arxiv_url,
                    json={
                        "query": topic,
                        "max_docs": max_docs,
                        "namespace": args.namespace,
                        "category": "papers",
                    },
                )
                arxiv_resp.raise_for_status()
                print(
                    f"  arXiv: {arxiv_resp.json()}",
                    file=sys.stderr,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  arXiv error: {exc}", file=sys.stderr)

            try:
                openalex_resp = client.post(
                    openalex_url,
                    json={
                        "query": topic,
                        "max_docs": max_docs,
                        "namespace": args.namespace,
                        "mailto": args.mailto,
                    },
                )
                openalex_resp.raise_for_status()
                print(
                    f"  OpenAlex: {openalex_resp.json()}",
                    file=sys.stderr,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  OpenAlex error: {exc}", file=sys.stderr)

    print("Seeding complete.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())