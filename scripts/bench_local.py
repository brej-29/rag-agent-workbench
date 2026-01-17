import argparse
import asyncio
import statistics
import time
from typing import Any, Dict, List, Tuple

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple asyncio load tester for the /chat endpoint."
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the backend (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="dev",
        help="Pinecone namespace to use for queries (default: dev)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=50,
        help="Total number of /chat requests to issue (default: 50)",
    )
    parser.add_argument(
        "--include-search",
        action="store_true",
        help="Also benchmark /search with the same concurrency and request count.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key to send as X-API-Key header.",
    )
    return parser.parse_args()


async def _run_one_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    json_body: Dict[str, Any],
    headers: Dict[str, str],
    semaphore: asyncio.Semaphore,
) -> Tuple[float, bool]:
    start = time.perf_counter()
    error = False
    async with semaphore:
        try:
            resp = await client.request(method, url, json=json_body, headers=headers)
            if resp.status_code >= 400:
                error = True
        except Exception:
            error = True
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed, error


async def _run_load_test(
    base_url: str,
    namespace: str,
    concurrency: int,
    total_requests: int,
    api_key: str | None,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat"
    payload: Dict[str, Any] = {
        "query": "Briefly explain retrieval-augmented generation.",
        "namespace": namespace,
        "top_k": 5,
        "use_web_fallback": True,
    }

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    semaphore = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    errors = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            _run_one_request(client, "POST", url, payload, headers, semaphore)
            for _ in range(total_requests)
        ]
        for coro in asyncio.as_completed(tasks):
            elapsed_ms, is_error = await coro
            latencies.append(elapsed_ms)
            if is_error:
                errors += 1

    return {
        "latencies_ms": latencies,
        "errors": errors,
        "total": total_requests,
    }


async def _run_search_test(
    base_url: str,
    namespace: str,
    concurrency: int,
    total_requests: int,
    api_key: str | None,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/search"
    payload: Dict[str, Any] = {
        "query": "retrieval-augmented generation",
        "top_k": 5,
        "namespace": namespace,
    }

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    semaphore = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    errors = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            _run_one_request(client, "POST", url, payload, headers, semaphore)
            for _ in range(total_requests)
        ]
        for coro in asyncio.as_completed(tasks):
            elapsed_ms, is_error = await coro
            latencies.append(elapsed_ms)
            if is_error:
                errors += 1

    return {
        "latencies_ms": latencies,
        "errors": errors,
        "total": total_requests,
    }


def _summarise(result: Dict[str, Any], label: str) -> None:
    latencies = result["latencies_ms"]
    errors = result["errors"]
    total = result["total"]
    successes = total - errors
    error_rate = (errors / total * 100.0) if total else 0.0

    if latencies:
        values = sorted(latencies)
        avg = sum(values) / len(values)
        p50 = statistics.median(values)
        # Simple index-based p95 that works for small samples.
        idx95 = max(0, int(round(0.95 * (len(values) - 1))))
        p95 = values[idx95]
    else:
        avg = p50 = p95 = 0.0

    print(f"=== {label} ===")
    print(f"Total requests: {total}")
    print(f"Successful:     {successes}")
    print(f"Errors:         {errors} ({error_rate:.1f}%)")
    print(f"Average latency: {avg:.2f} ms")
    print(f"p50 latency:     {p50:.2f} ms")
    print(f"p95 latency:     {p95:.2f} ms")
    print()


async def main_async() -> None:
    args = parse_args()
    print(
        f"Running /chat benchmark against {args.backend_url} "
        f"namespace='{args.namespace}' concurrency={args.concurrency} "
        f"requests={args.requests}"
    )
    chat_result = await _run_load_test(
        base_url=args.backend_url,
        namespace=args.namespace,
        concurrency=args.concurrency,
        total_requests=args.requests,
        api_key=args.api_key,
    )
    _summarise(chat_result, "/chat")

    if args.include_search:
        print(
            f"Running /search benchmark against {args.backend_url} "
            f"namespace='{args.namespace}' concurrency={args.concurrency} "
            f"requests={args.requests}"
        )
        search_result = await _run_search_test(
            base_url=args.backend_url,
            namespace=args.namespace,
            concurrency=args.concurrency,
            total_requests=args.requests,
            api_key=args.api_key,
        )
        _summarise(search_result, "/search")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()