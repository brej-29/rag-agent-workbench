"""Live MCP check over Streamable HTTP (spec §17.6): python scripts/smoke_mcp.py [url]
Needs the API running locally and API_KEY in the environment (never printed). Spends a few thousand Groq tokens;
no Pinecone writes."""
import os
import sys
from pathlib import Path

import anyio
import httpx2
from mcp.client.client import Client
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from tests.regs.test_standards import SYNTH  # noqa: E402  (synthetic copy: real clause ids, invented wording)

POLICY = ("Supplier security policy\n\nWe assess the security practices of every supplier before signing a contract "
          "and review critical suppliers once a year.\n\nContracts include security requirements and a right to audit.")
PROFILE = {"ai_system": True, "ai_roles": ["deployer"], "eu_established": True, "high_risk": "annex_iii"}


async def main(url: str) -> None:
    http = httpx2.AsyncClient(headers={"X-API-Key": os.environ["API_KEY"]}, timeout=180)
    async with Client(streamable_http_client(url, http_client=http), read_timeout_seconds=300) as c:
        print("tools:", sorted(t.name for t in (await c.list_tools()).tools))
        print("rag://facts:", len((await c.read_resource("rag://facts")).contents[0].text), "chars")
        cl = (await c.call_tool("checklist", {"profile": PROFILE})).structured_content["result"]
        print("checklist counts:", cl["counts"])
        xw = (await c.call_tool("crosswalk", {"provision": "nist-csf/GV.SC-01"})).structured_content["result"]
        print("crosswalk:", len(xw["items"]), "matches, first", [m["ref"] for m in xw["items"][:3]])
        seen = []

        async def on_progress(progress, total, message):
            seen.append(f"{progress:g}/{total:g} {message}")

        gap = (await c.call_tool("gap_check", {"policy_text": POLICY, "articles": ["nis2-ir/anx-1#5"]},
                                 progress_callback=on_progress)).structured_content["result"]
        print("gap_check progress:", seen, "summary:", gap["summary"])
        ans = (await c.call_tool("ask", {"question": "What does ISO/IEC 27001 A.5.19 require?",
                                         "standards": [{"text": SYNTH}]})).structured_content
        print("ask with your copy: abstained", ans["abstained"], "copy sources",
              [s["title"] for s in ans["sources"] if s["status"] == "licensed_copy"])


if __name__ == "__main__":
    anyio.run(main, sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8767/mcp")
