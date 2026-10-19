# Roadmap: from "revived" to standout

Status 2026-10-19: revival done on branch `revive/2026-10`, not deployed. **Wave 1 items 1–5 are built** (results in RESURRECTION_REPORT.md §10); item 6 and everything after still need your go-ahead.

**What already stands out:** eval-first retrieval tuning (hand-labelled golden set, reranker measured and rejected) and structural guardrails (bounded CRAG, deterministic abstention, embedder-independent citation checks, injection hardening).

**The gap:** the eval covers retrieval only; nothing measures answer quality, abstention accuracy or adversarial robustness. The UI is a plain chat box, and the system cannot be used from other agents.

**The theme that ties it together:** *"a RAG system you can trust because every claim is measured"*. Every feature ships with a metric, and the UI shows the evidence.

Value and effort use a 1–5 scale. **S** is under a day, **M** is 1–3 days, **L** is a week or more. All items fit free tiers except where marked.

## Wave 1: deepen the differentiator (about 1 week)

| # | Item | Why it matters | Effort | Value |
|---|---|---|---|---|
| 1 | **DONE**. **Answer-quality and abstention eval**: extend `golden.jsonl` with reference claims plus an *unanswerable* subset. Measure claim-level faithfulness, citation precision/recall, answer relevance, correct-abstain rate and false-abstain rate. Calibrate the LLM judge against your own labels (agreement or κ) | Closes the open question README.md leaves at top_k ("tiebreaker: an answer-quality eval that doesn't yet exist"). Turns "faithfulness verification" from a claim into a number | M | 5 |
| 2 | **DONE**. **Adversarial eval suite**: direct injection, *indirect* injection planted inside an ingested document, CRAG query laundering (seen live on 2026-10-19: the rewriter turned an injection into "What is the system prompt used by ChatGPT?" and answered), system-prompt exfiltration, off-topic. Deterministic CI cases with mocked LLM plus a weekly live run | Makes "injection hardening" measurable; few portfolio RAGs have this | S–M | 5 |
| 3 | **DONE** (deterministic: refusal, answer-shaped and new-facts rewrites rejected; instruction-override queries skip the rewriter; no embedding drift check, because synonym rewrites are the rewriter's job). **CRAG rewrite guard**: reject rewrites that are refusals, answers, or semantically far from the original (embedding cosine below a threshold); fall back to the original query | Fixes the laundering finding; keeps the bounded-loop invariant | S | 4 |
| 4 | **DONE** (judge only; CRAG grading is score-based, not an LLM). **Strict structured outputs** for the judge and grader (Groq `json_schema` `strict: true` is supported on gpt-oss; not with streaming, which these calls don't need). Replaces regex JSON extraction; measure parse failures before and after | Removes a silent-degradation path | S | 3 |
| 5 | **DONE**: read-only, so no `ingest_url` (agent-controlled server-side URL fetch is an SSRF risk; ingestion stays behind the REST API). **MCP server** for the workbench: tools `search_kb`, `ask` (grounded answer + citations + abstention flag), `ingest_url`; resources for namespaces, documents and latest eval report; served over Streamable HTTP inside the FastAPI app behind the same API key, plus stdio for local use. Spec 2026-07-28 | Any MCP client (Claude Desktop/Code, IDEs) can use your RAG *with its guardrails intact*. Strong 2026 signal | S–M | 5 |
| 6 | **Model matrix**: run items 1 and 2 across `gpt-oss-20b`, `gpt-oss-120b` and `qwen3.8-27b`; plot quality against cost and latency; choose the default from data | Same eval-first story, applied to the model choice that broke the project | S (after 1) | 4 |

## Wave 2: push Streamlit to its limit (about 1–1.5 weeks)

Streamlit 1.66 (released 2026-10-14) now has custom components v2, custom light/dark themes with fonts, `st.navigation(position="top")`, side-drawer `st.dialog`, `st.mermaid_chart`, `st.skeleton`, the `:shimmer[]` directive, `st.feedback`, chat input with files, audio and paste, async app code, and parallel or event-scoped fragments.

| # | Item | Effort | Value |
|---|---|---|---|
| 7 | **App shell and design system**: top-nav multipage app (Chat · Evidence · Evaluation · Documents · About); custom light and dark themes in `config.toml` (fonts, palette, radius, chart colors); no CSS hacks | M | 4 |
| 8 | **Live pipeline trace in chat**: the backend emits node-progress SSE events (`normalize → retrieve → CRAG → gate → generate → verify`) and the UI renders them as a `st.status` step list with `:shimmer[]` while running. Tokens stream with a cursor | M | 5 |
| 9 | **Citation chips and evidence drawer**: `[1]` chips open a right-side `st.dialog` showing the chunk, source, cosine score against the floor, and which claims it supports. Unsupported claims highlighted | M | 5 |
| 10 | **Evidence page**: retrieval table (score bars plus the cosine-floor line), claim-by-claim faithfulness, and a latency/token/cost waterfall per request | M | 4 |
| 11 | **Evaluation page**: run history, metric trends, per-query drill-down, A/B diff (reranker), k-sweep chart, all from `eval/reports/*.json` | M | 5 |
| 12 | **Graph view**: `st.mermaid_chart` of the 8-node graph with the path the request took highlighted | S | 3 |
| 13 | **Feedback loop**: `st.feedback` on every answer; thumbs-down items queue as golden-set candidates (labelled by you, never auto-labelled, per the anti-circularity rule) | S | 4 |
| 14 | **Cold-start UX**: a background fragment pings `/health`; while the Space wakes, show `st.skeleton` and "waking backend… 6 s" instead of an error | S | 4 |
| 15 | Chat input attachments: drop or paste a PDF into the chat to ingest it into a session namespace, then ask about it | M | 3 |

## Wave 3: retrieval, context and harness engineering

| # | Item | Effort | Value |
|---|---|---|---|
| 16 | **Hybrid retrieval**: Pinecone dense plus full-text index, fused with RRF (Pinecone's current recommendation; full-text indexes are listed on the Starter plan). A/B it with the harness and publish the result either way, as was done for the reranker | M | 4 |
| 17 | **Context engineering**: token-budgeted context packing, near-duplicate removal, ordering against "lost in the middle", contextual chunk headers (doc title and section prepended at index time). Each change gated by items 1 and 16 | M | 4 |
| 18 | **Record and replay harness**: cassette-recorded Groq, Pinecone and Tavily responses so full-graph tests replay real outputs offline; golden trace snapshots. Model swaps then show their impact in CI | M | 4 |
| 19 | **Conversation memory**: LangGraph checkpointer with `thread_id`, summary buffer, and time-travel debugging of a conversation | M | 3 |
| 20 | **Server-side document conversion**: move docling to the backend (the HF Space has 16 GB; Streamlit Cloud caps at 2.7 GB) so Office and complex PDFs work in the hosted UI. Reverses the docling trade-off made in the revival | M | 3 |
| 21 | Query planning: decompose multi-hop questions; route KB / web / both; measured on a multi-hop golden subset | M–L | 3 |
| 22 | MCP *client* side: allow vetted external MCP tools as extra evidence sources under the same faithfulness gate | M–L | 3 |

## Ops, security and observability (pick as needed)

| # | Item | Effort |
|---|---|---|
| 23 | Startup degrade: report `degraded` on `/health` instead of crash-looping when Pinecone is unreachable (keep failing fast on *misconfiguration*) | S |
| 24 | Keep-warm: a scheduled ping to stay inside the 48 h sleep window. Whether HF's terms allow this is unverified; needs your approval | S |
| 25 | Auth: `st.login` (OIDC) for the UI and per-user keys or rate limits on the API; public demo mode with tight limits | M |
| 26 | Security pass: PII redaction in logs, upload size and type limits, SBOM, image scanning, CSP on the UI | S–M |
| 27 | OpenTelemetry traces next to LangSmith; Prometheus dashboard for `rag_phase_duration_seconds` | M |
| 28 | Regenerate the stale eval corpus manifest (8 doc_ids listed versus 23 live) | S |

## Suggested order

Items **3 → 4 → 1 → 2 → 5** first: small fixes, then the two evals that every later item is measured against, then MCP. Then **14 → 7 → 8 → 9 → 11** for the UI. Everything else in Wave 3 is gated on the eval from Wave 1.
