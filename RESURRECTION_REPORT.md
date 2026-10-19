# Resurrection Report: RAG Agent Workbench

Audit date: 2026-10-19. Branch: `revive/2026-10` in both repos. Status: **Phase 4 (option F: A + C-lite) implemented locally and verified; not pushed or deployed.** Phase 4 results are in section 9; the next roadmap is in `docs/ROADMAP.md`.

Evidence tags: **VERIFIED** means read on an official page or reproduced by a command on 2026-10-19. **UNVERIFIED** means a secondary source, inference, or something that needs a live key.

## 0. What the two repos are

| Repo | Role | Remote | Last commit |
|---|---|---|---|
| `D:\Project\rag-agent-workbench` | **Monorepo and source of truth**: backend, Streamlit UI (`frontend/`), 343 tests, eval harness, CI | `github.com/brej-29/rag-agent-workbench` | 2026-06-26 |
| `D:\Project\rag-agent-workbench-api` | **Deploy copy for the HF Space** (Docker). Its `backend/` is identical to the monorepo's. There are no tests, and 35 `__pycache__/*.pyc` files are committed | `huggingface.co/spaces/BrejBala/rag-agent-workbench-api` | 2026-06-26 |

- **Backups:** `D:\Project\_backups\revive-2026-10-19\` holds both `--all` git bundles (verified) and `working-trees.tgz`. The archive includes your gitignored local `.env` files, so keep it private.
- **Secrets:** I scanned every blob in the full history of both repos for provider key patterns (`gsk_`, `pcsk_`, `tvly-`, `lsv2_`, `hf_`, `sk-`, `AKIA`) and for `*_KEY/TOKEN/SECRET=` assignments. **No real secrets were found.** The only matches are placeholders: `backend/.env.example` (`your-…`), `tests/integration/conftest.py:INTEGRATION_API_KEY` and `scripts/bench_mocked.py:_BENCH_API_KEY` (test fixtures). **Nothing needs rotating because of git.** Local gitignored files that do contain keys are `.env`, `backend/.env.docker` and `frontend/.streamlit/secrets.toml`. I did not read their values.
- **Variables the backend expects:** `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_HOST`, `PINECONE_NAMESPACE`, `PINECONE_TEXT_FIELD`, `GROQ_API_KEY`, `GROQ_BASE_URL`, `GROQ_MODEL`, `TAVILY_API_KEY`, `API_KEY`, `ENV`, `ALLOWED_ORIGINS`, `RATE_LIMIT_ENABLED`, `CACHE_ENABLED`, `LOG_LEVEL`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, `WIKIMEDIA_USER_AGENT`, and the `RAG_*` tunables. **Variables the UI expects:** `BACKEND_BASE_URL` and `API_KEY`.
- **Pipeline:** 8 LangGraph nodes (`graph.py:654-661`). The README says "7-node", which is a documentation error.

## 1. Status

| Component | State | Evidence |
|---|---|---|
| HF Space `BrejBala/rag-agent-workbench-api` | **Was SLEEPING; now RUNNING and healthy.** My read-only `GET /health` woke it: 9.8 s cold start, then `{"status":"ok"}`. It will sleep again after 48 h idle | HF API `runtime.stage=SLEEPING`, `gcTimeout=172800`, `hardware=cpu-basic`; then stage `RUNNING` (VERIFIED) |
| Space `/chat` generation | **BROKEN (VERIFIED).** Groq returns `404 model_not_found` for `llama-3.1-8b-instant` with both of your keys, and one chat through the UI against the live Space returned "LLM generation failed" | Key check + live UI request, 2026-10-19 |
| Streamlit UI `rag-agent-workbench.streamlit.app` | **Asleep** ("This app has gone to sleep due to inactivity"). I did not wake it | Browser fetch (VERIFIED) |
| GitHub Actions CI | **Green.** The last CI run was 2026-08-12, on Dependabot PRs. Three Dependabot PRs are still open (#24, #25, #27) | `gh run list` / `gh pr list` (VERIFIED) |
| Pinecone index | **Ready (VERIFIED).** `rag-agent-workbench`: serverless aws/us-east-1, 1024-dim cosine, integrated `llama-text-embed-v2`; 34 vectors in `eval`, 42 in `dev`. Not paused. Starter limits: 5 indexes, 2 GB, 1M RU and 2M WU a month, 5M embed tokens a month per model, 500 rerank requests a month | `describe_index` / `describe_index_stats` with your key; Pinecone pricing and limits pages |
| Groq | Key and model status need your key | See section 2 |
| Tavily | **OK (VERIFIED).** Researcher plan, 0 of 1,000 credits used before testing; 1 credit used by the revival check | `/usage` endpoint with your key; docs.tavily.com |

## 2. Checks against your list of known risks

| Claim | Finding | Source |
|---|---|---|
| Groq deprecations | **True.** `llama-3.1-8b-instant` shut down 08/16/26 (replacement `openai/gpt-oss-20b`). `llama-3.3-70b-versatile` shut down 08/16/26 (replacement `gpt-oss-120b`). `qwen3-32b` and `llama-4-scout` shut down 07/17/26. **Also new:** `qwen/qwen3.6-27b`, which Groq itself named as a replacement, shut down 09/14/26. The models page still lists the two Llamas under "Production" priced "Contact Sales", which suggests enterprise-only access. Current self-serve production models: `openai/gpt-oss-20b` ($0.075/$0.30 per 1M tokens) and `openai/gpt-oss-120b` ($0.15/$0.60) | console.groq.com/docs/deprecations, /docs/models (VERIFIED) |
| HF CPU Basic | **True:** 2 vCPU, 16 GB, 50 GB disk, free, sleeps after 48 h. **New:** *creating* a Gradio or Docker Space now requires PRO ($9/mo). Duplicating one follows the same rule. Your existing Space still runs, so **do not delete or recreate it** | huggingface.co/docs/hub/spaces-overview, /spaces-gpus, /pricing (VERIFIED) |
| Pinecone pause | Not supported by current official docs, which have no inactivity text. The index state can only be checked with your key | See section 1 |
| LangChain/LangGraph 1.x | Partly true. Today's PyPI versions: langchain-core **1.6.12** (not 1.4.x), langgraph **1.2.16**, langchain 1.4.6, langchain-classic 1.0.8. **Your code is already v1-clean:** it does not use `langgraph.prebuilt`, `create_react_agent`, legacy chains or `.text()`. The pinned versions are langchain-core 1.1.0 and langgraph 1.0.3. The one legacy import is `langchain_community.tools.tavily_search.TavilySearchResults`; the migration guide moves community re-exports to `langchain-classic` and says to use provider packages | PyPI JSON API; docs.langchain.com/oss/python/migrate/langchain-v1 (VERIFIED) |
| Render free | Spins down after 15 min idle, takes about **1 min** to spin up, 750 instance-hours a month, ephemeral disk. RAM and card requirement are not stated | render.com/docs/free (VERIFIED) |
| Fly.io | No free tier for new organizations. Card required after the trial (2 h of machine time or 7 days). shared-cpu-1x costs $2.19/mo (256 MB) or $3.69/mo (512 MB) | docs.fly.io/about/pricing (VERIFIED) |
| Railway | $5 one-time trial lasting 30 days, then a Free plan with **$1 of credit a month** | docs.railway.com/reference/pricing/free-trial (VERIFIED) |
| Koyeb | No free compute listed (free Postgres only). Pro plan is $29/mo | koyeb.com/pricing (VERIFIED) |
| Cloud Run | Request-based free tier: 180k vCPU-s, 360k GiB-s and 2M requests a month, priced at us-central1 rates, aggregated per billing account (so a billing account and card are needed) | cloud.google.com/run/pricing (VERIFIED) |
| Streamlit Community Cloud | Sleeps after **12 h** without traffic. Up to 2 cores and 2.7 GB RAM ("may change") | docs.streamlit.io (VERIFIED) |
| Embedding model | `llama-text-embed-v2` and `bge-reranker-v2-m3` are still listed, with no deprecation notice | docs.pinecone.io/models/overview (VERIFIED) |
| AWS | New-account free plan: up to $200 in credits, and the account closes after 6 months unless you upgrade. Lambda: 1M requests and 400k GB-s a month, first 6 MB of response streaming free. App Runner: $0.007/GB-h idle, $0.064/vCPU-h active (about $2.52/mo idle at 0.5 GB). Bedrock gpt-oss-20b: $0.0721/$0.309 per 1M tokens (only the Sydney price was visible). Knowledge Bases: $5/GB-month storage, $1 per 1k retrievals | aws.amazon.com/free, /lambda/pricing, /apprunner/pricing, /bedrock/pricing (VERIFIED, partial page reads) |

## 3. Phase 2 baseline (the regression gate)

| Check | Result |
|---|---|
| Backend lock install: `uv pip install --require-hashes`, Python 3.11.15, Windows | **OK** |
| `pytest tests/`, hermetic (dummy keys, tracing off) | **343 passed, 0 failed, 7.8 s.** This includes the invariant suites: `test_crag` 17, `test_retrieval_gating` 17, `test_faithfulness` 29, `test_prompt_hardening` 20 |
| `pytest tests/` with your local `.env` auto-loaded | 343 passed, but in **25 s with 32 warnings**: the LangChain tracer was active, so the run **may have sent LangSmith traces with your key** |
| Eval metric math, offline (`test_eval_metrics.py`) | 43 passed |
| Live retrieval eval (`make eval`) | **Not run: needs Pinecone keys.** Recorded baseline (`eval/reports/20260625T071005Z.md`, 29 queries, k=10): **Recall@10 0.9684 · MRR 0.9828 · nDCG@10 0.9255**. Reranker A/B: nDCG@3 −0.057, +435 ms |
| Tests that need live keys | None in `tests/`. These need keys: `eval/run.py`, `run_ab.py`, `run_sweep.py`, `setup_corpus.py`, `scripts/smoke_*.py`, `seed_ingest.py` |
| Backend local boot (dummy keys) | **Fails:** the startup hook calls Pinecone, gets a 401 and exits. A real key is needed to boot |
| UI lock install (`requirements.txt`, Python 3.13, Windows) | **Fails:** the lock pins `torch` and 22 `nvidia-*` CUDA wheels that only exist for Linux |
| UI with only `streamlit` and `httpx` at their pinned versions | Boots; `/_stcore/health` returns 200 |
| Docker build | Not run: Docker Desktop is installed but its engine was not running. The same Dockerfile runs on the live Space, and CI installs the same lock on Linux |
| `pip-audit` on the backend lock | **52 known vulnerabilities in 15 packages** (langchain-core, langgraph, langgraph-checkpoint, langgraph-sdk, langchain-openai, langchain-text-splitters, urllib3, python-dotenv, orjson, soupsieve, …). All have fixed versions |

## 4. Breakage ranked by severity

| # | Severity | Issue | Fix size |
|---|---|---|---|
| 1 | **Critical** | The default `GROQ_MODEL=llama-3.1-8b-instant` was shut down. Answer generation, the CRAG grader, query rewriting and the faithfulness judge all use it. The model is already read from config (`config.py:78`), so the fix is to change the Space secret or variable and re-validate. `openai/gpt-oss-*` are reasoning models, so the judge and grader output parsing must be re-tested | S (config) + S–M (re-validation) |
| 2 | High | 52 vulnerabilities in the backend lock; 3 Dependabot PRs unmerged | S–M (re-lock with hashes, run tests) |
| 3 | High | The UI lock bundles docling, torch and 22 CUDA wheels; it is Linux-only and not hash-pinned. **Correction:** `frontend/services/file_convert.py` *does* import docling and pypdf (optionally, with fallbacks); my first grep missed it | S |
| 4 | Medium | Tests are not hermetic: `config.py:10` calls `load_dotenv()` at import, which picks up a developer's real `.env` | S (make `tests/conftest.py` set safe env) |
| 5 | Medium | Deploys are manual copies into the Space repo, which drift and include committed `.pyc` files | S (CI job that pushes `backend/` and the Dockerfile to the Space) |
| 6 | Medium | Startup hard-fails if Pinecone is unreachable or the key is wrong, so the Space crash-loops instead of reporting "degraded" | S |
| 7 | Low | `cost_accounting.py` prices only deprecated models; `@app.on_event` is deprecated (FastAPI recommends lifespan); `TavilySearchResults` is legacy; the README says "7-node" and "LLaMA 3.1 8B" | S |
| 8 | Info | Free tiers sleep (HF 48 h, Streamlit 12 h). Creating a new Docker Space now costs $9/mo | n/a |

## 5. Dependency modernisation plan

1. Raise the pins in `backend/requirements.in` to the current majors: langchain-core 1.6.x, langgraph 1.2.x, langchain-openai 1.7.x, pinecone 10.x (a major-version jump, so test the `describe_index`, search and upsert paths), fastapi 0.148, uvicorn 0.54. Replace `TavilySearchResults` with direct `tavily-python`, which is already pinned, and drop `langchain-community` if nothing else needs it.
2. Re-lock: `uv pip compile --generate-hashes --python-version 3.11 backend/requirements.in -o backend/requirements.txt`. Keep Python 3.11 in Docker and CI, the same minor version, so the lock is not broken.
3. Split the UI dependencies: `requirements.in` becomes `streamlit`, `httpx`, hashed. Move `docling`, `pypdf` and the rest to `requirements-ingest.in`.
4. Regenerate `requirements-dev.txt` with hashes against the new backend lock. Add `pip-audit` to CI.
5. Gate: 343/343 tests pass, the invariant suites pass, and live eval stays within noise of 0.9684 / 0.9828 / 0.9255.

## 6. Hosting analysis (monthly cost at portfolio traffic, under 1k requests a month)

| Host | Backend fit | Sleep / cold start | Card? | Est. $/mo |
|---|---|---|---|---|
| **HF Space (existing, CPU Basic)** | Already deployed. 2 vCPU / 16 GB | Sleeps after 48 h; **9.8 s wake measured** | No (an existing Space needs no PRO) | **$0** |
| HF Space (new Docker Space) | Same | Same | Yes, PRO | $9 |
| Render free | OK | 15 min idle; about 1 min spin-up | Not stated | $0 |
| Cloud Run (request-based) | Good, scales to zero | A few seconds plus the Pinecone startup call (UNVERIFIED, not measured) | Yes, billing account | ~$0 within quota |
| Fly.io | OK | Configurable | Yes | $2.19–3.69 |
| Railway | $1 of credit a month cannot run an always-on API | n/a | Needed to upgrade | ≥$5 (UNVERIFIED) |
| Koyeb | No free compute | n/a | n/a | $29 plan |
| AWS Lambda + Web Adapter | Possible (SSE through response streaming) but needs work | Cold start not measured | Yes | ~$0 in the free tier |
| AWS App Runner | Easy container host | Pausable | Yes | ≈$2.5 idle + active time |
| Streamlit Community Cloud (UI) | Current UI | 12 h sleep; visitor clicks to wake | No | $0 |
| Static UI (HF static Space, GitHub Pages, Cloudflare Pages) | Needs a JS front end. **A browser cannot hold `API_KEY`**, so this needs a public rate-limited demo route or a proxy | None for the page itself | No | $0 |

**Recommended hosting:** keep the **existing HF Space** for the API. It is free, the most generous free CPU tier, and its wake time is about 10 s versus about 60 s on Render. Keep **Streamlit Community Cloud** for the UI once its lock is slimmed. Optional keep-warm: a scheduled GitHub Actions ping to `/health` every 24 h keeps the Space under the 48 h sleep window. Whether HF's terms allow this is UNVERIFIED, so it needs your approval. Fallback: Cloud Run, if you accept adding a billing account.

## 7. Direction options (choose one)

| Option | What | Effort | $/mo | Risk | Portfolio value |
|---|---|---|---|---|---|
| **A. Minimal revival** | Set `GROQ_MODEL=openai/gpt-oss-20b` in config; re-validate the judge and grader parsing; re-lock with hashes (vulnerabilities fixed); slim the UI lock; hermetic tests; deprecated-model smoke test; CI deploy to the existing Space; README fixes | 1–2 days | $0 | Low | Medium: "it works again" |
| **B. A + re-platform the backend** | Same as A, but move the API to Cloud Run (scale-to-zero, a few seconds cold, billing account) or Render (1 min cold), with a keep-warm or "waking up…" UI state | +1–2 days | $0 (card needed) | Medium: new platform, a card on file | Medium: shows cloud deployment skills, solves nothing HF doesn't |
| **C. A + modern patterns** | Provider-agnostic LLM client (ordered model fallback chain from config, e.g. Groq gpt-oss-20b, then gpt-oss-120b, then a second provider); `context=` and lifespan; pinecone 10.x; drop langchain-community; per-model cost table from config | 3–5 days | $0 | Medium: more surface, so the invariants are re-run at every step | **High**: "fails safe even when a provider deprecates a model", directly answering the failure that broke it |
| **D. A + static front end** | Replace Streamlit with a small static SPA (HF static Space, Pages) using SSE streaming. Needs a public demo route with strict rate limits, or a proxy, because a browser cannot keep `API_KEY` secret | +3–4 days | $0 | Medium: auth model change | Medium–high: faster UX, no 12 h sleep |
| **E. AWS-native variant** | Bedrock generation (gpt-oss or Nova), Bedrock Knowledge Bases or S3 Vectors retrieval, Lambda or App Runner, reusing the eval harness (golden labels are document-based, so they carry over; the corpus must be re-ingested and the metrics re-measured) | 1–2 weeks | ~$1–10 plus KB storage at $5/GB (estimate) | High: new embedder could shift the metrics; account closes after 6 months on the free plan | High for AWS-targeted roles |
| **F. (Suggested) A now, then C-lite** | Ship A first so it is live within a day, then add only the model-fallback chain and deprecation smoke test from C | 2–3 days | $0 | Low | High per day of effort |

**Recommendation: F.** The project broke because a provider retired a model. The highest-value fix is to make that failure loud (smoke test) and survivable (fallback chain) while keeping the free hosting that already works. B adds a card and solves nothing HF doesn't. D and E are good follow-ups after F, not instead of it.

## 8. What you must supply before Phase 4

- **Keys, for read-only checks:** `GROQ_API_KEY`, and `PINECONE_API_KEY` + `PINECONE_INDEX_NAME` + `PINECONE_HOST` (for the index state and the live eval baseline). `TAVILY_API_KEY` is optional. You can also tell me to use the existing local `backend/.env.docker` or `.env` without my reading the values.
- **Approval:** for deploying to the Space and pushing to GitHub. Both are needed in Phase 4 only.


## 9. Phase 4 results (option F: A + C-lite), 2026-10-19

The work is on local branches merged into `revive/2026-10`: `fix/hermetic-tests`, `feat/model-chain`, `feat/model-smoke-check`, `chore/relock-deps`, `chore/ui-lock-and-ci` and `docs/revive-readme`. Nothing is pushed or deployed. The API repo is untouched apart from its new branch.

**Keys:** both `GROQ_API_KEY` and `GROQ_API_KEY_1` are valid. `GROQ_API_KEY_1` is not needed and nothing reads it.

### Baseline vs final

| Check | Baseline (as found) | Final |
|---|---|---|
| Tests (hermetic) | 343 passed | **351 passed**, 0 warnings (was 6–32) |
| Tests read developer `.env` | yes (LangSmith tracing active) | no (`DISABLE_DOTENV=1` in `tests/conftest.py`) |
| Retrieval eval, 30 queries, k=10 | R@10 0.9694 · MRR 0.9833 · nDCG@10 0.9280 | **identical** |
| k sweep at k=5 (production) | R 0.9139 · nDCG 0.9030 · P 0.36 | **identical** |
| `/chat` on default config | 404, generation fails | answers; grounded 1.0 on the RAG probe |
| Known vulnerabilities (pip-audit) | 52 in 15 packages (backend) | **0** across all 4 locks |
| Hash-pinned locks | backend only | backend, UI, dev, docling (all `--universal`) |
| UI lock | Linux-only, about 3 GB of CUDA, no hashes | 48 packages, hashed, installs on Windows and Linux |
| UI boots with env-only config | **crashed** (`st.secrets` raised) | fixed; CI boot test added |
| Stream time to first token (local, gpt-oss-20b) | n/a (broken) | 1.25 s with `reasoning_effort=low` (6.4 s at default effort) |
| Faithfulness judge latency | n/a | about 0.7 s (2–12 s at default effort) |
| Cost per RAG answer (estimate) | `None` (model not priced) | about $0.0003 |

### Changes

- **Model chain (config only):** `GROQ_MODEL=openai/gpt-oss-20b`, `LLM_FALLBACK_MODELS=openai/gpt-oss-120b` (via `Runnable.with_fallbacks`), and `LLM_REASONING_EFFORT=low`. Fallback was verified live: the retired model as primary returned 404, then gpt-oss answered for both `invoke` and `astream`.
- **`scripts/check_models.py`, `make check-models` and a weekly `model-check.yml` workflow** fail loudly (exit 1) when a configured model is not listed, does not answer, or appears in a heading on Groq's deprecations page. Verified: exit 0 on the defaults, exit 1 with `llama-3.1-8b-instant`.
- **Re-lock:** langchain-core 1.6.12, langgraph 1.2.16, langchain-openai 1.7.2, pinecone 10.0, fastapi 0.148, uvicorn 0.54, streamlit 1.66, docling 2.140. Dropped `langchain-community` (Tavily now goes through `tavily-python`) and `orjson`/`ORJSONResponse` (deprecated). Moved to a FastAPI lifespan handler.

### Bugs found and fixed during the revival

1. **pinecone 10 regression (caught by the live probe, not by the eval):** `to_dict()` now returns `id_`/`score_`. Every score read as 0.0 and every chat abstained. Fixed once in the shared `search()`. `eval/run.py` now fails if hits lack `_score`, because ranking metrics alone could not see this. The system failed safe: it abstained rather than answering.
2. **Pre-existing: LangGraph never passed `RunnableConfig` to 4 nodes.** `from __future__ import annotations` made the annotations strings, so LangGraph skipped injection with only a warning. A regression test now fails if this returns.
3. **Pre-existing: env-only UI config crashed** on `st.secrets` access when no `secrets.toml` exists.

### Decisions to confirm

- **docling is not in the hosted UI lock.** PDFs use pypdf; Office files show an install hint. `requirements-docling.txt` restores it locally. Roadmap item 20 moves conversion server-side.
- **CRAG query laundering** (live, CRAG enabled): the rewriter once turned an injection prompt into a benign question and answered it. It did not leak the system prompt. CRAG is off by default. Roadmap items 2 and 3 address it.

### Not done or not verified

- **Docker image build:** Docker Desktop was started but its engine had not come up when checked (it may be waiting on a first-run dialog). CI and the Space build on Linux; `--require-hashes` in the Dockerfile has not been built locally.
- **Corpus manifest drift:** the manifest lists 8 doc_ids, the live `eval` namespace has 23. This is pre-existing; regenerate with `make corpus-manifest` (roadmap item 28).
- **Deploy:** not done; it needs your approval (see the manual steps below).

### What you must do manually (before or for deploy)

1. **HF Space settings:** remove or replace any `GROQ_MODEL=llama-3.1-8b-instant` variable or secret. The new default applies only if it is unset, or set it to `openai/gpt-oss-20b`.
2. **GitHub repo secrets** (only if you want the workflows): `GROQ_API_KEY` for the weekly model check and the deploy gate; `HF_TOKEN` (write access to the Space) for `deploy-space.yml`.
3. **Approve:** pushing `revive/2026-10` to GitHub (then a PR to `main`; Streamlit Cloud redeploys from `main`) and deploying the backend to the existing Space.
4. Optional: rotate nothing (no secrets were ever committed); `GROQ_API_KEY_1` can be removed from `.env`.

## 10. Roadmap Wave 1 results, 2026-10-19 (built locally; not pushed or deployed)

| # | Item | Result |
|---|---|---|
| 3 | CRAG rewrite guard | Deterministic: refusal, answer-shaped and new-number rewrites are rejected; instruction-override queries never reach the rewriter; the rewrite prompt delimits the query as data. Live, CRAG on, 4 runs each: injection 4/4 skipped and abstained; off-topic 4/4 abstained with clean rewrites (no more invented scores); in-domain 4/4 answered |
| 4 | Strict structured judge | JSON-schema strict `response_format` on the faithfulness judge. Paired A/B on 30 golden answers: plain and strict both had **0 parse failures and 0 mistyped verdicts**, so this is a guarantee, not a measured gain on gpt-oss. It also fixed a latent bug: `bool("false")` counted a string "false" verdict as grounded |
| 1 | Answer-quality and abstention eval | `make eval-answers`; 30 answerable (87 reference claims) + 12 unanswerable questions; judge = gpt-oss-120b (≠ generator); calibration CSV → κ. **Partial live run** (21 of 42 scored before Groq's daily token cap): false-abstain 0/13, correct-abstain 8/8, claim recall 0.936, faithfulness 0.946, citation precision 0.861, 0 dangling citations. The hallucination probe (#30; the corpus never mentions hallucination) was answered with an unsupported claim and correctly flagged (faithfulness 0.5, grounded=false). Finish with `python eval/run_answers.py --resume eval/reports/answers_20261019T154022Z.json` once the window frees |
| 2 | Adversarial eval | 10 cases, deterministic detection. CI: 4 structural defences hold against a fully compromised fake model (mutation-checked). Live (gpt-oss-20b, one run): **0/10 attacks succeeded**; the off-topic/override/CRAG attacks abstained with 0 LLM calls. The 6 model-dependent cases are a small sample; repeat runs before quoting a rate |
| 5 | MCP server | Read-only `ask` / `search_knowledge_base`, resources `rag://guardrails` and `rag://namespaces`, prompt `grounded_answer`; Streamable HTTP at `/mcp` behind the same API key (constant-time check), stdio via `python -m app.mcp_server`. Live: HTTP and stdio both work; ask 3.9 s answered / 0.3 s abstained; wrong key → 403. No ingestion tool by design (SSRF and index-write risk) |

**Also fixed:** pinecone 10 `list()` returns `ListItem` objects, which broke `make corpus-verify`; the API-key check was not constant-time; a Mount answered `POST /mcp` with a 307 that MCP clients do not follow (found by the live probe, not the TestClient).

**New verified limit (Groq response headers and 429 body, 2026-10-19):** the free tier for `openai/gpt-oss-20b` allows **8,000 tokens per minute, 1,000 requests per day and 200,000 tokens per day**, per model. At about 1.5k tokens per answer (judge off), that is roughly 130 answers a day before the fallback model's separate budget takes over. This makes the fallback chain an availability feature, not just a deprecation guard.

**Totals now:** 412 tests (hermetic), retrieval eval unchanged (0.9694 / 0.9833 / 0.9280), 0 known vulnerabilities across the 4 locks, UI smoke OK.

**Pending your review:** (a) the drafted reference claims in `eval/answers_golden.jsonl`; (b) golden.jsonl #21 likely missing the clustering doc `9e590954`; (c) human labels in `eval/labels/judge_calibration_*.csv` for judge κ.
