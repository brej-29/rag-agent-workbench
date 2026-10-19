# Resurrection Report: RAG Agent Workbench

Audit date: 2026-10-19. Branch: `revive/2026-10` in both repos. Status: **Phase 3 complete, waiting for a direction choice.**

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
| Space `/chat` generation | **Almost certainly broken.** The default model `llama-3.1-8b-instant` was shut down on 2026-08-16 | Groq deprecations page (VERIFIED). A live call needs your Space `API_KEY` and Groq key, so the failure itself is UNVERIFIED |
| Streamlit UI `rag-agent-workbench.streamlit.app` | **Asleep** ("This app has gone to sleep due to inactivity"). I did not wake it | Browser fetch (VERIFIED) |
| GitHub Actions CI | **Green.** The last CI run was 2026-08-12, on Dependabot PRs. Three Dependabot PRs are still open (#24, #25, #27) | `gh run list` / `gh pr list` (VERIFIED) |
| Pinecone index | **Unknown, needs a key.** Starter plan: 5 indexes, 2 GB, `us-east-1` only, 1M RU and 2M WU a month, 5M embed tokens a month per model, 500 rerank requests a month | Pinecone pricing, object-limits and rate-limits pages (VERIFIED). The current docs contain no inactivity pause or archive rule. A Pinecone blog post says auto-archiving was removed (UNVERIFIED, search summary only) |
| Groq | Key and model status need your key | See section 2 |
| Tavily | Unknown, needs a key. Free plan: 1,000 credits a month, basic search 1 credit, no card | docs.tavily.com/documentation/api-credits (VERIFIED) |

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
| 3 | High | The UI lock bundles docling, torch and CUDA, although `frontend/app.py` imports only streamlit and httpx. It is not hash-pinned and is Linux-only, which risks Streamlit Cloud's 2.7 GB cap and slow builds | S (move ingest tooling to its own lock; UI lock becomes streamlit + httpx with hashes) |
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
