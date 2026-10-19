# Changelog

## [Unreleased] — revive/2026-10

### 2026-10-19 — Phases 0–3 (audit only, no code changes)
- Backups: git bundles + working-tree archive in `D:\Project\_backups\revive-2026-10-19\`.
- Created branch `revive/2026-10` in both repos.
- Added `RESURRECTION_REPORT.md` (status, verified risks, baseline, hosting analysis, options).
- Baseline: 343/343 tests pass hermetically; live eval not yet re-run (needs Pinecone key).

### 2026-10-19 — Phase 4: option F (minimal revival + modern patterns lite)
- fix(tests): hermetic suite — `DISABLE_DOTENV=1`, tracing off (`tests/conftest.py`).
- feat(llm): config-driven chain `GROQ_MODEL` (default `openai/gpt-oss-20b`) + `LLM_FALLBACK_MODELS`
  (`openai/gpt-oss-120b`) + `LLM_REASONING_EFFORT=low`; gpt-oss prices in cost table.
- feat: `scripts/check_models.py`, `make check-models`, weekly `model-check.yml`.
- chore(deps): re-lock all four locks with hashes (`--universal`); 52 vulns -> 0; drop
  langchain-community and orjson; FastAPI lifespan.
- fix: pinecone 10 `id_`/`score_` keys normalised in `search()`; eval fails without `_score`.
- fix: LangGraph `RunnableConfig` injection restored on 4 nodes (removed future annotations).
- fix(ui): env-only config no longer crashes on `st.secrets`; UI lock slimmed, docling split out.
- ci: `--require-hashes`, UI boot smoke (py3.13), pip-audit on all locks; manual `deploy-space.yml`.
- docs: README live/eval/limits section, `docs/ROADMAP.md`.

### 2026-10-19 — Roadmap Wave 1
- feat(crag): deterministic rewrite guard + hardened rewrite prompt.
- feat(judge): strict JSON-schema structured output; fix string-boolean verdict parsing.
- feat(eval): answer-quality/abstention harness (`make eval-answers`, `--resume`, `--calibrate`).
- feat(eval): adversarial suite (`make eval-adversarial`; structural cases in CI).
- feat(mcp): read-only MCP server at `/mcp` (Streamable HTTP) and stdio; constant-time API-key check.
- fix(eval): corpus manifest on pinecone 10 (`ListItem`).

### 2026-10-19 — Roadmap Wave 2 (Streamlit UI)
- feat(stream): `step` SSE events for a live pipeline trace; `doc_id` on sources.
- feat(ui): top-nav shell, light/dark themes, cold-start wake indicator, live trace with relevance-floor bars,
  citation badges, evidence drawer, Evaluation page from `eval/results/`; markdown images stripped from answers.
- fix(ci): UI smoke test path for Streamlit 1.66 AppTest.
