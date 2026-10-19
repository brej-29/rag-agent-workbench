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
