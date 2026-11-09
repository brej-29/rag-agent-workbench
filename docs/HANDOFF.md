# Compliance Workbench: handoff and portfolio notes

Written 2026-11-09. Facts below come from the repo, its README, `git log` and what was done in the sessions. Numbers marked "README" are the README's measured results (2026-11-08); re-check them before quoting if the corpus or models change. Nothing here is legal advice, and the app says so too.

## 1. One-paragraph summary

Compliance Workbench answers "which EU and US rules apply to my AI, data or cyber product?" with plain answers quoted from the official text of the EU AI Act, GDPR, NIS2 (plus Implementing Regulation 2024/2690), DORA, the Cyber Resilience Act and NIST (SP 800-53, CSF 2.0, AI RMF), each with the date the rule starts to apply. It is a measured RAG pipeline that declines instead of guessing. The repo is still named `rag-agent-workbench`, because the app grew out of an earlier general RAG backend.

## 2. Links

| What | Where |
|---|---|
| Repo | https://github.com/brej-29/rag-agent-workbench |
| App (Streamlit Community Cloud, follows `main`) | https://rag-agent-workbench.streamlit.app |
| API (Hugging Face Space, Docker) | https://brejbala-rag-agent-workbench-api.hf.space (docs at `/docs`, MCP at `/mcp`) |
| Landing site (GitHub Pages, built from `site/`) | https://brej-29.github.io/rag-agent-workbench/ |
| Design notes | `docs/DESIGN.md`; UI spec `docs/superpowers/specs/2026-11-08-ui-redesign.md`; compliance spec `docs/superpowers/specs/2026-10-20-compliance-copilot-design.md` |
| Demo media | `docs/media/` (home, ask, checklist, dark GIFs) |

## 3. What it does

| Tool | Gives you |
|---|---|
| Ask | A cited answer, each provision marked in force or from date X; a plain decline for rules it does not cover (HIPAA, CCPA, PCI DSS, licensed ISO/SOC 2 text) |
| Checklist | Which of 112 duty-bearing articles bind your product, now and later; 8 ready-made examples; export to Markdown or CSV; no model words or decides anything |
| Gap check | Paste a policy; a verdict per duty (covered, partial, missing) with quotes from both the law and the policy, checked word for word; documents stay in the browser session |
| MCP server | The same tools for Claude and other MCP clients (5 tools per the README badge) |

Corpus: 3,946 provisions from 9 official texts, each with a stable reference such as `GDPR Art. 33(1)`. Start dates live in `backend/app/data/regs/applicability.yaml`, every one with a verbatim quote of the law, and a test fails if a quote is not in the corpus.

## 4. Stack

Python 3.11 (API) and 3.13 (Streamlit Cloud), FastAPI, LangGraph, Pinecone (integrated embeddings), an OpenAI-compatible LLM gateway (OmniRoute) with Groq as fallback, Tavily web search (legacy path), Streamlit 1.66, Prometheus metrics, Docker, GitHub Actions, uv-compiled hash-pinned lockfiles, MCP over SSE.

## 5. Results (README, measured 2026-11-08)

| Measure | Result |
|---|---|
| Right article in top 6, 50 hand-labelled questions | recall 0.847, MRR 0.862 (dense 0.818, fused 0.779, lexical 0.607 on the same budget) |
| Questions about uncovered rules declined with no model call | 13 of 14 |
| Covered questions wrongly declined | 0 of 50 |
| Requests routed to the right tool | 34 of 36 |
| Gap check: planted gaps wrongly called "covered" | 0 in every run |
| Prompt-injection suite | 10 cases, no breaches |
| Tests | 677 passing, 1 skipped (CI-safe, no network) |

Method points worth stating: golden labels were set by reading documents, not by running the retriever (avoids circular validation); a human reviewed all four compliance golden sets on 2026-11-08; every parameter change needed a measurement.

## 6. Timeline

| Date | What happened |
|---|---|
| Earlier 2026 | General RAG backend: 8-node LangGraph, rerank, faithfulness judge, CRAG, history-aware retrieval, Prometheus, cost accounting (all behind flags, default off) |
| 2026-10-19 | Revival. Groq retired the model the Space used (llama-3.1-8b-instant, 2026-08-16), so live chat was broken; moved to a model chain, refreshed hash-pinned locks, added adversarial eval, MCP server |
| 2026-10-20 to 25 | Refocus on compliance, built as six sub-projects: corpus, router + Ask, applicability + checklist, crosswalks, gap check + private workspace, UI + MCP |
| 2026-11-08 | OmniRoute gateway, UI redesign ("Seven rulebooks, one desk"), PR #28 merged to `main`; API deployed to the Space |
| 2026-11-08 | SEO landing site: PR #31 (site), PR #32 (Dependabot ignore for docling), PR #33 (visual redesign); Pages enabled and live |
| 2026-11-08 | Google Search Console: HTML-tag verification worked, home page indexing requested; sitemap submitted |

## 7. Problems faced and how they were solved

| Problem | Cause | Fix or lesson |
|---|---|---|
| Live `/chat` broken | Groq retired the Space's default model | Config-driven model chain, then an OpenAI-compatible gateway with Groq as fallback |
| Free-tier limits (Groq 8k TPM, 1k RPD, 200k tokens/day per model) stalled eval runs | A full answer eval plus debugging exhausts a day | Resumable eval runs (`--resume`); never rotate keys to dodge limits |
| Model invented user profile facts | gpt-oss-20b filled gaps | Every extracted fact must quote the user's own words (router "evidence" check) |
| Wrong dates are a legal risk | Dates scattered in code | One registry with verbatim quotes, enforced by a test |
| Gap check could flatter a policy | LLM judging coverage | Quote-verified verdicts; golden gate requires no false "covered" |
| Streamlit is invisible to search engines | SPA serves only a noscript line | Static `site/` generated by `site/build.py` from the same data files; test checks every date and quote on the pages |
| PR #28 merged without the site | The site commit had never been pushed | Rebased onto `main`, PR #31 |
| `.gitignore` hid the whole `site/` folder | Template rule `/site` (mkdocs) | Narrowed to `/site/dist/` |
| Dependabot run failed | `docling` pulls `torch`, which Dependabot's resolver cannot install | PR #32 ignores docling; bump it by hand |
| Space deploy workflow cannot push | No `HF_TOKEN` secret in the repo | Deployed by local `hf upload` from a `git archive` of `main` (still to fix, see section 9) |
| Docker Desktop engine would not start locally | Machine issue | Tested without Docker |
| Search Console "Couldn't fetch" on the sitemap | New site; the file itself is valid (HTTP 200, 9 URLs) | Wait; URL Inspection live test passed and indexing was requested |
| Test suite needs `mcp` | Only some local venvs have it | Use `.venv-relock` for the full suite |

## 8. Process and design decisions (good interview material)

- Spec, plan, TDD, eval gate, then merge; no Pinecone upsert without approval.
- Compliance path is gated on the namespace `regs-v1`; legacy namespaces behave exactly as before. Flags default off so the off path is unchanged.
- Cosine-gated abstention and a relevance floor: declining is a feature, and is measured.
- Crosswalks use only published mappings and say when none exists. Retrieval with crosswalks was measured, showed no gain, and was left off.
- The landing site reuses the app's tokens, fonts and copy by importing its stdlib-only helpers, so site and app cannot drift.
- Hash-pinned locks, pip-audit in CI, a weekly model-availability check.

## 9. Still open

| Item | Owner |
|---|---|
| Check Search Console in about 3 days: home page indexed? sitemap status? Request indexing for `faq.html`, `gdpr.html`, `eu-ai-act.html` | You |
| Put the site address in the repo "Website" field | You |
| Add `HF_TOKEN` (and OmniRoute secrets) as repo secrets so the Space deploy workflow works; never put the OmniRoute key on Streamlit Cloud | You |
| Optional custom domain for the site (costs money, price not checked); set repository variable `SITE_URL` | You |
| Bump `docling` by hand when wanted | You |
| Dependabot PRs #29 and #30 are open; #30's title says frontend but it changes only the backend and dev locks, so check before merging | You |
| Not verified: visual check of the live Pages site after deploy (HTML and links were checked, the local build was inspected in a browser) | You |

## 10. Portfolio and LinkedIn material

### Elevator pitch (30 seconds)

"I built a compliance copilot for the EU AI Act, GDPR, NIS2, DORA, the Cyber Resilience Act and NIST. It answers in plain language, quotes the exact article, and tells you the date the rule starts to apply. The part I care about most is that it says no when it should: it declined 13 of 14 questions about rules it does not cover, and I measured that."

### Resume bullets (edit to fit)

- Built a RAG compliance assistant over 3,946 provisions from 9 official EU and NIST texts (LangGraph, FastAPI, Pinecone, Streamlit, MCP server).
- Designed an eval-first workflow: hand-labelled golden sets, recall 0.847 at top 6, 0 of 50 covered questions wrongly declined, 0 false "covered" verdicts in gap check across runs, 677 CI tests.
- Enforced legal accuracy in code: every start date carries a verbatim quote of the law, and a test fails if a quote or date on any page is not in the source data.
- Shipped a stdlib-only static landing site generated from the same data, with SEO basics (canonical URLs, Open Graph, JSON-LD, sitemap) and a GitHub Pages workflow.

### LinkedIn post angles

| Angle | Hook | Proof to use |
|---|---|---|
| Declining is a feature | "My RAG app is proud of the questions it refuses." | 13 of 14 declined, 0 of 50 wrongly declined |
| Evals before tuning | "I labelled the test set by reading the law, not by running my retriever." | Anti-circular validation; human-reviewed golden sets |
| A wrong date is a legal bug | "In compliance, a wrong date is a defect, so I made it fail the build." | Registry of verbatim quotes plus the test |
| SEO for a Streamlit app | "Streamlit apps are nearly invisible to Google. Here is the fix I shipped." | noscript-only HTML, static site from the same data, Search Console steps |
| Revival story | "My project broke when a model was retired. Here is what I changed." | Groq retirement 2026-08-16, model chain, gateway |

### Draft post (first person, edit freely; no em dashes)

> Most RAG demos try to answer everything. Mine is proud of what it refuses.
>
> I built Compliance Workbench: ask which EU and US rules apply to your AI, data or cyber product (EU AI Act, GDPR, NIS2, DORA, the Cyber Resilience Act, NIST) and get a plain answer that quotes the article and shows the date the rule starts to apply.
>
> What I measured: recall 0.847 in the top 6 results on 50 hand-labelled questions, 13 of 14 questions about uncovered rules declined with no model call, 0 of 50 covered questions wrongly declined, and 0 policy gaps wrongly called "covered" in the gap check.
>
> Two decisions I would repeat: I labelled the test sets by reading the law, not by running my own retriever, and every start date must carry a verbatim quote or the build fails.
>
> It is information from official text, not legal advice. Link in the comments.

### Screenshots and assets

`docs/media/home.gif`, `ask.gif`, `checklist.gif`, `dark.gif`; share image `site/static/og.png`.

### Questions to expect in interviews

Why abstain instead of answering? How were golden sets labelled and why not from retriever output? How do you stop an LLM inventing legal dates? What did the crosswalk experiment show and why leave it off? How would you scale the gap check? What would you do about stale law (amendments, consolidated versions)?
