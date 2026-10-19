# Compliance copilot for AI and cloud teams: design

Date: 2026-10-20 · Status: approved in conversation, pending written review · Branch base: `revive/2026-10`

## 1. Problem (black box)

For teams shipping AI and cloud products: describe your product once, then ask what the rules say, get a
dated checklist of what you must do, or check your own policies for gaps. Every answer quotes the exact
clause, says when it takes effect, shows the matching requirement in other frameworks, and says plainly
when nothing applies.

Why this is not a search engine: answers depend on (a) the user's product profile, (b) the date (rules
switch on in phases and get amended), and (c) mappings between frameworks. The current RAG demo over
papers and Wikipedia does none of these.

## 2. Decisions taken

| Topic | Decision |
|---|---|
| Domain | Regulations and standards for AI, IT and cloud |
| Frameworks | EU AI Act (+ amendments), GDPR, NIS2, DORA, Cyber Resilience Act, NIST AI RMF, NIST CSF 2.0, NIST SP 800-53; ISO/IEC 27001 and 42001 and SOC 2 via licensing rule below |
| Licensing | **Open core + bring-your-own.** Public corpus holds only openly reusable texts. ISO and SOC 2 appear only as clause identifiers inside published crosswalks. Users who own licensed copies upload them to a private workspace that is never part of the public demo |
| Jobs | All three: **Ask**, **Checklist**, **Gap check**, chosen per request by an orchestrator |
| Orchestration | **Typed router + three bounded workflows** (not a free tool-calling agent) |
| Build order | Corpus → Router + Ask → Applicability + Checklist → Crosswalks → Gap check + private workspace → UI + MCP |

## 3. Verified facts this design relies on (2026-10-19/20)

| Fact | Status | Source |
|---|---|---|
| EU legal texts are reusable with attribution (Commission Decision 2011/833/EU) | VERIFIED (decision text); EUR-Lex notice wording to confirm | eur-lex.europa.eu/eli/dec/2011/833/oj |
| ISO text may not be copied or redistributed without permission, including OBP previews | VERIFIED | iso.org/copyright.html; ISO OBP terms |
| SOC 2 Trust Services Criteria: free download, separate copyright terms; translations used "with permission" | UNVERIFIED terms; treat as licensed | aicpa-cima.com download page |
| NIST AI Resource Center hosts an AI RMF ↔ ISO/IEC 42001 crosswalk (community-contributed) | UNVERIFIED (secondary sources) | to confirm on airc.nist.gov |
| AI Act official XHTML via CELLAR content negotiation: 113 `art_N`, 180 `rct_N`, 13 `anx_N` elements | VERIFIED | publications.europa.eu/resource/celex/32024R1689 (Accept: application/xhtml+xml) |
| EUR-Lex HTML returns 202 with empty body to scripted clients | VERIFIED | curl, 2026-10-19 |
| AI Act Art. 113 base dates: in force 20 days after publication; applies 2 Aug 2026; Ch. I–II 2 Feb 2025; Ch. III s.4, Ch. V, VII, XII, Art. 78 2 Aug 2025 (not Art. 101); Art. 6(1) 2 Aug 2027 | VERIFIED (official text) | CELEX 32024R1689 |
| Regulation (EU) 2026/1744 (Digital Omnibus on AI), OJ L 24.7.2026: Chapter III Sections 1–3 apply from 2 Dec 2027 (Art. 6(2)/Annex III) and 2 Aug 2028 (Art. 6(1)/Annex I) | VERIFIED (recital text); operative article and entry-into-force date to read at build | CELEX 32026R1744 |
| NIST SP 800-53 Rev 5 OSCAL JSON catalog | VERIFIED (HTTP 200, 10.4 MB) | raw.githubusercontent.com/usnistgov/oscal-content/.../NIST_SP-800-53_rev5_catalog.json |
| NIST CSF 2.0 OSCAL catalog (6 functions, 22 categories, 106 subcategories) | UNVERIFIED location | OSCAL pocket guide, 2026-03 |
| Groq free tier, gpt-oss-20b: 8k tokens/min, 1k requests/day, 200k tokens/day per model | VERIFIED (response headers / 429 body) | api.groq.com |

## 4. Architecture

```
request + (profile, as-of date) ──► Orchestrator (LangGraph)
  1 extract_profile  structured output; merges stated facts into the session profile; never invents fields
  2 route            ask | checklist | gap_check | clarify; deterministic overrides first
                     (attachment → gap_check; missing required profile fields → clarify)
  3 workflow         one bounded subgraph per intent
  4 verify + format  provision-level citation check, faithfulness, "applies from" notes, crosswalk refs

Ask        hybrid retrieval (dense + lexical, fused) filtered by in-scope frameworks and as-of date
           → relevance-floor gate (abstain when nothing clears it) → cited answer
Checklist  applicability engine (deterministic rules) → obligations for the profile → dated checklist
Gap check  policy split into passages → per applicable obligation (bounded N): covered / partial /
           missing, with quotes from both the clause and the policy
```

**Invariants (kept and extended):** bounded loops; deterministic abstention when nothing clears the
floor; answers may cite only provisions actually retrieved; faithfulness checks independent of the
embedder; uploaded documents are untrusted input (existing injection defences apply); applicability is
never decided by the LLM (rules table only); "not legal advice" is shown with every answer.

**Shared data:** provisions (the unit of retrieval and citation), the applicability registry and rules,
the crosswalk table (published mappings only), and the private workspace (per-user namespace).

**Reused unchanged:** FastAPI app, SSE trace (new step ids), cosine floor, bounded CRAG and its rewrite
guard, structured judge, answer/adversarial harnesses (new golden sets), MCP server (new tools), UI shell,
theme and evidence drawer. **Retired from the default path:** the arXiv/Wikipedia corpus (kept as a
legacy namespace for regression only).

## 5. Sub-projects

| # | Sub-project | Delivers | Eval gate |
|---|---|---|---|
| 1 | **Corpus foundation** (this spec, §6) | Provision corpus, applicability registry, dense + lexical indexes | Provision-level retrieval golden set; registry unit tests |
| 2 | Router + Ask | Profile extraction, intent routing, as-of/framework filters, hybrid retrieval in the live path, cited answers | Router accuracy; as-of questions; out-of-scope abstention |
| 3 | Applicability + Checklist | Rules table with legal basis per rule; dated obligation checklist; export | Golden profiles → expected frameworks/obligations (deterministic, CI) |
| 4 | Crosswalks | Published mappings (e.g. AI RMF ↔ ISO 42001 by clause id) attached to answers and checklist items | Mapping lookups match source tables |
| 5 | Gap check + private workspace | Policy upload, per-obligation coverage, BYO licensed standards in a private namespace | Precision/recall on policies written with known gaps |
| 6 | UI + MCP | Profile panel, as-of date, checklist and gap pages, obligation cards; MCP tools for all jobs | AppTest smoke; MCP in-memory tests |

Each sub-project gets its own plan and goes through the existing gates (hermetic tests, eval harness,
adversarial suite). Open Wave 2/3 roadmap items fold in: hybrid retrieval and context packing (1–2),
record/replay harness (all), feedback loop (6), server-side document conversion (5).

## 6. Sub-project 1: corpus foundation

### 6.1 Scope

In: fetch, parse and normalise the v1 sources into provision records; the applicability registry; a
committed corpus snapshot with attribution; a local lexical index; dense vectors in a new Pinecone
namespace; a provision-level retrieval golden set; a dense vs hybrid measurement.
Out: router, profiles, checklist, crosswalks, gap check, UI.

### 6.2 Sources (v1)

| Framework | Instrument | Fetch | Licence |
|---|---|---|---|
| EU AI Act | Reg. (EU) 2024/1689 + Reg. (EU) 2026/1744 | CELLAR XHTML (CELEX 32024R1689, 32026R1744); prefer an official consolidated version if one exists | EU reuse, attribution |
| GDPR | Reg. (EU) 2016/679 | CELLAR XHTML (32016R0679) | EU reuse |
| NIS2 | Dir. (EU) 2022/2555 | CELLAR XHTML (32022L2555) | EU reuse |
| DORA | Reg. (EU) 2022/2554 | CELLAR XHTML (32022R2554) | EU reuse |
| Cyber Resilience Act | Reg. (EU) 2024/2847 | CELLAR XHTML (32024R2847) | EU reuse |
| NIST SP 800-53 Rev 5 | catalog | OSCAL JSON (usnistgov/oscal-content) | US public domain (verify statement) |
| NIST CSF 2.0 | catalog | OSCAL JSON (location to verify) | US public domain |
| NIST AI RMF 1.0 | NIST AI 100-1 | official PDF/Playbook (format to verify) | US public domain |

Every fetch records URL, retrieval date, content hash and licence in `corpus/manifest.json`. A source
that cannot be fetched from an official endpoint is left out, not scraped from a third party.

### 6.3 Provision record

```
id            eu-ai-act/art-26/par-1            stable, path-like
ref           AI Act Art. 26(1)                 human citation
framework     EU AI Act
instrument    Regulation (EU) 2024/1689
kind          article_paragraph | recital | annex_point | control | control_enhancement | subcategory
title         Obligations of deployers of high-risk AI systems
text          paragraph text, points (a), (b)... kept inside their paragraph
path          [Chapter III, Section 3, Article 26]
binding       regulation | directive | voluntary_framework
applies_from  2027-12-02 | null                 computed from the registry
date_basis    eu-ai-act-amend-2026-1744/art-1/...  provision that sets the date
amended_by    [Regulation (EU) 2026/1744]
source_url, licence, version_date
```

Recitals are stored as their own kind and down-weighted at retrieval: they guide interpretation but do
not create obligations.

### 6.4 Applicability registry

`corpus/applicability.yaml`, hand-reviewed. Entries apply a date to a set of provisions (by instrument,
chapter, section, article or annex), each with `basis` (the provision that establishes it). Later
amendments override earlier entries; the registry resolver returns `(applies_from, basis)` for any
provision and an `applies(provision, as_of)` predicate. Directives carry `binding: directive` and the
transposition deadline, with the note that obligations reach companies through national law.

### 6.5 Storage and indexes

- `corpus/regs-v1.jsonl`: the processed corpus, committed, with `corpus/NOTICE.md` (attributions).
- Lexical index: BM25 built from the JSONL at startup (small corpus; no new service).
- Dense: new Pinecone namespace `regs-v1` in the existing index (integrated `llama-text-embed-v2`);
  metadata `framework`, `kind`, `binding`, `applies_from` as an integer `yyyymmdd` for range filters.
  **This is a write to the user's Pinecone account: estimate records, storage and embedding tokens
  against Starter limits (2 GB; 5M embedding tokens per month per model) and get approval before the
  first upsert.** Existing namespaces are not touched.

### 6.6 Evaluation (regression gate for later sub-projects)

- `eval/regs_golden.jsonl`: about 40 questions, relevant provision ids labelled by reading the texts
  (same anti-circularity rule as today), mixing reference-style ("What does Article 26 require of
  deployers?") and concept questions, across all v1 frameworks.
- Registry unit tests, including: as of 2026-10-20, AI Act Annex III high-risk obligations do not yet
  apply (from 2027-12-02, basis 2026/1744); prohibited practices apply since 2025-02-02.
- Retrieval metrics at provision level (recall@k, MRR, nDCG@k) for dense-only, lexical-only and fused;
  the fusion weight is chosen from this run and recorded.

### 6.7 Risks

- EUR-Lex HTML blocks scripted clients; CELLAR works today but could change. Mitigation: committed
  snapshot plus content hashes.
- The free LLM tier is not used in sub-project 1 except for optional labelling help; labels are human.
- Corpus size vs Pinecone free tier: measured before upload (§6.5).
- Amendments after the snapshot: the manifest records version dates; a re-fetch diff shows changes.

## 7. Out of scope (all sub-projects)

Legal advice; non-EU national laws other than via NIST frameworks; automatic monitoring of new
legislation; storing licensed standards in the public corpus.
