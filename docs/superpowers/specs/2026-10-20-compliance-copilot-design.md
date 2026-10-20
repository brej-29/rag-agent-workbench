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

## 8. Sub-project 1 results (2026-10-20)

Corpus `regs-v1`: 3,830 provisions (AI Act 854 incl. 180 recitals, GDPR 563, NIS2 349, DORA 375, CRA 475,
SP 800-53 1,014, CSF 2.0 128, AI RMF 72); every date quoted from its basis provision and checked in CI.
Uploaded to Pinecone namespace `regs-v1` (3,830 vectors) with approval.

Retrieval, 46 hand-labelled questions, provision level, k = 10:

| Mode | Recall | MRR | nDCG | reference R/MRR | concept R/MRR | cross R/MRR |
|---|---|---|---|---|---|---|
| Lexical (BM25) | 0.661 | 0.436 | 0.473 | 1.00 / 0.54 | 0.69 / 0.46 | 0.29 / 0.24 |
| **Dense** | **0.864** | **0.750** | **0.729** | 1.00 / 0.71 | 0.91 / 0.77 | 0.48 / 0.64 |
| Fused RRF, lexical weight 1.0 | 0.846 | 0.580 | 0.611 | 1.00 / 0.56 | 0.90 / 0.60 | 0.43 / 0.49 |
| Fused RRF, lexical weight 0.25 | 0.864 | 0.703 | 0.698 | 1.00 / 0.71 | 0.91 / 0.73 | 0.48 / 0.53 |

Decisions for sub-project 2 (measured, like the earlier reranker decision):
- **Dense retrieval is the default.** Fusion did not beat it at any tested weight; BM25 stays available for
  explicit identifier lookups, to be re-measured with the reference resolver.
- **Cross-framework questions are the weak spot (recall 0.48)** for every method: a single ranked list fills
  with one framework. Sub-project 2 retrieves per in-scope framework and merges (diversification), measured
  on the `cross` subset.
- **A cosine floor alone cannot abstain correctly on legal questions.** 14 unanswerable questions
  (`eval/regs_unanswerable.jsonl`): off-topic ones score 0.09–0.29, but wrong-jurisdiction and licensed-text
  questions score as high as real answers (CCPA → CRA Art. 64 at 0.450; DPDP → GDPR recital 32 at 0.426;
  ISO 42001 → AI Act Art. 43 at 0.431), while the lowest answerable relevant hit scores 0.412 (overlap 6).
  Sub-project 2 therefore uses three layers: (1) a recalibrated floor near 0.30 for off-topic text,
  (2) a deterministic scope guard for named out-of-scope instruments (CCPA, HIPAA, DPDP, SOX, PCI DSS,
  UK Online Safety Act; ISO/SOC 2 → "bring your licensed copy"), (3) an instrument/jurisdiction mismatch
  check at verification. All three are scored on the unanswerable set.

## 9. Sub-project 2 design: router + Ask (approved 2026-10-20)

Scope: the compliance path for Ask, behind a typed router. Requests to namespace `REGS_NAMESPACE`
(`regs-v1`) take it; every other namespace runs the existing pipeline unchanged.

1. **Request context.** Optional `as_of` (default today), `frameworks` filter and `profile` (roles, AI system,
   GPAI model, high-risk class `none|annex_iii|annex_i|unknown`, personal data, EU market, sector).
2. **Router.** One strict-schema LLM call returns `intent` (`ask | checklist | gap_check | clarify`) and the
   profile facts the user stated (`profile_updates`, shown back, never guessed; invalid values dropped).
   Fails safe to `ask`. Only Ask is built here; the other intents get a plain "not available yet" reply.
   Measured on 30 labelled requests (`eval/regs_router.jsonl`).
3. **Abstention, three layers.** (1) scope guard before any LLM call: named out-of-scope instruments
   (CCPA/CPRA, HIPAA, DPDP, SOX, PCI DSS, UK OSA, LGPD, PIPL, PIPEDA) → "not covered, here is what is";
   licensed standards (ISO/IEC 27001/27002/42001, SOC 2) → "bring your licensed copy"; mixed questions get a
   warning instead; (2) floor `REGS_MIN_SCORE = 0.30`; (3) mismatch warning at verification when the
   question names a covered framework that no cited source comes from.
4. **Retrieval.** Dense; optional Pinecone filter on `framework`; 30 candidates, recitals weighted 0.9,
   diversified to at most 3 per framework, top 6. Pinecone only ranks ids: provision text comes from the
   committed corpus, status from the registry. No web fallback, no CRAG re-retrieval on this path.
5. **Answer.** Each context block is headed by its reference, framework, binding type and status on the
   as-of date ("in force since …", "applies from … (not yet in force)", "depends on classification",
   "voluntary framework"); the model is told to state applicability from those headers, never compute it.
   Responses (JSON and SSE `done`) add `as_of`, `intent`, `abstain_reason`, `warnings`, `profile_updates`
   and per-source `ref`, `framework`, `status`, `applies_from`, `status_note`.
6. **UI/MCP (minimal).** Ask defaults to `regs-v1` with as-of date, framework and high-risk controls and
   status chips; MCP `ask` takes `as_of` and `frameworks`. The full profile panel is sub-project 6.
7. **Gates.** Diversified retrieval: overall recall/MRR within 0.01 of dense and cross recall not lower;
   abstention (deterministic, no LLM): 14/14 unanswerable abstain, at most 1/46 false abstain; temporal
   status tests on real provisions; router accuracy reported; adversarial suite stays green.

## 10. Sub-project 2 results (2026-10-20)

Retrieval on the 46-question golden set (provision level), production ranking vs plain dense:

| Ranking | k | Recall | MRR | nDCG | cross recall (6 q) |
|---|---|---|---|---|---|
| Dense | 10 | 0.864 | 0.750 | 0.729 | 0.483 |
| **Diversified** (30 candidates, ≤3 per framework, recitals ×0.9) | 10 | **0.904** | **0.836** | **0.804** | 0.483 |
| Dense | 6 | 0.835 | 0.750 | 0.717 | 0.344 |
| **Diversified** (production) | 6 | **0.860** | **0.834** | **0.789** | **0.483** |

Sweep (k = 10): 60 or 100 candidates with a cap of 2 lift cross recall to 0.62–0.66 but lower single-framework
recall at k = 6 (0.831), so the defaults stay. Most of the MRR gain comes from down-weighting recitals, which
outranked the articles they explain. Cross-framework recall remains the weak spot; per-framework queries or
query decomposition belong with the crosswalks (sub-project 4).

Abstention (`run_regs.py --mode abstain`, deterministic, no LLM): **13/14** unanswerable questions declined
(scope guard 9, floor 4), **0/46** answerable questions refused. Gate was 14/14: the miss is "labelling rules for
allergens on packaged food in the EU", which scores 0.378 against AI Act Annex V (declaration of conformity);
the lowest answerable relevant hit is 0.412, so no floor separates them without overfitting this set. (§8's
"off-topic 0.09–0.29" was wrong for this question.) Live, the model declined it from the provisions ("the
provisions provided do not address allergen labelling"), i.e. the model-dependent layer caught it.

Router (`run_router.py`, gpt-oss-20b, 30 draft-labelled requests): **0.90**. All three misses were
checklist/gap-check requests classified as `ask` (the fail-safe direction: the user still gets an answer).

Applicability: status tests on 13 real provisions on 2026-10-20 plus the Annex III flip on 2027-12-02.
Live (local API, real index and Groq): "Do the AI Act's high-risk obligations already apply to a CV-screening
tool?" answers "not yet in force as of 2026-10-20 … from 2027-12-02 (2028-08-02 for Annex I)"; with as-of
2028-01-01 and Annex III it answers "yes" with every Art. 26 provision marked in force. HIPAA declines with no LLM
call; a checklist request is routed and its profile (deployer, Annex III, AI system, EU market, personal data)
extracted. About 2–3 s per answer, 1.8–3.1k tokens.

Adversarial suite (legacy namespace, unchanged prompt): structural failures 0 in three runs; model-dependent
attack success 1/18 over three runs (`exfil-roleplay` leaked once, then held twice), versus 0 in the Wave 1
baseline: model variance, not a code change, but it is the case to harden first.

Found and fixed live: gpt-oss sometimes cites as `【1】`, which bypassed the dangling-citation check and the UI
badges; answers are now normalised to `[n]` before verification.

## 11. Sub-project 3 design: applicability + checklist (approved 2026-10-20)

Job: given a profile and an as-of date, list which articles of the five EU laws bind you, since or from when,
which ones depend on facts not yet given (with the questions that decide them), plus voluntary NIST practices.
Deterministic throughout: no LLM decides or words anything in the checklist.

**11.1 Data.**
- `backend/app/data/regs/facts.yaml`: each profile fact once: `id`, allowed values, the question asked when it
  is unknown, and the provision that makes it matter. Facts: AI chain role (provider, deployer, importer,
  distributor); GDPR role (controller, processor); CRA role (manufacturer, importer, distributor of products with
  digital elements); AI system; GPAI model; GPAI with systemic risk; high-risk class (none, annex_iii, annex_i);
  AI uses that trigger Art. 50 (chatbot, synthetic content, emotion recognition / biometric categorisation,
  deepfakes); established in the EU; offers to or monitors people in the EU; size band (micro, small, medium,
  large; NIS2 Art. 2(1) via Recommendation 2003/361); NIS2 sector (tech-relevant Annex I/II types, other, none);
  DORA financial entity; processes personal data; special-category or criminal data at scale.
- `backend/app/data/regs/obligations.yaml`: one entry per duty-bearing article (about 90: AI Act ~30, GDPR ~30,
  DORA ~20, CRA ~12, NIS2 ~5). Fields: `article` (provision id prefix, e.g. `eu-ai-act/art-26`; the title is read
  from the corpus, never typed), `when` (condition over facts), `basis` (provision stating whom it binds) with a
  verbatim `quote`, optional `kind: prohibition`, optional `note` (e.g. narrower addressees of some paragraphs).
- Recommended practices (voluntary), chosen by rule: NIST CSF 2.0's 22 categories when any cyber law (NIS2, DORA,
  CRA) is in scope or unknown; NIST AI RMF's subcategories grouped by function when an AI system is built or
  deployed. SP 800-53 is not listed (reached through crosswalks, sub-project 4).

**11.2 Evaluator.** Conditions use `all`, `any`, `not` and fact comparisons (`fact: value` or a list of values),
evaluated in three-valued logic (yes / no / unknown). Item outcome: yes → applies; unknown → depends on the
unanswered facts that could change it; no → not applicable (with the deciding fact). Unknown facts are ranked by
how many items they would decide; the top three become follow-up questions. Dates come from the registry
(§6.4) using the profile's high-risk class: in force since, applies from, partly from (paragraph dates differ),
depends on classification, and the national-law note for NIS2. Groups: Applies now, Coming (by date), Depends
on your answers, Recommended practices (voluntary); not-applicable items are counted and expandable. Each item
shows why it applies (the facts that made it true). The checklist is a pure function of (profile, as-of date).

**11.3 Integration.** `services/regs/checklist.py` (load, evaluate, build) and `POST /checklist`
`{profile, as_of}` → grouped JSON (API key, rate limit). Chat: the `checklist` intent builds the checklist from
the merged profile; the reply carries a deterministic summary, the `checklist` object and the top questions.
The UI keeps one session profile, sends it with each request and merges `profile_updates`. Request field
`previous_intent`: when it is `checklist`, the message states facts (`profile_updates` non-empty), does not end
with a question mark and the router did not return `gap_check`, the intent is `checklist` (deterministic
override; a new question after a checklist is still answered). Facts stated in a message update the profile the
client sent (the message is the latest word; the Checklist page form remains the place to edit them). New Checklist page:
profile form ("Not sure" defaults) and as-of date; groups of item cards (official title, status, why, paragraphs
and official link via the evidence drawer); deciding questions as inline controls; Markdown and CSV download
built in the UI. The profile is shared with Ask. MCP `checklist` tool stays in sub-project 6.

**11.4 Gates.** (1) Catalog integrity in CI: articles exist, quotes verbatim in their basis, facts defined with
the compared values. (2) 12 golden profiles (`eval/regs_profiles.jsonl`), each listing the articles whose outcome
(binds / depends / does not bind) was labelled from the scope articles before the rules were written (draft,
for review); every listed outcome must match, and each mismatch is investigated as a rule or label bug. Labels
cover the legally interesting articles per profile rather than all ~99: full sets written by the rules' author
would only restate the rules. (3) Properties:
answering an unknown never moves an item from applies to depends; every depends item names a deciding fact;
an all-unknown profile marks nothing as applies. (4) Time: a high-risk deployer's AI Act
items move from Coming to Applies now on 2027-12-02 (Annex III); NIS2 items carry the national-law note.
(5) Routing: ~6 fact-only follow-ups added to `eval/regs_router.jsonl`; unit tests for the `previous_intent`
override. (6) UI: AppTest smoke for the page, export tests, a live two-turn chat check.

**11.5 Stated simplifications.** NIS2 size bands without linked/partner-enterprise calculations; national
transposition differences not modelled; article-level items whose paragraphs bind narrower addressees carry a
note. Not legal advice, shown on every view.

## 12. Sub-project 3 results (2026-10-20)

Built: 19 profile facts (`facts.yaml`), a 99-article catalog (`obligations.yaml`: AI Act 32, GDPR 34, DORA 19,
CRA 10, NIS2 4), a three-valued evaluator, `build_checklist`, `POST /checklist` + `GET /checklist/facts`, the
`checklist` intent in chat with fact-only follow-ups, and a Checklist page (profile form shared with Ask, article
drawer, Markdown/CSV export). Every catalog entry quotes, verbatim, the provision that says whom it binds; CI
rejects any quote not found in its basis. Arts. 9–15 of the AI Act and GDPR Arts. 6, 9, 10 have no addressee of
their own and cite the provision that puts them on someone (AI Act Art. 16(a); GDPR Art. 24(1)).

Gates (all deterministic, in CI): catalog integrity clean; 12 golden profiles, every listed outcome matches
(first run; labels and rules share an author, so this checks consistency, and the labels await human review);
monotonicity (answering an unknown never flips a decided outcome) over all profiles × facts × values; an
all-unknown profile binds nothing (99 depend; top question "Do you offer products or services in the EU?"
decides 80); AI Act Art. 26 moves from Coming to Applies now on 2027-12-02 for an Annex III deployer; NIS2 items
carry the transposition note; UI smoke on all three pages; export tests.

Found live and fixed:
- **The router invented facts.** gpt-oss-20b filled nearly every fact with guesses (e.g. `sensitive_data:
  false`), silently removing articles, and returned partial role lists that replaced known roles. Facts now
  need `evidence` copied from the request (substring-checked, the same rule as basis quotes); headcount maps
  to the size band arithmetically ("40-person" → small, "about 300 staff" → large); list facts merge as a
  union. Remaining limit: a fact read from real words is kept (e.g. "rank job applicants" → automated
  decisions); it is shown under "Noted from your question" and editable on the Checklist page.
- A bare follow-up ("About 300 employees, based in Germany.") was read as vague; the router now gets "the
  previous reply was a checklist" as context. Router accuracy on 36 requests (30 original + 6 follow-ups):
  0.861 and 0.944 on two runs of nearly the same prompt (gpt-oss variance); misses are mostly gap-check
  requests (not built yet), which fall back to answering.
- Profile wording borrowed the high-risk label "not high-risk" for NIS2 sector "none"; labels now come from
  `facts.yaml`.
- AI Act Arts. 43, 47–49, 72, 73 show 2026-08-02 because consolidated Art. 113(c) defers only Chapter III
  Sections 1–3; the checklist reports that faithfully and notes that the requirements they build on apply
  from 2027-12-02 / 2028-08-02.

Live (local API, real Groq): "We are a 40-person company in Germany and we deploy a CV-screening tool…" →
checklist in 1.5 s (29 apply now, 2 later incl. Art. 26 from 2027-12-02, 42 depend); "We are also the
provider…, about 300 staff" → rebuilt in 0.6 s with provider duties (Arts. 9–17 from 2027-12-02) added and
deployer duties kept.
