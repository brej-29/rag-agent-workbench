# Makefile — RAG Agent Workbench
#
# Evaluation targets require the backend Python environment to be active
# (pip install -r backend/requirements.txt).
# Frontend targets require the root requirements (pip install -r requirements.txt).
#
# Override defaults:  make eval PYTHON=python3.11 EVAL_NS=staging TOP_K=15

PYTHON       ?= python
EVAL_NS      ?= eval
TOP_K        ?= 10
GOLDEN       ?= eval/golden.jsonl
MAILTO       ?=
CANDIDATES   ?= 20
RERANK_MODEL ?= bge-reranker-v2-m3

.PHONY: eval-corpus eval eval-ab eval-ab-topk eval-sweep corpus-manifest corpus-verify test help

## help: Print this help message.
help:
	@grep -E '^## ' Makefile | sed 's/^## //'

## eval-corpus: Ingest the fixed mini-corpus into the eval Pinecone namespace.
##   Issues Pinecone UPSERT calls — run ONCE, then label eval/golden.jsonl.
##   Prints doc_ids for all ingested documents so you can fill in the golden set.
##   Pass MAILTO=you@example.com to also ingest from OpenAlex.
eval-corpus:
	@echo ">>> Ingesting eval corpus into namespace='$(EVAL_NS)'..."
	$(PYTHON) eval/setup_corpus.py --namespace $(EVAL_NS) $(if $(MAILTO),--mailto $(MAILTO),)

## eval: Run retrieval evaluation (read-only Pinecone queries, no LLM calls).
##   Computes recall@k, MRR, nDCG@k for each golden entry.
##   Report written to eval/reports/ as JSON + markdown.
##   Prerequisite: run `make eval-corpus` first and fill in eval/golden.jsonl.
eval:
	@echo ">>> Running retrieval eval (namespace='$(EVAL_NS)', top_k=$(TOP_K))..."
	$(PYTHON) eval/run.py --namespace $(EVAL_NS) --top-k $(TOP_K) --golden $(GOLDEN)

## eval-ab: Run baseline-vs-rerank A/B eval (ON-DEMAND — NOT in CI, NOT invoked by make eval).
##   Issues LIVE Pinecone query + rerank calls (read-only, no upserts) — incurs inference cost.
##   Produces a delta table (recall@k, MRR, nDCG@k, latency) in eval/reports/ab_*.
##   Prerequisite: golden set populated with non-PLACEHOLDER relevant_doc_ids.
##   Override pool size:  make eval-ab CANDIDATES=30
##   Override model:      make eval-ab RERANK_MODEL=pinecone-rerank-v0
eval-ab:
	@echo ">>> Running A/B eval (namespace='$(EVAL_NS)', top_k=$(TOP_K), candidates=$(CANDIDATES))..."
	$(PYTHON) eval/run_ab.py \
		--namespace $(EVAL_NS) \
		--top-k $(TOP_K) \
		--candidates $(CANDIDATES) \
		--golden $(GOLDEN) \
		--rerank-model $(RERANK_MODEL)

## eval-ab-topk: Run A/B eval with top-heavy precision/nDCG metrics at multiple k values.
##   Computes precision@1, recall@3/5, nDCG@3/5 plus the standard @top_k metrics.
##   Same golden set and corpus as eval-ab — only the reported metrics differ.
##   ON-DEMAND only — NOT in CI, NOT invoked by make eval.
eval-ab-topk:
	@echo ">>> Running top-heavy A/B eval (namespace='$(EVAL_NS)', top_k=$(TOP_K), candidates=$(CANDIDATES))..."
	$(PYTHON) eval/run_ab.py \
		--namespace $(EVAL_NS) \
		--top-k $(TOP_K) \
		--candidates $(CANDIDATES) \
		--golden $(GOLDEN) \
		--rerank-model $(RERANK_MODEL) \
		--multi-k

## eval-sweep: Run top-k sweep to find the minimum context-preserving k value.
##   Retrieves at chunk_fetch_k=20 ONCE per query (dense-only, rerank OFF),
##   then computes recall@k, nDCG@k, precision@k at k = 1, 2, 3, 5, 8, 10.
##   Reports the knee (smallest k within 0.02 of the k=10 quality ceiling)
##   and a cosine-floor safety bound.
##   ON-DEMAND only -- NOT in CI, NOT invoked by make eval.
eval-sweep:
	@echo ">>> Running top-k sweep eval (namespace='$(EVAL_NS)', golden=$(GOLDEN))..."
	$(PYTHON) eval/run_sweep.py \
		--namespace $(EVAL_NS) \
		--golden $(GOLDEN)

## corpus-manifest: Snapshot the live eval-namespace index into eval/corpus_manifest.json.
##   READ-ONLY — no upserts or deletes.  ON-DEMAND only (not in CI).
##   Run this once after `make eval-corpus` and re-run whenever the corpus changes.
##   Override namespace:  make corpus-manifest EVAL_NS=staging
corpus-manifest:
	@echo ">>> Snapshotting eval corpus manifest (namespace='$(EVAL_NS)')..."
	$(PYTHON) eval/corpus_manifest.py generate --namespace $(EVAL_NS)

## corpus-verify: Compare the committed manifest to the live index and report drift.
##   READ-ONLY — no upserts or deletes.  ON-DEMAND only (not in CI).
##   Exit code 0 means no drift; exit code 1 means missing/extra doc_ids found.
##   Override namespace:  make corpus-verify EVAL_NS=staging
corpus-verify:
	@echo ">>> Verifying eval corpus manifest (namespace='$(EVAL_NS)')..."
	$(PYTHON) eval/corpus_manifest.py validate --namespace $(EVAL_NS)

## test: Run CI-safe unit tests (zero network calls).
test:
	pytest tests/ -v
