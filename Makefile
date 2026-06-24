# Makefile — RAG Agent Workbench
#
# Evaluation targets require the backend Python environment to be active
# (pip install -r backend/requirements.txt).
# Frontend targets require the root requirements (pip install -r requirements.txt).
#
# Override defaults:  make eval PYTHON=python3.11 EVAL_NS=staging TOP_K=15

PYTHON    ?= python
EVAL_NS   ?= eval
TOP_K     ?= 10
GOLDEN    ?= eval/golden.jsonl
MAILTO    ?=

.PHONY: eval-corpus eval test help

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

## test: Run CI-safe unit tests (zero network calls).
test:
	pytest tests/test_eval_metrics.py -v
