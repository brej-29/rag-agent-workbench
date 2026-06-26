# Eval Corpus Labeling Worksheet

**Purpose:** Human-label `eval/golden.jsonl` by reading this document, not by running the retriever.
Draft queries are starting points — review, replace, and fill `relevant_doc_ids` by reading the
content excerpts below. Transfer finished entries into `eval/golden.jsonl`.

**Labeling rule (non-negotiable):** Assign `relevant_doc_ids` by reading the excerpts in Section 1
and deciding which documents genuinely answer the query. Do NOT run the retriever and copy its
output — that encodes the embedder's biases and makes recall@k circular.

**Target:** ≥ 20 labeled entries in `eval/golden.jsonl`, each with ≥ 1 `relevant_doc_id` that is an
actual corpus doc_id (not a PLACEHOLDER).

---

## Section 1 — Corpus Content Map

This is the reference for labeling. Each entry shows: the doc_id, the document's title, its source,
and the actual first-chunk text indexed in Pinecone. Read the excerpts to decide relevance.

### Wikipedia documents (8) — STABLE, use these as golden-set anchors

These doc_ids are deterministic SHA256 hashes and will not change on re-ingestion of the same titles.
**Prefer Wikipedia doc_ids when labeling** — arXiv doc_ids may shift if the corpus is re-ingested
on a different date (different recent papers).

---

**DOC-W1**
```
doc_id : eaf004fec892968d5561c536c50eb1b0c12fd5fb08e18607ef9ed0666bb4a975
title  : Retrieval-augmented generation
source : wiki
url    : https://en.wikipedia.org/wiki/Retrieval-augmented_generation
```
> Retrieval-augmented generation (RAG) is a technique that enables large language models (LLMs)
> to retrieve and incorporate new information from external data sources. With RAG, LLMs first
> refer to a specified set of documents, then respond to user queries. These documents supplement
> information from the LLM's pre-existing training data. This allows LLMs to use domain-specific
> and/or updated information that is not available in the training data.

---

**DOC-W2**
```
doc_id : 2b529e6ae899cf68473c9269463a1eeb7ff769ad6ae5d428c9b9c6fb1d9d05b5
title  : Vector database
source : wiki
url    : https://en.wikipedia.org/wiki/Vector_database
```
> A vector database, vector store or vector search engine is a database that stores and retrieves
> embeddings of data in vector space. Vector databases typically implement approximate nearest
> neighbor algorithms so users can search for records semantically similar to a given input,
> unlike traditional databases which primarily look up records by exact match. Use-cases for
> vector databases include similarity search, semantic search, multi-modal search, recommendation
> engines, and object recognition.

---

**DOC-W3**
```
doc_id : 1c168904a9a26aadacbedeb48bb0a7f0cbf661a83958bdb1792d151bcec8c119
title  : Large language model
source : wiki
url    : https://en.wikipedia.org/wiki/Large_language_model
```
> A large language model (LLM) is a neural network trained on a vast amount of text for natural
> language processing tasks, especially language generation. LLMs can typically generate,
> summarize, translate, and analyze text in many contexts, and are a foundational technology
> behind modern chatbots. Biased or inaccurate training data can make an LLM's output less
> reliable.

---

**DOC-W4**
```
doc_id : 0fd1724654d44b871e048f59c382b9b033fb748e2fd6e6c28a6ff4f91e9cc333
title  : Transformer (deep learning architecture)
source : wiki
url    : https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)
```
> In deep learning, the transformer is a family of artificial neural network architectures based
> on the multi-head attention mechanism, in which text is converted to numerical representations
> called tokens, and each token is converted into a vector via lookup from a word embedding table.
> At each layer, each token is then contextualized within the scope of the context window with
> other (unmasked) tokens via a parallel multi-head attention mechanism.

---

**DOC-W5**
```
doc_id : 95af516f7ab30dd9d834fe530d599915070a06ab0d9a54b181848f13790d8fe9
title  : Approximate nearest neighbor search
source : wiki
url    : https://en.wikipedia.org/wiki/Approximate_nearest_neighbor_search
```
> Nearest neighbor search (NNS), as a form of proximity search, is the optimization problem of
> finding the point in a given set that is closest to a given point. Closeness is typically
> expressed in terms of a dissimilarity function: the less similar the objects, the larger the
> function values. [Article covers approximate variants, trade-offs between recall and speed,
> and algorithms such as HNSW and IVF.]

---

**DOC-W6**
```
doc_id : 2e087b3b53d1df66054abd4804242600f6d16bde1347817c857d1758dd948bb2
title  : Information retrieval
source : wiki
url    : https://en.wikipedia.org/wiki/Information_retrieval
```
> Information retrieval (IR) in computing and information science is the task of identifying and
> retrieving information system resources that are relevant to an information need. The information
> need can be specified in the form of a search query. In the case of document retrieval, queries
> can be based on full-text or other content-based indexing. Information retrieval is the science
> of searching for information in a document, searching for documents themselves, and also
> searching for metadata that describes data.

---

**DOC-W7**
```
doc_id : fa08a772f25ab55d7dc0dc6722a0322e82481ed33098a049e1faba627817567e
title  : Semantic search
source : wiki
url    : https://en.wikipedia.org/wiki/Semantic_search
```
> Semantic search denotes search with meaning, as distinguished from lexical search where the
> search engine looks for literal matches of the query words or variants of them, without
> understanding the overall meaning of the query. Semantic search is an approach to information
> retrieval that seeks to improve search accuracy by understanding the searcher's intent and the
> contextual meaning of terms as they appear in the searchable dataspace.

---

**DOC-W8**
```
doc_id : 181ab66a9bc68c0460eca4c35cc4199b7395ac3f30132b4b084e7cf18cfa716e
title  : Question answering
source : wiki
url    : https://en.wikipedia.org/wiki/Question_answering
```
> Question answering (QA) is a computer science discipline within the fields of information
> retrieval and natural language processing (NLP) that is concerned with building systems that
> automatically answer questions that are posed by humans in a natural language.

---

### arXiv documents (15) — NON-STABLE

> ⚠️ arXiv doc_ids are only stable if the exact titles remain in `eval/corpus.py`. If the corpus
> is re-ingested on a different date, arXiv queries may return different papers. You may safely
> include these doc_ids in your golden set, but also commit the title to `eval/corpus.py`
> `ARXIV_QUERIES` or note the dependency here.

> ⚠️ **DOC-A12 is an off-topic physics paper** (see below). Do not assign it as relevant to
> any IR/RAG query — it is in the index only because "nearest neighbor" appears in its title in
> a physics context. Good queries should NOT retrieve it; it is a useful noise document for
> testing precision.

---

**DOC-A1** _(RAG query batch)_
```
doc_id : 078af71a5bee07ec816c45c139cfe1678dd2d3b4c915e26200d5d9abbbfa5945
title  : AR-RAG: Autoregressive Retrieval Augmentation for Image Generation
source : arxiv
```
> We introduce Autoregressive Retrieval Augmentation (AR-RAG), a novel paradigm that enhances
> image generation by autoregressively incorporating k-nearest neighbor retrievals at the patch
> level. Unlike prior methods that perform a single, static retrieval before generation, AR-RAG
> performs context-aware retrievals at each generation step.

---

**DOC-A2** _(RAG query batch)_
```
doc_id : fce7c6eac0e5cd800f1555202a67f9b4893572acacff7b1e0e1a0d7538a283d7
title  : Intelligent Interaction Strategies for Context-Aware Cognitive Augmentation
source : arxiv
```
> Human cognition is constrained by processing limitations, leading to cognitive overload. Large
> Language Models present an opportunity for cognitive augmentation, but their current reactive
> nature limits their real-world applicability. Explores context-aware LLM augmentation where
> LLMs dynamically adapt to user needs.

---

**DOC-A3** _(RAG query batch)_
```
doc_id : d1f5554128b4918654e6e79d6e627a0a353d0acc8571e486cbf3cbee48fabfa3
title  : Factually: Exploring Wearable Fact-Checking for Augmented Truth Discernment
source : arxiv
```
> Proposes a voice-based, interactive learning companion designed to amplify cognitive abilities
> through informal learning. Enables users to discover new knowledge through contextual
> interactive quizzes, fostering critical thinking. [Wearable AI / augmented cognition focus,
> not core RAG/IR.]

---

**DOC-A4** _(RAG query batch)_
```
doc_id : c0ef3c6101a07de94871d33d2b94497fe384c8f975d89f9037962f4cde0d59a8
title  : Designing AI Systems that Augment Human Performed vs. Demonstrated Critical Thinking
source : arxiv
```
> Examines the impact of LLM-based AI systems on human cognitive abilities, especially critical
> thinking. Explores the distinction between AI that performs thinking for the user vs. AI that
> demonstrates or scaffolds thinking. [Cognitive science / HCI focus, not core RAG/IR.]

---

**DOC-A5** _(RAG query batch)_
```
doc_id : 6abde0ae4237fe9dcaed71d53787be4af88d249eb54c65c69c964d6968813d04
title  : Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation
source : arxiv
```
> Presents and compares multiple approaches to automate the generation of literature reviews
> using NLP techniques and retrieval-augmented generation (RAG) with a Large Language Model.
> The ever-increasing number of research articles provides a challenge for manual review.

---

**DOC-A6** _(Dense retrieval query batch)_
```
doc_id : 82f543cc6b407e91bcabab2b4618838f5c461ecc8f4067d4a8ec6ab79bf1d530
title  : Privacy-Preserving Important Passage Retrieval
source : arxiv
```
> Presents a privacy-preserving method for passage retrieval that relies on creating secure
> representations of documents using Secure Binary Embeddings, allowing third parties to retrieve
> important passages without learning anything about document content.

---

**DOC-A7** _(Dense retrieval query batch)_
```
doc_id : 7e776e26bb476eeda54c523511af848518aca2ed8c4970a75fc9b1c3b0ea5c45
title  : On Single and Multiple Representations in Dense Passage Retrieval
source : arxiv
```
> Reviews two dense retrieval families: single-representation (entire passages encoded as one
> vector) vs. multiple-representation (late interaction like ColBERT). Covers gains from
> contextualised language models in search effectiveness.

---

**DOC-A8** _(Dense retrieval query batch)_
```
doc_id : 1ccde846e825ac6a4016233e685845521b721c427fca21c940447cb0d5c01fd7
title  : Retrieval Oriented Masking Pre-training Language Model for Dense Passage Retrieval
source : arxiv
```
> Shows that conventional random masking in MLM pre-training selects tokens with limited effect
> on passage retrieval (e.g., stop-words). Proposes a retrieval-oriented masking strategy to
> improve dense retrieval pre-training.

---

**DOC-A9** _(Dense retrieval query batch)_
```
doc_id : 55d1d1e4d13fcd7e903fe3e4315ebf9f40162c9848f5f8e39634f685517e2c1d
title  : Dense Passage Retrieval for Open-Domain Question Answering
source : arxiv
```
> Shows that retrieval can be practically implemented using dense representations alone, learned
> from a small number of questions and passages by a simple dual-encoder framework. Demonstrates
> that dense retrieval outperforms BM25 for open-domain QA.

---

**DOC-A10** _(Dense retrieval query batch)_
```
doc_id : a0643195a3218ae4b105bdff96c6dd3f9d10b7b534b75dec27595ba7163eed8e
title  : Improving Passage Retrieval with Zero-Shot Question Generation
source : arxiv
```
> Proposes a re-ranking method for passage retrieval that re-scores passages with a zero-shot
> question generation model — a pre-trained language model computes the probability of the input
> question conditioned on a retrieved passage. Applied on top of any retrieval method.

---

**DOC-A11** _(ANN query batch)_
```
doc_id : 457202a3c6ab06efd4d062daae6c351154d058dc72f85042df3d84617841fec4
title  : I/O Optimizations for Graph-Based Disk-Resident Approximate Nearest Neighbor Search: A Design Space Exploration
source : arxiv
```
> ANN search on SSD-backed indexes is increasingly I/O-bound (I/O accounts for 70–90% of query
> latency). Presents an I/O-first framework for disk-based ANN that organizes techniques along
> memory layout, disk layout, and search algorithm dimensions.

---

**DOC-A12** ⚠️ OFF-TOPIC — DO NOT USE AS RELEVANT
```
doc_id : 70d250f973bfec8a5bad1d1be628d2096e4bf59e84cf585bbb423b0121167ef7
title  : Magnon excitations in Cs2CuAl4O8 - a bond alternating S=1/2 spin chain with next nearest neighbor coupling
source : arxiv
```
> A physics paper about spin-chain quantum systems captured by the arXiv query because "nearest
> neighbor" appears in a physics context. This document is NOT relevant to any IR/RAG/NLP query.
> It is a useful noise document — correct retrieval should NOT surface it for any corpus query.
> Do not assign this doc_id as relevant to any entry.

---

**DOC-A13** _(ANN query batch)_
```
doc_id : ce46ed6fc0d9bc53a250213926400524cbe21352edc099e7f11de756d19fc509
title  : Subspace Approximation for Approximate Nearest Neighbor Search in NLP
source : arxiv
```
> Most NLP tasks can be formulated as approximate nearest neighbor search (e.g., word analogy,
> document similarity, machine translation, question answering). Proposes subspace approximation
> methods tailored for NLP embedding spaces.

---

**DOC-A14** _(ANN query batch)_
```
doc_id : 1dffe01c4ff11522924153faf36c550e94785d88aed7df8fa93d572cb9dd6241
title  : Approximate Nearest Neighbor Search with Window Filters
source : arxiv
```
> Defines and investigates c-approximate window search: ANN search where each point has a
> numeric label and queries target arbitrary label ranges. Applies to semantic search with
> metadata filters (e.g., image search with timestamp, product search with cost filters).

---

**DOC-A15** _(ANN query batch)_
```
doc_id : 9e5909547c859b138204dd201cc7e6b8f9524d3b3af6a5c0b5908a143466c56d
title  : Learning Cluster Representatives for Approximate Nearest Neighbor Search
source : arxiv
```
> A primary approach to ANN search is clustering: partitioning the dataset into groups
> characterized by representative data points. Retrieving top-k requires identifying the most
> relevant cluster representatives. This paper improves how cluster representatives are learned.

---

## Section 2 — DRAFT Query Table

> **⚠️ THESE ARE DRAFTS — review, edit, and replace as needed.**
> The query text is a starting point; the human must read Section 1 and decide relevance.
> `relevant_doc_ids` is LEFT EMPTY — fill it yourself from the doc_ids above.
> Transfer finished entries into `eval/golden.jsonl` using the schema in that file.
> Target: ≥ 20 finished entries covering a mix of easy and hard difficulties.

Use the short labels (DOC-W1…W8, DOC-A1…A15) from Section 1 to plan your labels,
then copy the full SHA256 doc_ids when writing `eval/golden.jsonl`.

**Difficulty guide:**
- **easy** — one document is obviously the primary source; expect recall@1 ≈ 1.0
- **medium** — 2–3 documents share relevance; tests whether the retriever finds all
- **hard** — relevance is spread across document types or requires cross-doc reasoning;
  expect recall@k to be sensitive to k

---

| # | DRAFT query (EDIT/REPLACE) | Difficulty guess | Notes for labeler | relevant_doc_ids (YOU FILL) |
|---|---------------------------|-----------------|-------------------|----------------------------|
| 1 | What is retrieval-augmented generation and how does it work? | easy | Primarily DOC-W1; possibly DOC-A5 if article mentions RAG mechanics | eaf004fec892968d5561c536c50eb1b0c12fd5fb08e18607ef9ed0666bb4a975; 6abde0ae4237fe9dcaed71d53787be4af88d249eb54c65c69c964d6968813d04;  |
| 2 | What is a vector database and what is it used for in AI? | easy | Primarily DOC-W2 | 2b529e6ae899cf68473c9269463a1eeb7ff769ad6ae5d428c9b9c6fb1d9d05b5; |
| 3 | What is a large language model? | easy | Primarily DOC-W3 | 1c168904a9a26aadacbedeb48bb0a7f0cbf661a83958bdb1792d151bcec8c119; |
| 4 | How does the transformer self-attention mechanism work? | easy | Primarily DOC-W4 | 0fd1724654d44b871e048f59c382b9b033fb748e2fd6e6c28a6ff4f91e9cc333; |
| 5 | What is approximate nearest neighbor search? | easy | Primarily DOC-W5; possibly DOC-A11/A14/A15 | 95af516f7ab30dd9d834fe530d599915070a06ab0d9a54b181848f13790d8fe9; 457202a3c6ab06efd4d062daae6c351154d058dc72f85042df3d84617841fec4; |
| 6 | What is information retrieval? | easy | Primarily DOC-W6 | 2e087b3b53d1df66054abd4804242600f6d16bde1347817c857d1758dd948bb2; |
| 7 | What is semantic search and how does it differ from keyword search? | easy | Primarily DOC-W7; possibly DOC-W6 | 2e087b3b53d1df66054abd4804242600f6d16bde1347817c857d1758dd948bb2; fa08a772f25ab55d7dc0dc6722a0322e82481ed33098a049e1faba627817567e; |
| 8 | What is question answering in the context of NLP? | easy | Primarily DOC-W8 | 181ab66a9bc68c0460eca4c35cc4199b7395ac3f30132b4b084e7cf18cfa716e; |
| 9 | What is dense passage retrieval for open-domain QA? | easy | Primarily DOC-A9; possibly DOC-A7/A8 | 7e776e26bb476eeda54c523511af848518aca2ed8c4970a75fc9b1c3b0ea5c45; 55d1d1e4d13fcd7e903fe3e4315ebf9f40162c9848f5f8e39634f685517e2c1d; |
| 10 | How does zero-shot question generation improve passage retrieval? | easy | Primarily DOC-A10 | a0643195a3218ae4b105bdff96c6dd3f9d10b7b534b75dec27595ba7163eed8e; |
| 11 | How do vector databases support RAG pipelines? | medium | DOC-W1 + DOC-W2 likely both relevant | eaf004fec892968d5561c536c50eb1b0c12fd5fb08e18607ef9ed0666bb4a975; 2b529e6ae899cf68473c9269463a1eeb7ff769ad6ae5d428c9b9c6fb1d9d05b5;  |
| 12 | How do large language models use retrieval to access external knowledge? | medium | DOC-W1 + DOC-W3; possibly DOC-A5 | eaf004fec892968d5561c536c50eb1b0c12fd5fb08e18607ef9ed0666bb4a975; 1c168904a9a26aadacbedeb48bb0a7f0cbf661a83958bdb1792d151bcec8c119; 6abde0ae4237fe9dcaed71d53787be4af88d249eb54c65c69c964d6968813d04;  |
| 13 | What are the trade-offs between exact and approximate nearest neighbor search? | medium | DOC-W5 + possibly DOC-A11, DOC-A15 | 95af516f7ab30dd9d834fe530d599915070a06ab0d9a54b181848f13790d8fe9; 457202a3c6ab06efd4d062daae6c351154d058dc72f85042df3d84617841fec4; 9e5909547c859b138204dd201cc7e6b8f9524d3b3af6a5c0b5908a143466c56d; |
| 14 | How do embedding vectors enable semantic similarity search? | medium | DOC-W2 + DOC-W7; possibly DOC-W5 | 2b529e6ae899cf68473c9269463a1eeb7ff769ad6ae5d428c9b9c6fb1d9d05b5; 95af516f7ab30dd9d834fe530d599915070a06ab0d9a54b181848f13790d8fe9; fa08a772f25ab55d7dc0dc6722a0322e82481ed33098a049e1faba627817567e; |
| 15 | What are the challenges and solutions for open-domain question answering? | medium | DOC-W8 + DOC-A9; possibly DOC-A10 | 181ab66a9bc68c0460eca4c35cc4199b7395ac3f30132b4b084e7cf18cfa716e; 55d1d1e4d13fcd7e903fe3e4315ebf9f40162c9848f5f8e39634f685517e2c1d; a0643195a3218ae4b105bdff96c6dd3f9d10b7b534b75dec27595ba7163eed8e; |
| 16 | How does semantic search improve over traditional information retrieval? | medium | DOC-W7 + DOC-W6 | 2e087b3b53d1df66054abd4804242600f6d16bde1347817c857d1758dd948bb2; fa08a772f25ab55d7dc0dc6722a0322e82481ed33098a049e1faba627817567e; |
| 17 | What pre-training techniques improve dense retrieval models? | medium | DOC-A8 + DOC-A7 + DOC-A9 | 7e776e26bb476eeda54c523511af848518aca2ed8c4970a75fc9b1c3b0ea5c45; 1ccde846e825ac6a4016233e685845521b721c427fca21c940447cb0d5c01fd7; 55d1d1e4d13fcd7e903fe3e4315ebf9f40162c9848f5f8e39634f685517e2c1d; |
| 18 | What are single-representation versus multi-representation dense retrieval models? | medium | DOC-A7; possibly DOC-A9 | 7e776e26bb476eeda54c523511af848518aca2ed8c4970a75fc9b1c3b0ea5c45; 55d1d1e4d13fcd7e903fe3e4315ebf9f40162c9848f5f8e39634f685517e2c1d; |
| 19 | How can retrieval-augmented generation help automate literature review? | medium | DOC-A5 + DOC-W1 | eaf004fec892968d5561c536c50eb1b0c12fd5fb08e18607ef9ed0666bb4a975; 6abde0ae4237fe9dcaed71d53787be4af88d249eb54c65c69c964d6968813d04;  |
| 20 | What graph-based algorithms are used for disk-resident ANN search? | medium | DOC-A11 + DOC-W5 | 95af516f7ab30dd9d834fe530d599915070a06ab0d9a54b181848f13790d8fe9; 457202a3c6ab06efd4d062daae6c351154d058dc72f85042df3d84617841fec4; |
| 21 | How does clustering improve approximate nearest neighbor search efficiency? | hard | DOC-A15 + DOC-W5; possibly DOC-A11 | 95af516f7ab30dd9d834fe530d599915070a06ab0d9a54b181848f13790d8fe9; 457202a3c6ab06efd4d062daae6c351154d058dc72f85042df3d84617841fec4; |
| 22 | What NLP tasks can be formulated as nearest neighbor search problems? | hard | DOC-A13 + DOC-W5; possibly DOC-W7 | 95af516f7ab30dd9d834fe530d599915070a06ab0d9a54b181848f13790d8fe9; fa08a772f25ab55d7dc0dc6722a0322e82481ed33098a049e1faba627817567e; ce46ed6fc0d9bc53a250213926400524cbe21352edc099e7f11de756d19fc509; |
| 23 | How do language model representations differ from sparse (BM25/TF-IDF) retrieval? | hard | DOC-A9 + DOC-A7 + DOC-W6 | 2e087b3b53d1df66054abd4804242600f6d16bde1347817c857d1758dd948bb2; 7e776e26bb476eeda54c523511af848518aca2ed8c4970a75fc9b1c3b0ea5c45; 55d1d1e4d13fcd7e903fe3e4315ebf9f40162c9848f5f8e39634f685517e2c1d; |
| 24 | What are the privacy risks and mitigations in passage retrieval systems? | hard | Primarily DOC-A6 | 82f543cc6b407e91bcabab2b4618838f5c461ecc8f4067d4a8ec6ab79bf1d530; |
| 25 | How can AI systems augment human cognitive abilities without replacing critical thinking? | hard | DOC-A2 + DOC-A4; possibly DOC-A3 | fce7c6eac0e5cd800f1555202a67f9b4893572acacff7b1e0e1a0d7538a283d7; d1f5554128b4918654e6e79d6e627a0a353d0acc8571e486cbf3cbee48fabfa3; c0ef3c6101a07de94871d33d2b94497fe384c8f975d89f9037962f4cde0d59a8; 82f543cc6b407e91bcabab2b4618838f5c461ecc8f4067d4a8ec6ab79bf1d530;  |
| 26 | What are the I/O bottlenecks in approximate nearest neighbor search on SSDs? | hard | Primarily DOC-A11 | 457202a3c6ab06efd4d062daae6c351154d058dc72f85042df3d84617841fec4; |
| 27 | How does masked language model pre-training relate to retrieval quality? | hard | DOC-A8; possibly DOC-A9 | 1ccde846e825ac6a4016233e685845521b721c427fca21c940447cb0d5c01fd7; 55d1d1e4d13fcd7e903fe3e4315ebf9f40162c9848f5f8e39634f685517e2c1d; |
| 28 | What is ANN search with metadata/attribute filters (window search)? | hard | Primarily DOC-A14 | 1dffe01c4ff11522924153faf36c550e94785d88aed7df8fa93d572cb9dd6241; |
| 29 | What are the key components of a transformer-based language model architecture? | hard | DOC-W4 + DOC-W3 | 1c168904a9a26aadacbedeb48bb0a7f0cbf661a83958bdb1792d151bcec8c119; 0fd1724654d44b871e048f59c382b9b033fb748e2fd6e6c28a6ff4f91e9cc333; |
| 30 | How do RAG systems combine retrieval and generation to reduce hallucination? | hard | DOC-W1 + DOC-W3; possibly DOC-A5 | eaf004fec892968d5561c536c50eb1b0c12fd5fb08e18607ef9ed0666bb4a975; 1c168904a9a26aadacbedeb48bb0a7f0cbf661a83958bdb1792d151bcec8c119; 6abde0ae4237fe9dcaed71d53787be4af88d249eb54c65c69c964d6968813d04; |

---

## Section 3 — Labeling Instructions

1. **Read Section 1** for each query you want to label. Decide which documents genuinely answer
   the question based on their content excerpts (and the full Wikipedia article / arXiv abstract
   if needed). The excerpt is the first ~500 characters of the first indexed chunk — the full
   document contains more content.

2. **Fill in `relevant_doc_ids`** using the full SHA256 strings from Section 1 (not the DOC-Wn
   short labels). A document is relevant if a reasonable reader would expect a RAG system to
   surface it for the query. Include all genuinely relevant docs, not just the most obvious one.

3. **Edit or replace draft queries** freely. The 30 drafts are starting points. Replace any that
   feel unnatural or too narrow. Ensure you have a mix of difficulty levels.

4. **Transfer to `eval/golden.jsonl`** using this schema:
   ```json
   {"query": "...", "relevant_doc_ids": ["sha256_1", "sha256_2"], "_note": "optional"}
   ```
   The `_note` field is optional — use it to record why you chose those doc_ids.

5. **Do NOT** include DOC-A12 (`70d250...`) as relevant to any query — it is an off-topic
   physics paper and should never be surfaced for any IR/RAG/NLP question.

6. **Minimum:** 20 entries with ≥ 1 relevant_doc_id each. Recommend 25–30 for a robust eval.

7. **Tell Claude Code** when `eval/golden.jsonl` is populated with ≥ 20 real entries and it will
   validate the labels and then run `make eval` + `make eval-ab`.
