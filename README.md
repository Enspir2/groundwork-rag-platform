# Groundwork — Grounded RAG Platform with Hybrid Retrieval & Evaluation

> Enterprise-style question answering over documents and structured data, with hybrid
> retrieval, reranking, permission-aware access control, and a versioned evaluation
> framework for comparing retrieval strategies objectively.

**Status:** 🚧 In active development — see [Progress](#progress) below.

## Why this exists

Most RAG demos answer questions by embedding documents and doing vector similarity search,
with no way to tell whether the retrieval is actually any good, no handling of structured
data, and no concept of who's allowed to see what. This project exists to address those
three gaps specifically:

1. **Is retrieval actually good?** — a versioned benchmark dataset and evaluation harness
   (recall, precision, MRR, faithfulness) so retrieval quality is measured, not assumed.
2. **What about numbers, not just documents?** — a controlled, read-only SQL tool so
   questions requiring structured data (e.g. "compare Q1 vs Q2 revenue") aren't limited to
   whatever happens to be written in a PDF.
3. **Who's allowed to see this?** — permission-aware retrieval, enforced at the retrieval
   layer itself, not bolted on as a UI-level filter.

## Architecture

*(diagram added once the pipeline is end-to-end — Phase 4)*

## Tech stack

| Layer | Choice | Why (see docs/decisions.md for full reasoning) |
|---|---|---|
| LLM | Claude API (Anthropic) | Strong instruction-following & grounded citation behavior |
| Embeddings | Voyage AI | Anthropic's recommended embeddings partner; no first-party Claude embeddings API |
| Structured data | SQLite (dev) → PostgreSQL (prod-style) | Start simple, swap in a real driver once the query layer is proven |
| Retrieval | Custom hybrid (BM25 + vector) | Demonstrates the underlying mechanics rather than a framework black box |

## Project structure

```
data/
  docs/               fictional company documents (knowledge base source)
  build_revenue_db.py generates the structured revenue dataset
src/
  chunking.py         document -> chunk splitting logic
eval/                 benchmark dataset + evaluation harness (Phase 3+)
```

## Setup

```bash
git clone <repo-url>
cd groundwork-rag-platform
python3 -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # then fill in your real ANTHROPIC_API_KEY and VOYAGE_API_KEY
python3 data/build_revenue_db.py
```

## Progress

- [x] Knowledge base + structured data source
- [x] Document chunking
- [x] Embeddings + vector search
- [x] Hybrid retrieval (BM25 + vector via Reciprocal Rank Fusion)
- [x] Reranking (Voyage cross-encoder)
- [x] Evaluation framework (recall@5, precision@5, MRR on a 12-question hand-labeled benchmark)
- [x] Query decomposition — evaluated, **not adopted** (see Evaluation Results below)
- [ ] Controlled SQL tool
- [ ] Permission-aware retrieval

## Evaluation results

Measured on a 12-question hand-labeled benchmark (`eval/benchmark.json`), recall@5 / precision@5 / MRR:

| Strategy | Recall@5 | Precision@5 | MRR |
|---|---|---|---|
| Vector-only | 0.944 | 0.217 | 0.806 |
| Hybrid (BM25 + vector, RRF) | 0.903 | 0.200 | 0.703 |
| Hybrid + reranked | 0.944 | 0.217 | **0.833** |
| Hybrid + reranked + query planning | 0.944 | 0.217 | 0.792 |

**Note on precision@5**: most benchmark questions have exactly 1 relevant chunk, so precision@5 is mathematically capped near 0.2 regardless of retrieval quality — recall and MRR carry the real signal here.

**Key findings:**
- **Hybrid retrieval alone underperformed vector-only** on this small (21-chunk) corpus. Diagnosed cause: BM25's IDF statistics are unreliable at this scale, and summing scores across several generic term matches (e.g. "revenue", "Q1", "Q2") can outrank a single highly specific match (e.g. "India") — see `eval/debug_question.py` output for q8.
- **Reranking recovered that regression and improved MRR** (0.806 → 0.833) by re-ordering retrieved chunks based on direct query-document relevance rather than term-overlap arithmetic — it corrected exactly the q8 failure case.
- **Query decomposition was evaluated and rejected.** It was designed to fix a diagnosed multi-hop retrieval gap (q1: "why did European revenue decline" requires evidence from both the finance report and the product recall document, which share no vocabulary). Measured result: no recall improvement on the target case, and a slight MRR regression overall, because the LLM decomposer has no visibility into the actual corpus and produces plausible-sounding but ungrounded sub-questions (e.g. guessing "supply chain issues" rather than the actual cause, a product recall). The code is kept in the repo (`src/query_planning.py`) as a documented negative result rather than wired into the active pipeline. A grounded, evidence-aware planning loop (retrieve first, then decide if more targeted sub-queries are needed) is a more promising direction, noted here as a considered next step rather than implemented given time constraints.
- **The unsolved case**: q1 (the genuine multi-hop question) scored 0.33 recall across every strategy tested — none of vector search, hybrid, reranking, or one-shot decomposition can retrieve evidence that shares no vocabulary or obvious semantic similarity with the query. This is a real, acknowledged limitation of the current architecture.

## What I'd do differently at scale

*(added at the end — a short, honest section on limitations is more credible than pretending
the project is production-ready)*
