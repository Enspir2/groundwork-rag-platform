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
- [ ] Embeddings + vector search
- [ ] Hybrid retrieval (BM25 + vector)
- [ ] Reranking
- [ ] Evaluation framework
- [ ] Controlled SQL tool
- [ ] Query planning
- [ ] Permission-aware retrieval

## Evaluation results

*(added once the eval harness exists — this section is the centerpiece of the project)*

## What I'd do differently at scale

*(added at the end — a short, honest section on limitations is more credible than pretending
the project is production-ready)*
