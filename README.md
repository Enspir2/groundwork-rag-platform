# Groundwork — Grounded RAG Platform with Hybrid Retrieval & Evaluation

> Enterprise-style question answering over documents and structured data, with hybrid
> retrieval, reranking, permission-aware access control, query routing across SQL and
> documents, and a hand-labeled evaluation framework for comparing retrieval strategies
> objectively rather than assuming one approach is better than another.

**Status:** Core pipeline complete and demoable end-to-end (see [Demo](#demo)).

## Why this exists

Most RAG demos answer questions by embedding documents and doing vector similarity search,
with no way to tell whether the retrieval is actually any good, no handling of structured
data, and no concept of who's allowed to see what. This project addresses those three gaps
specifically:

1. **Is retrieval actually good?** — a hand-labeled benchmark dataset and evaluation harness
   (recall@5, precision@5, MRR) so retrieval quality is measured across four strategies,
   not assumed. Includes a documented case where an added technique was measured and
   **rejected** based on evidence, not just cases where things worked.
2. **What about numbers, not just documents?** — a controlled, read-only SQL tool so
   questions requiring structured data (e.g. "compare Q1 vs Q2 revenue") aren't limited to
   whatever happens to be restated in prose, plus a router that combines SQL and document
   evidence into a single answer when a question needs both.
3. **Who's allowed to see this?** — permission-aware retrieval, enforced by filtering the
   candidate set *before* ranking happens, not by hiding results after the fact.

## Architecture

```
                        ┌─────────────────────┐
   Question + Role ───► │   Query Router       │
                        │  (Gemini classifies:  │
                        │  SQL / docs / both)   │
                        └──────────┬───────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                          ▼
   ┌─────────────────────┐                  ┌─────────────────────────┐
   │  Controlled SQL Tool  │                  │  Document Retrieval      │
   │  NL → SQL (Gemini)    │                  │  1. Permission filter    │
   │  → validated          │                  │     (role-based, BEFORE  │
   │  → read-only execute  │                  │     ranking)             │
   └──────────┬───────────┘                  │  2. Hybrid search         │
              │                               │     (BM25 + vector, RRF) │
              │                               │  3. Rerank (cross-       │
              │                               │     encoder)             │
              │                               └───────────┬─────────────┘
              │                                            │
              └───────────────────┬────────────────────────┘
                                   ▼
                     ┌─────────────────────────┐
                     │  Grounded Synthesis       │
                     │  (Claude, citation-        │
                     │   enforced, refuses to     │
                     │   guess beyond evidence)   │
                     └──────────────┬────────────┘
                                    ▼
                         Cited answer + sources
                         (shown in Streamlit UI)
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Final answer synthesis | Claude API (Anthropic) | Strong instruction-following & grounded citation behavior — used only for the highest-value step |
| Query routing / SQL generation / decomposition | Gemini (free tier) | Narrow, structured classification/generation tasks don't need frontier-model reasoning — heterogeneous model routing by cost/complexity |
| Embeddings | Voyage AI | Anthropic's recommended embeddings partner; Claude has no first-party embeddings API |
| Reranking | Voyage rerank (cross-encoder) | Corrects term-overlap failures that BM25 and bi-encoder embeddings both miss |
| Structured data | SQLite | Simple, zero-setup for project scope; a production system would use Postgres with a least-privilege read-only role instead of a read-only connection flag |
| Retrieval | Custom hybrid (BM25 + vector via Reciprocal Rank Fusion) | Built from scratch rather than a framework, to demonstrate and evaluate the underlying mechanics directly |
| UI | Streamlit | Fast to build for a demo; not intended as a production frontend |

## Project structure

```
data/
  docs/                        fictional company documents (knowledge base source)
  build_revenue_db.py          generates the structured revenue dataset (SQLite)
src/
  chunking.py                  document -> chunk splitting + role-based permission tagging/filtering
  embeddings.py                Voyage embeddings with caching + rate-limit-aware retry
  hybrid_retrieval.py          BM25 + vector search, combined via Reciprocal Rank Fusion
  reranking.py                 Voyage cross-encoder reranking, cached
  query_planning.py            query decomposition (Gemini) — evaluated, NOT used in the active pipeline (see Evaluation Results)
  sql_tool.py                  NL-to-SQL generation, validation, read-only execution
  synthesis.py                 grounded answer synthesis with Claude, citation-enforced
  router.py                    ties everything together: classify -> gather evidence -> synthesize
  app.py                       Streamlit demo UI
  test_permissions.py          verifies role-based filtering doesn't leak restricted content
  list_models.py               debug utility for checking available Gemini models
eval/
  benchmark.json                12-question hand-labeled benchmark with gold chunk IDs
  evaluate.py                   computes recall@5, precision@5, MRR across 4 strategies
  debug_question.py             per-question ranking diagnostics (BM25 vs vector vs hybrid)
  results.json                  saved evaluation output
```

## Setup

```bash
git clone <repo-url>
cd groundwork-rag-platform
python -m venv venv && source venv/Scripts/activate   # or venv/bin/activate on Mac/Linux
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY, VOYAGE_API_KEY, GEMINI_API_KEY
python data/build_revenue_db.py
```

## Demo

Run the full interactive demo:
```bash
cd src
streamlit run app.py
```
Pick a role (`all` / `hr` / `finance` / `manager`), ask a question, and see the routed,
cited answer along with which sources (SQL and/or documents) were used. Try asking an
HR-restricted question while set to the `finance` role to see permission filtering in action.

*(Screenshot: add one here after running the demo — a visible working result is worth
more than a feature list to anyone reviewing this repo.)*

## Evaluation results

Measured on a 12-question hand-labeled benchmark (`eval/benchmark.json`), recall@5 / precision@5 / MRR:

| Strategy | Recall@5 | Precision@5 | MRR |
|---|---|---|---|
| Vector-only | 0.944 | 0.217 | 0.806 |
| Hybrid (BM25 + vector, RRF) | 0.903 | 0.200 | 0.703 |
| Hybrid + reranked | 0.944 | 0.217 | **0.833** |
| Hybrid + reranked + query planning | 0.944 | 0.217 | 0.792 |

**Note on precision@5**: most benchmark questions have exactly 1 relevant chunk, so
precision@5 is mathematically capped near 0.2 regardless of retrieval quality — recall
and MRR carry the real signal here.

### Key findings

- **Hybrid retrieval alone underperformed vector-only** on this small (21-chunk) corpus.
  Diagnosed cause: BM25's IDF statistics are unreliable at this scale, and summing scores
  across several generic term matches (e.g. "revenue", "Q1", "Q2") can outrank a single
  highly specific match (e.g. "India") — see `eval/debug_question.py` output for q8.
- **Reranking recovered that regression and improved MRR** (0.806 → 0.833) by re-ordering
  retrieved chunks based on direct query-document relevance rather than term-overlap
  arithmetic — it corrected exactly the q8 failure case.
- **Query decomposition was evaluated and rejected.** It was designed to fix a diagnosed
  multi-hop retrieval gap (q1: "why did European revenue decline" requires evidence from
  both the finance report and the product recall document, which share no vocabulary).
  Measured result: no recall improvement on the target case, and a slight MRR regression
  overall, because the LLM decomposer has no visibility into the actual corpus and
  produces plausible-sounding but ungrounded sub-questions (e.g. guessing "supply chain
  issues" rather than the actual cause, a product recall). The code is kept in the repo
  (`src/query_planning.py`) as a documented negative result rather than wired into the
  active pipeline. A grounded, evidence-aware planning loop (retrieve first, then decide
  if more targeted sub-queries are needed based on what's actually missing) is a more
  promising direction, noted here as a considered next step rather than implemented given
  time constraints.
- **The unsolved case**: q1 scored 0.33 recall across every retrieval strategy tested —
  none of vector search, hybrid, reranking, or one-shot decomposition can retrieve
  evidence that shares no vocabulary or obvious semantic similarity with the query. This
  is a real, acknowledged limitation of the current architecture.
- **Retrieval recall and end-to-end answer quality are not the same thing.** Asking the
  vague q1 phrasing produced a complete-*sounding* cited answer built almost entirely from
  one summary chunk that happens to condense all three causes in prose — not from the
  deeper "gold" evidence the benchmark labeled as necessary. Asking a more specific version
  of the same underlying question ("what was the financial cost of the recall and how many
  units were affected") correctly retrieved the actual detailed source chunks and cited the
  real figures ($1.8M, 14,000 units) accurately. This shows retrieval quality depends
  heavily on query specificity and corpus redundancy, not just the retrieval algorithm —
  and that recall@k against a fixed gold set can understate real-world answer quality when
  the corpus contains overlapping/summarized information. A more complete evaluation would
  separately measure answer-level correctness (e.g. LLM-graded faithfulness) rather than
  relying on chunk recall alone.
- **No hallucination observed in manual review**: synthesis is instructed to cite every
  claim and avoid guessing beyond retrieved evidence. When the specific recall document
  wasn't retrieved, the model correctly used the vaguer term available in its actual
  evidence ("manufacturing defect") rather than inventing the more specific term ("wiring
  defect") that only existed in an unretrieved source — a concrete, verifiable example of
  grounding behaving correctly rather than filling gaps plausibly.

## What I'd do differently at scale

Being explicit about the current limitations rather than implying this is production-ready:

- **Retrieval indexing**: current retrieval is a naive linear scan over all chunks
  (fine at 21 chunks, would not scale). At real scale I'd move to pgvector with an HNSW
  index for approximate nearest neighbor search, and a proper inverted index (e.g.
  Postgres full-text search or Elasticsearch) for the keyword side instead of an in-memory
  BM25 rebuild per query.
- **Permission model**: SQL evidence is currently treated as visible to all roles
  (`allowed_roles: ["all"]`) — a real system handling sensitive structured data would need
  column- or row-level permissions on the database itself (e.g. Postgres row-level
  security), not just document-level tagging.
- **SQL read-only enforcement**: SQLite's read-only connection URI is a reasonable
  approximation for this project, but a production Postgres deployment should use a
  dedicated least-privilege database role with `GRANT SELECT` only, not just a connection
  flag — genuine defense-in-depth requires enforcement the application code can't
  accidentally bypass.
- **Query decomposition**: the one-shot, ungrounded approach was measured and rejected.
  An evidence-aware iterative version (retrieve first, identify actual gaps, then decide
  whether targeted sub-queries are needed) is the direction I'd pursue next, though it's
  materially more complex to implement and evaluate correctly.
- **Evaluation set size**: 12 questions is enough to catch real, meaningful regressions
  (as it did for the BM25/reranking findings above) but is too small for statistically
  confident precision estimates — one question flipping changes metrics by ~8 percentage
  points. A production evaluation harness would need 50-100+ questions per category.
- **Testing**: current verification is real but ad-hoc (debug scripts, manual review of
  outputs) rather than a formal `pytest` suite. The SQL validator and permission filter
  are the two places a bug would be genuinely dangerous and are the highest-priority
  candidates for real automated test coverage.
- **Observability**: no structured per-query logging (latency, token usage, cost,
  retrieval strategy used) exists yet — straightforward to add given the pipeline is
  already modular, but not yet built.
