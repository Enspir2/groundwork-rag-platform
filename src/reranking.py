"""
Reranking: takes a candidate set from first-stage retrieval (hybrid search) and
reorders it using a cross-encoder model that scores each (query, candidate) pair
jointly — catching relevance signals that term-overlap (BM25) and independently-
encoded vectors (embeddings) can miss.

Pattern: retrieve MORE candidates than you need (e.g. top 15) via cheap hybrid
search, then rerank down to the final top_k. Reranking a small candidate set is
fine; reranking the entire corpus per query would be far too slow, since a
cross-encoder can't precompute document representations ahead of time.
"""
import os
import time
import json
import hashlib
from dotenv import load_dotenv
import voyageai

from chunking import Chunk

load_dotenv()

RERANK_MODEL = "rerank-2-lite"  # cheaper tier, sufficient for this candidate set size
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rerank_cache.json")

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set. Copy .env.example to .env and add your key.")
        _client = voyageai.Client(api_key=api_key)
    return _client


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def _cache_key(query: str, candidates: list[Chunk], top_k: int) -> str:
    # Key on query + exact set of candidate chunk_ids + top_k, so a different
    # candidate pool (e.g. from a retrieval change) correctly misses the cache
    # instead of returning a stale rerank result for the wrong input set.
    candidate_ids = ",".join(c.chunk_id for c in candidates)
    raw = f"{RERANK_MODEL}:{query}:{candidate_ids}:{top_k}"
    return hashlib.md5(raw.encode()).hexdigest()


def rerank(query: str, candidates: list[Chunk], top_k: int = 5, max_retries: int = 5) -> list[tuple[Chunk, float]]:
    """
    Reranks `candidates` (already retrieved by a cheaper first-stage method)
    against `query`, returning the top_k as (Chunk, relevance_score) pairs,
    ordered best-first. Cached by (query, candidate set, top_k) so re-running
    evaluation doesn't re-call the API for unchanged inputs.
    """
    if not candidates:
        return []

    cache = _load_cache()
    key = _cache_key(query, candidates, top_k)
    chunk_by_id = {c.chunk_id: c for c in candidates}

    if key in cache:
        return [(chunk_by_id[cid], score) for cid, score in cache[key]]

    client = _get_client()
    documents = [c.text for c in candidates]

    for attempt in range(max_retries):
        try:
            result = client.rerank(query=query, documents=documents, model=RERANK_MODEL, top_k=top_k)
            break
        except Exception as e:
            if "RateLimitError" in type(e).__name__ or "rate limit" in str(e).lower():
                wait = 21 * (attempt + 1)
                print(f"  [rate limited] waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError("Exceeded max retries due to rate limiting.")

    output = [(candidates[r.index], r.relevance_score) for r in result.results]

    cache[key] = [(c.chunk_id, score) for c, score in output]
    _save_cache(cache)

    return output


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from chunking import chunk_directory
    from hybrid_retrieval import bm25_rank, vector_rank, reciprocal_rank_fusion

    here = os.path.dirname(__file__)
    docs_dir = os.path.join(here, "..", "data", "docs")
    chunks = chunk_directory(docs_dir)
    chunk_by_id = {c.chunk_id: c for c in chunks}

    query = "Did India's revenue grow or shrink between Q1 and Q2?"
    print(f"Query: {query}\n")

    # First-stage: hybrid retrieval, but pull MORE candidates than final top_k
    bm25_ranking = bm25_rank(query, chunks)
    vec_ranking = vector_rank(query, chunks)
    fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking])
    candidate_ids = [cid for cid, _ in fused[:15]]  # wider net than the final top 5
    candidates = [chunk_by_id[cid] for cid in candidate_ids]

    print("Hybrid top 8 (before reranking):")
    for i, c in enumerate(candidates[:8], 1):
        print(f"  {i}. [{c.doc_title}] chunk {c.chunk_index}: {c.text[:70]!r}")

    reranked = rerank(query, candidates, top_k=5)

    print("\nAfter reranking (top 5):")
    for i, (c, score) in enumerate(reranked, 1):
        print(f"  {i}. score={score:.4f} [{c.doc_title}] chunk {c.chunk_index}: {c.text[:70]!r}")
