"""
Embeddings + naive vector search.

Design choices worth being able to defend in an interview:
1. We cache embeddings to disk (embeddings_cache.json) keyed by chunk_id, so re-running
   this script doesn't re-call the API for chunks we've already embedded. Embedding APIs
   cost money and are rate-limited — re-embedding unchanged content on every run is wasteful
   and, at scale, would be a real cost/latency problem.
2. Search here is a naive linear scan (compare query vector against every chunk vector).
   This is intentional at this stage — it's the mechanism an ANN index (e.g. pgvector's HNSW)
   is built to approximate at scale. Understand this before reaching for the index that hides it.
3. We embed with a fixed model name stored alongside the cache, so if you change the model
   later, we can detect the mismatch instead of silently comparing incompatible vectors.
"""
import os
import json
import numpy as np
from dotenv import load_dotenv
import voyageai

from chunking import chunk_directory, Chunk

load_dotenv()

EMBED_MODEL = "voyage-3.5-lite"  # cheap + fast; fine for this corpus size
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings_cache.json")

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = voyageai.Client(api_key=api_key)
    return _client


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            data = json.load(f)
        if data.get("model") != EMBED_MODEL:
            print(
                f"[warning] Cache was built with model '{data.get('model')}', "
                f"but current EMBED_MODEL is '{EMBED_MODEL}'. Ignoring stale cache."
            )
            return {"model": EMBED_MODEL, "vectors": {}}
        return data
    return {"model": EMBED_MODEL, "vectors": {}}


def _save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def embed_chunks(chunks: list[Chunk]) -> dict[str, list[float]]:
    """
    Returns {chunk_id: vector}. Uses cache for chunks already embedded;
    only calls the API for new/uncached chunks.
    """
    cache = _load_cache()
    vectors = cache["vectors"]

    to_embed = [c for c in chunks if c.chunk_id not in vectors]
    if to_embed:
        print(f"Embedding {len(to_embed)} new chunk(s) (out of {len(chunks)} total)...")
        client = _get_client()
        texts = [c.text for c in to_embed]
        # input_type="document" tells Voyage this is indexed content, not a live query —
        # Voyage's models are trained to treat queries and documents asymmetrically for
        # better retrieval quality.
        result = _embed_with_retry(client, texts, input_type="document")
        for chunk, vec in zip(to_embed, result.embeddings):
            vectors[chunk.chunk_id] = vec
        _save_cache({"model": EMBED_MODEL, "vectors": vectors})
    else:
        print("All chunks already cached — no API calls made.")

    return {c.chunk_id: vectors[c.chunk_id] for c in chunks}


def _query_cache_key(query: str) -> str:
    import hashlib
    return hashlib.md5(f"{EMBED_MODEL}:{query}".encode()).hexdigest()


def embed_query(query: str) -> list[float]:
    """
    Cached + rate-limit-aware. Voyage's free tier (no payment method attached)
    allows only 3 requests/minute — without caching and throttling, evaluating
    12+ benchmark questions in a loop blows through that limit immediately.
    """
    cache = _load_cache()
    query_cache = cache.setdefault("queries", {})
    key = _query_cache_key(query)

    if key in query_cache:
        return query_cache[key]

    client = _get_client()
    result = _embed_with_retry(client, [query], input_type="query")
    vec = result.embeddings[0]

    query_cache[key] = vec
    _save_cache(cache)
    return vec


def _embed_with_retry(client, texts: list[str], input_type: str, max_retries: int = 5):
    """
    Handles Voyage's free-tier rate limit (3 RPM without a payment method) with
    exponential backoff. Without this, a batch of queries fails outright instead
    of just slowing down — and a real production system would need the same kind
    of resilience against any third-party API's rate limits, not just in dev.
    """
    import time
    for attempt in range(max_retries):
        try:
            return client.embed(texts, model=EMBED_MODEL, input_type=input_type)
        except Exception as e:
            if "RateLimitError" in type(e).__name__ or "rate limit" in str(e).lower():
                wait = 21 * (attempt + 1)  # free tier resets roughly every 20s for 3 RPM
                print(f"  [rate limited] waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded max retries due to rate limiting.")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def vector_search(query: str, chunks: list[Chunk], top_k: int = 5) -> list[tuple[Chunk, float]]:
    """Naive linear-scan search: compare query vector against every chunk vector."""
    chunk_vectors = embed_chunks(chunks)
    q_vec = embed_query(query)

    scored = [
        (chunk, cosine_similarity(q_vec, chunk_vectors[chunk.chunk_id]))
        for chunk in chunks
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    docs_dir = os.path.join(here, "..", "data", "docs")
    chunks = chunk_directory(docs_dir)

    query = "Why did European revenue decline in Q2?"
    print(f"\nQuery: {query}\n")

    results = vector_search(query, chunks, top_k=5)
    for chunk, score in results:
        print(f"score={score:.4f}  [{chunk.doc_title}] chunk {chunk.chunk_index}")
        print("  ", chunk.text[:120].replace("\n", " "), "...")
        print()
