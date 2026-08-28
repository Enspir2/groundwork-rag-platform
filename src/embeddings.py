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
        result = client.embed(texts, model=EMBED_MODEL, input_type="document")
        for chunk, vec in zip(to_embed, result.embeddings):
            vectors[chunk.chunk_id] = vec
        _save_cache({"model": EMBED_MODEL, "vectors": vectors})
    else:
        print("All chunks already cached — no API calls made.")

    return {c.chunk_id: vectors[c.chunk_id] for c in chunks}


def embed_query(query: str) -> list[float]:
    client = _get_client()
    result = client.embed([query], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]


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
