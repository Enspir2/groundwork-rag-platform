"""
Hybrid retrieval: combines BM25 (keyword) and vector (semantic) search using
Reciprocal Rank Fusion (RRF).

Why RRF instead of normalizing + weighted-summing raw scores:
BM25 scores are unbounded and corpus-dependent; cosine similarity is bounded [-1, 1].
There's no principled way to say "a BM25 score of 8 equals a cosine score of 0.7" —
they're not the same unit. RRF sidesteps this entirely by only looking at RANK
(1st place, 2nd place, ...) from each method, which is always comparable regardless
of the underlying scoring scale.

RRF formula: for each chunk, score = sum over each ranking method of 1 / (k + rank)
where rank is 1-indexed position in that method's results, and k=60 is a constant
from the original RRF paper (Cormack et al.) — a convention, not something we derived.
"""
import os
import re
from rank_bm25 import BM25Okapi

from chunking import chunk_directory, Chunk
from embeddings import embed_chunks, embed_query, cosine_similarity

RRF_K = 60


def _tokenize(text: str) -> list[str]:
    # Simple whitespace + lowercase tokenizer. Good enough for BM25 here —
    # a production system might add stemming, but that adds a tuning knob
    # we don't need to justify for a corpus this size.
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_rank(query: str, chunks: list[Chunk]) -> list[str]:
    """Returns chunk_ids ranked best-to-worst by BM25 score."""
    tokenized_corpus = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c.chunk_id for c, _ in ranked]


def vector_rank(query: str, chunks: list[Chunk]) -> list[str]:
    """Returns chunk_ids ranked best-to-worst by cosine similarity."""
    chunk_vectors = embed_chunks(chunks)
    q_vec = embed_query(query)
    scored = [
        (c.chunk_id, cosine_similarity(q_vec, chunk_vectors[c.chunk_id]))
        for c in chunks
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in scored]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """
    rankings: a list of ranked chunk_id lists (one per retrieval method).
    Returns chunk_ids sorted by fused RRF score, descending.
    """
    fused_scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query: str, chunks: list[Chunk], top_k: int = 5) -> list[tuple[Chunk, float]]:
    bm25_ranking = bm25_rank(query, chunks)
    vec_ranking = vector_rank(query, chunks)
    fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking])

    chunk_by_id = {c.chunk_id: c for c in chunks}
    return [(chunk_by_id[cid], score) for cid, score in fused[:top_k]]


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    docs_dir = os.path.join(here, "..", "data", "docs")
    chunks = chunk_directory(docs_dir)

    query = "Why did European revenue decline in Q2?"
    print(f"\nQuery: {query}\n")

    print("=" * 70)
    print("VECTOR-ONLY top 5:")
    print("=" * 70)
    vec_ranking = vector_rank(query, chunks)
    chunk_by_id = {c.chunk_id: c for c in chunks}
    for i, cid in enumerate(vec_ranking[:5], 1):
        c = chunk_by_id[cid]
        print(f"{i}. [{c.doc_title}] chunk {c.chunk_index}: {c.text[:80]}...")

    print()
    print("=" * 70)
    print("HYBRID (BM25 + Vector via RRF) top 5:")
    print("=" * 70)
    results = hybrid_search(query, chunks, top_k=5)
    for i, (c, score) in enumerate(results, 1):
        print(f"{i}. score={score:.5f} [{c.doc_title}] chunk {c.chunk_index}: {c.text[:80]}...")
