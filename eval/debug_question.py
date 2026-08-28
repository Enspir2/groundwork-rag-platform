"""
Debug helper: for a single benchmark question, show BOTH the BM25 ranking and
the vector ranking side by side, so we can see exactly which chunks each method
favored and why the fused (hybrid) result differs from vector-only.

Usage: python debug_question.py q8
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunking import chunk_directory
from hybrid_retrieval import vector_rank, bm25_rank, reciprocal_rank_fusion


def main(question_id: str):
    here = os.path.dirname(__file__)
    docs_dir = os.path.join(here, "..", "data", "docs")
    chunks = chunk_directory(docs_dir)
    chunk_by_id = {c.chunk_id: c for c in chunks}

    with open(os.path.join(here, "benchmark.json")) as f:
        questions = json.load(f)["questions"]
    q = next(q for q in questions if q["id"] == question_id)

    print(f"Question: {q['question']}")
    print(f"Gold relevant chunk_ids: {q['relevant_chunk_ids']}\n")

    bm25_ranking = bm25_rank(q["question"], chunks)
    vec_ranking = vector_rank(q["question"], chunks)
    fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking])
    hybrid_ranking = [cid for cid, _ in fused]

    def describe(cid):
        c = chunk_by_id[cid]
        tag = " <== GOLD" if cid in q["relevant_chunk_ids"] else ""
        return f"[{c.doc_title}] chunk {c.chunk_index}: {c.text[:70]!r}{tag}"

    print("BM25 ranking (top 8):")
    for i, cid in enumerate(bm25_ranking[:8], 1):
        print(f"  {i}. {describe(cid)}")

    print("\nVector ranking (top 8):")
    for i, cid in enumerate(vec_ranking[:8], 1):
        print(f"  {i}. {describe(cid)}")

    print("\nHybrid (RRF fused) ranking (top 8):")
    for i, cid in enumerate(hybrid_ranking[:8], 1):
        print(f"  {i}. {describe(cid)}")


if __name__ == "__main__":
    qid = sys.argv[1] if len(sys.argv) > 1 else "q8"
    main(qid)
