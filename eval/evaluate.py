"""
Evaluation harness: runs three retrieval strategies against the benchmark and
computes recall@k, precision@k, and MRR for each — so "does hybrid help? does
reranking help?" have actual measured answers instead of eyeballed ones.

Run from the eval/ directory: python evaluate.py
"""
import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunking import chunk_directory
from hybrid_retrieval import vector_rank, bm25_rank, reciprocal_rank_fusion
from reranking import rerank

TOP_K = 5
RERANK_CANDIDATE_POOL = 15  # retrieve this many via hybrid before reranking down to TOP_K


def load_benchmark() -> list[dict]:
    path = os.path.join(os.path.dirname(__file__), "benchmark.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data["questions"]


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    retrieved = set(ranking[:k])
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    retrieved = ranking[:k]
    if not retrieved:
        return 0.0
    hits = sum(1 for cid in retrieved if cid in relevant)
    return hits / len(retrieved)


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranking, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def _pack(name, questions, recalls, precisions, rrs):
    n = len(questions)
    return {
        "strategy": name,
        "recall_at_5": sum(recalls) / n,
        "precision_at_5": sum(precisions) / n,
        "mrr": sum(rrs) / n,
        "per_question": [
            {"id": q["id"], "recall": r, "precision": p, "rr": rr}
            for q, r, p, rr in zip(questions, recalls, precisions, rrs)
        ],
    }


def evaluate_all_strategies(questions: list[dict], chunks) -> tuple[dict, dict, dict]:
    """
    Computes vector_rank and bm25_rank ONCE per question and reuses them across
    all three strategies (vector-only, hybrid, hybrid+reranked) — avoiding
    redundant embed_query API calls, which matters under free-tier rate limits.
    """
    v_recalls, v_precisions, v_rrs = [], [], []
    h_recalls, h_precisions, h_rrs = [], [], []
    r_recalls, r_precisions, r_rrs = [], [], []

    chunk_by_id = {c.chunk_id: c for c in chunks}

    for q in questions:
        relevant = set(q["relevant_chunk_ids"])
        query = q["question"]

        vec_ranking = vector_rank(query, chunks)
        bm25_ranking = bm25_rank(query, chunks)
        fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking])
        hybrid_ranking = [cid for cid, _ in fused]

        # Reranking stage: take a WIDER candidate pool from hybrid, then rerank down to TOP_K
        candidate_ids = hybrid_ranking[:RERANK_CANDIDATE_POOL]
        candidates = [chunk_by_id[cid] for cid in candidate_ids]
        reranked = rerank(query, candidates, top_k=TOP_K)
        reranked_ranking = [c.chunk_id for c, _ in reranked]

        v_recalls.append(recall_at_k(vec_ranking, relevant, TOP_K))
        v_precisions.append(precision_at_k(vec_ranking, relevant, TOP_K))
        v_rrs.append(reciprocal_rank(vec_ranking, relevant))

        h_recalls.append(recall_at_k(hybrid_ranking, relevant, TOP_K))
        h_precisions.append(precision_at_k(hybrid_ranking, relevant, TOP_K))
        h_rrs.append(reciprocal_rank(hybrid_ranking, relevant))

        r_recalls.append(recall_at_k(reranked_ranking, relevant, TOP_K))
        r_precisions.append(precision_at_k(reranked_ranking, relevant, TOP_K))
        r_rrs.append(reciprocal_rank(reranked_ranking, relevant))

    return (
        _pack("vector-only", questions, v_recalls, v_precisions, v_rrs),
        _pack("hybrid (BM25+vector via RRF)", questions, h_recalls, h_precisions, h_rrs),
        _pack("hybrid + reranked", questions, r_recalls, r_precisions, r_rrs),
    )


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    docs_dir = os.path.join(here, "..", "data", "docs")
    chunks = chunk_directory(docs_dir)
    questions = load_benchmark()

    print(f"Running evaluation on {len(questions)} benchmark questions, top_k={TOP_K}")
    print("(Reranking calls the API fresh each time — no caching for rerank calls yet —")
    print(" so this run will make real API calls and may hit rate limits. That's expected.)\n")

    vector_results, hybrid_results, reranked_results = evaluate_all_strategies(questions, chunks)

    print(f"{'Strategy':<30} {'Recall@5':>10} {'Precision@5':>13} {'MRR':>8}")
    print("-" * 65)
    for r in [vector_results, hybrid_results, reranked_results]:
        print(f"{r['strategy']:<30} {r['recall_at_5']:>10.3f} {r['precision_at_5']:>13.3f} {r['mrr']:>8.3f}")

    print("\nPer-question recall@5: vector | hybrid | hybrid+reranked")
    print("-" * 65)
    for vq, hq, rq in zip(vector_results["per_question"], hybrid_results["per_question"], reranked_results["per_question"]):
        marker = "  <-- DIFFERS" if len({vq["recall"], hq["recall"], rq["recall"]}) > 1 else ""
        print(f"{vq['id']:<6} vector={vq['recall']:.2f}  hybrid={hq['recall']:.2f}  reranked={rq['recall']:.2f}{marker}")

    out_path = os.path.join(here, "results.json")
    with open(out_path, "w") as f:
        json.dump(
            {"vector_only": vector_results, "hybrid": hybrid_results, "hybrid_reranked": reranked_results},
            f, indent=2,
        )
    print(f"\nFull results saved to {out_path}")
