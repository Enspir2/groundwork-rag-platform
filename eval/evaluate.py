"""
Evaluation harness: runs both retrieval strategies against the benchmark and
computes recall@k, precision@k, and MRR for each — so "is hybrid better?" has
an actual measured answer instead of an eyeballed one.

Run from the eval/ directory: python evaluate.py
"""
import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunking import chunk_directory
from hybrid_retrieval import vector_rank, bm25_rank, reciprocal_rank_fusion

TOP_K = 5


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


def evaluate_both_strategies(questions: list[dict], chunks) -> tuple[dict, dict]:
    """
    Computes vector_rank ONCE per question and reuses it for both the vector-only
    strategy and as an input to the hybrid strategy's RRF fusion. This halves the
    number of embed_query API calls compared to calling vector_rank and a separate
    hybrid_rank independently — which matters under a 3 requests/minute free-tier limit.
    """
    vector_recalls, vector_precisions, vector_rrs = [], [], []
    hybrid_recalls, hybrid_precisions, hybrid_rrs = [], [], []

    for q in questions:
        relevant = set(q["relevant_chunk_ids"])
        query = q["question"]

        vec_ranking = vector_rank(query, chunks)          # 1 embed_query call (cached after first run)
        bm25_ranking = bm25_rank(query, chunks)            # no API call at all
        fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking])
        hybrid_ranking = [cid for cid, _ in fused]

        vector_recalls.append(recall_at_k(vec_ranking, relevant, TOP_K))
        vector_precisions.append(precision_at_k(vec_ranking, relevant, TOP_K))
        vector_rrs.append(reciprocal_rank(vec_ranking, relevant))

        hybrid_recalls.append(recall_at_k(hybrid_ranking, relevant, TOP_K))
        hybrid_precisions.append(precision_at_k(hybrid_ranking, relevant, TOP_K))
        hybrid_rrs.append(reciprocal_rank(hybrid_ranking, relevant))

    n = len(questions)

    def _pack(name, recalls, precisions, rrs):
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

    return (
        _pack("vector-only", vector_recalls, vector_precisions, vector_rrs),
        _pack("hybrid (BM25+vector via RRF)", hybrid_recalls, hybrid_precisions, hybrid_rrs),
    )


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    docs_dir = os.path.join(here, "..", "data", "docs")
    chunks = chunk_directory(docs_dir)
    questions = load_benchmark()

    print(f"Running evaluation on {len(questions)} benchmark questions, top_k={TOP_K}")
    print("(First run will be slow due to free-tier rate limits on new queries — subsequent")
    print(" runs reuse the query cache and are near-instant.)\n")

    vector_results, hybrid_results = evaluate_both_strategies(questions, chunks)

    print(f"{'Strategy':<30} {'Recall@5':>10} {'Precision@5':>13} {'MRR':>8}")
    print("-" * 65)
    for r in [vector_results, hybrid_results]:
        print(f"{r['strategy']:<30} {r['recall_at_5']:>10.3f} {r['precision_at_5']:>13.3f} {r['mrr']:>8.3f}")

    print("\nPer-question breakdown (recall@5): vector-only vs hybrid")
    print("-" * 65)
    for vq, hq in zip(vector_results["per_question"], hybrid_results["per_question"]):
        marker = ""
        if vq["recall"] != hq["recall"]:
            marker = "  <-- DIFFERS"
        print(f"{vq['id']:<6} vector={vq['recall']:.2f}  hybrid={hq['recall']:.2f}{marker}")

    # Save full results for the README / future comparisons
    out_path = os.path.join(here, "results.json")
    with open(out_path, "w") as f:
        json.dump({"vector_only": vector_results, "hybrid": hybrid_results}, f, indent=2)
    print(f"\nFull results saved to {out_path}")
