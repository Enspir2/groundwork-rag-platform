"""
Evaluation harness: runs four retrieval strategies against the benchmark and
computes recall@k, precision@k, and MRR for each — vector-only, hybrid, hybrid+reranked,
and hybrid+reranked+query-decomposition — so every architectural addition has a
measured, honest answer to "did this actually help?"

Run from the eval/ directory: python evaluate.py
"""
import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunking import chunk_directory
from hybrid_retrieval import vector_rank, bm25_rank, reciprocal_rank_fusion
from reranking import rerank
from query_planning import multi_query_retrieve

TOP_K = 5
RERANK_CANDIDATE_POOL = 15


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


def evaluate_all_strategies(questions: list[dict], chunks) -> tuple[dict, dict, dict, dict]:
    metrics = {name: {"recall": [], "precision": [], "rr": []}
               for name in ["vector", "hybrid", "reranked", "planned"]}

    chunk_by_id = {c.chunk_id: c for c in chunks}

    for q in questions:
        relevant = set(q["relevant_chunk_ids"])
        query = q["question"]
        print(f"  Evaluating {q['id']}...")

        vec_ranking = vector_rank(query, chunks)
        bm25_ranking = bm25_rank(query, chunks)
        fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking])
        hybrid_ranking = [cid for cid, _ in fused]

        candidate_ids = hybrid_ranking[:RERANK_CANDIDATE_POOL]
        candidates = [chunk_by_id[cid] for cid in candidate_ids]
        reranked = rerank(query, candidates, top_k=TOP_K)
        reranked_ranking = [c.chunk_id for c, _ in reranked]

        planned_results = multi_query_retrieve(query, chunks, top_k=TOP_K)
        planned_ranking = [c.chunk_id for c, _ in planned_results]

        for name, ranking in [
            ("vector", vec_ranking),
            ("hybrid", hybrid_ranking),
            ("reranked", reranked_ranking),
            ("planned", planned_ranking),
        ]:
            metrics[name]["recall"].append(recall_at_k(ranking, relevant, TOP_K))
            metrics[name]["precision"].append(precision_at_k(ranking, relevant, TOP_K))
            metrics[name]["rr"].append(reciprocal_rank(ranking, relevant))

    names = {
        "vector": "vector-only",
        "hybrid": "hybrid (BM25+vector via RRF)",
        "reranked": "hybrid + reranked",
        "planned": "hybrid + reranked + query planning",
    }
    return tuple(
        _pack(names[key], questions, metrics[key]["recall"], metrics[key]["precision"], metrics[key]["rr"])
        for key in ["vector", "hybrid", "reranked", "planned"]
    )


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    docs_dir = os.path.join(here, "..", "data", "docs")
    chunks = chunk_directory(docs_dir)
    questions = load_benchmark()

    print(f"Running evaluation on {len(questions)} benchmark questions, top_k={TOP_K}")
    print("(Query planning adds Gemini calls per question plus extra rerank calls per")
    print(" sub-question — this run will be slower and may hit rate limits. Expected.)\n")

    results = evaluate_all_strategies(questions, chunks)

    print(f"\n{'Strategy':<38} {'Recall@5':>10} {'Precision@5':>13} {'MRR':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['strategy']:<38} {r['recall_at_5']:>10.3f} {r['precision_at_5']:>13.3f} {r['mrr']:>8.3f}")

    print("\nPer-question recall@5:")
    header = "  ".join(f"{r['strategy'].split()[0]:<10}" for r in results)
    print(f"{'':<6} {header}")
    print("-" * 75)
    for i, q in enumerate(questions):
        vals = [f"{r['per_question'][i]['recall']:<10.2f}" for r in results]
        differs = len({r['per_question'][i]['recall'] for r in results}) > 1
        marker = "  <-- DIFFERS" if differs else ""
        print(f"{q['id']:<6} {'  '.join(vals)}{marker}")

    out_path = os.path.join(here, "results.json")
    with open(out_path, "w") as f:
        json.dump({r["strategy"]: r for r in results}, f, indent=2)
    print(f"\nFull results saved to {out_path}")
