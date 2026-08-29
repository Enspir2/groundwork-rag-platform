"""
Query planning: uses Gemini (Flash) to decide whether a question needs decomposition
into multiple narrower sub-questions, and if so, produces them. Single retrieval
queries tend to converge on chunks that resemble the "average meaning" of the
question, missing causally-relevant-but-lexically-distant evidence (e.g. a product
recall document that never uses the words "revenue" or "decline"). Decomposing into
sub-questions lets each piece of evidence be retrieved via a query that's actually
close to it in meaning.

Why Gemini here and Claude elsewhere in this project: this is a narrow, structured
classification/decomposition task, not a task that needs frontier-model reasoning.
Using a fast, free-tier model for this routing step and reserving Claude for the
final grounded-answer synthesis (later phase) is a real production pattern —
heterogeneous model routing — where cost/latency-sensitive sub-tasks use a cheaper
model and only the highest-value step uses the most capable one.
"""
import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3.6-flash"  # free-tier model name as of Aug 2026 — Google retires/renames
                             # these periodically; if this 404s again, check
                             # https://aistudio.google.com for the current free-tier model list

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
        _client = genai.Client(api_key=api_key)
    return _client


DECOMPOSE_SYSTEM_PROMPT = """You are a query planning component in a retrieval system.

Given a user's question, decide whether answering it well requires evidence from
MULTIPLE distinct angles/topics, or whether it's a single, focused question that a
single retrieval query can handle well.

If it's simple/single-topic: return a JSON array containing just the original question, unchanged.

If it genuinely needs multiple angles (e.g. it asks "why" about something with several
possible contributing factors, or asks to compare two different things): break it into
2-4 focused sub-questions. Each sub-question should be independently answerable and
should target a DISTINCT piece of evidence, not just reword the same question.

Respond with ONLY a JSON array of strings. No preamble, no explanation, no markdown
code fences — just the raw JSON array.

Examples:

Question: "How many days of paid annual leave do employees get?"
Response: ["How many days of paid annual leave do employees get?"]

Question: "Why did European revenue decline in Q2 compared to Q1?"
Response: ["What factors does the Q2 finance report give for Europe's revenue decline?", "What happened with the Electronics product recall in Europe?", "What issues occurred with the European ad platform migration?"]
"""


def decompose_query(question: str, max_retries: int = 3) -> list[str]:
    """
    Returns a list of sub-questions. For simple questions, this will be a
    single-item list containing the original question unchanged.
    Falls back to [question] on any parsing failure or API error, so a bad
    decomposition never breaks the pipeline — it just degrades to single-query behavior.
    """
    client = _get_client()

    raw_text = ""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=f"Question: {question!r}",
                config=types.GenerateContentConfig(system_instruction=DECOMPOSE_SYSTEM_PROMPT),
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:].strip()

            sub_questions = json.loads(raw_text)
            if isinstance(sub_questions, list) and all(isinstance(s, str) for s in sub_questions) and sub_questions:
                return sub_questions
            else:
                print(f"  [decompose] Unexpected format, falling back to original question: {raw_text!r}")
                return [question]

        except json.JSONDecodeError:
            print(f"  [decompose] Failed to parse JSON, falling back to original question. Raw: {raw_text!r}")
            return [question]
        except Exception as e:
            if "rate" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e):
                wait = 15 * (attempt + 1)
                print(f"  [decompose] Rate limited, waiting {wait}s and retrying ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                print(f"  [decompose] Unexpected error ({e}), falling back to original question.")
                return [question]

    print("  [decompose] Exceeded retries, falling back to original question.")
    return [question]


def multi_query_retrieve(question: str, chunks, top_k: int = 5, per_subquery_k: int = 5):
    """
    Decomposes `question`, retrieves+reranks for EACH sub-question independently,
    then merges results by taking the best rerank score seen for each chunk_id
    across all sub-queries (a chunk retrieved strongly by any sub-question counts).
    """
    import sys
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__)))
    from hybrid_retrieval import bm25_rank, vector_rank, reciprocal_rank_fusion
    from reranking import rerank

    chunk_by_id = {c.chunk_id: c for c in chunks}
    sub_questions = decompose_query(question)

    best_score_by_chunk: dict[str, float] = {}

    for sub_q in sub_questions:
        bm25_ranking = bm25_rank(sub_q, chunks)
        vec_ranking = vector_rank(sub_q, chunks)
        fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking])
        candidate_ids = [cid for cid, _ in fused[:15]]
        candidates = [chunk_by_id[cid] for cid in candidate_ids]

        reranked = rerank(sub_q, candidates, top_k=per_subquery_k)
        for c, score in reranked:
            if c.chunk_id not in best_score_by_chunk or score > best_score_by_chunk[c.chunk_id]:
                best_score_by_chunk[c.chunk_id] = score

    merged = sorted(best_score_by_chunk.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_by_id[cid], score) for cid, score in merged[:top_k]]


if __name__ == "__main__":
    test_questions = [
        "How many days of paid annual leave do employees get?",
        "Why did European revenue decline in Q2?",
        "Did India's revenue grow or shrink between Q1 and Q2?",
    ]
    for q in test_questions:
        print(f"\nOriginal: {q}")
        sub_qs = decompose_query(q)
        for i, sq in enumerate(sub_qs, 1):
            print(f"  {i}. {sq}")

    print("\n" + "=" * 70)
    print("Full multi-query retrieval test on the hard question (q1):")
    print("=" * 70)
    import os as _os
    from chunking import chunk_directory
    here = _os.path.dirname(__file__)
    docs_dir = _os.path.join(here, "..", "data", "docs")
    chunks = chunk_directory(docs_dir)

    q1 = "Why did European revenue decline in Q2?"
    results = multi_query_retrieve(q1, chunks, top_k=5)
    for i, (c, score) in enumerate(results, 1):
        print(f"{i}. score={score:.4f} [{c.doc_title}] chunk {c.chunk_index}: {c.text[:80]!r}")
