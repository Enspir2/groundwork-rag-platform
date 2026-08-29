"""
Grounded answer synthesis: takes a question + retrieved evidence chunks, and produces
an answer with inline citations tying every claim back to a specific source. Explicitly
instructed to say "insufficient information" rather than filling gaps from the model's
own training knowledge — that's what makes this "grounded" rather than just "has some
context in the prompt."
"""
import os
import time
from dotenv import load_dotenv
import anthropic

from chunking import Chunk

load_dotenv()

MODEL = "claude-sonnet-4-6"

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


SYNTHESIS_SYSTEM_PROMPT = """You are a grounded question-answering system. You will be given
a question and a numbered list of evidence excerpts retrieved from a document/data store.

Rules:
- Answer ONLY using the provided evidence. Do not use any outside knowledge.
- Every factual claim in your answer MUST be followed by a citation to the evidence
  number(s) that support it, in the format [1], [2], etc. Use multiple like [1][3] if
  a claim draws on more than one piece of evidence.
- If the evidence does NOT fully answer the question, say so explicitly — name what's
  missing rather than guessing or inferring beyond what's stated.
- Do not pad the answer with generic statements not grounded in the evidence.
- Be concise. A few sentences is usually enough.
"""


def _format_evidence(chunks: list[Chunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] (source: {c.doc_title}, section {c.chunk_index})\n{c.text}\n")
    return "\n".join(lines)


def synthesize_answer(question: str, evidence_chunks: list[Chunk], max_retries: int = 3) -> dict:
    """
    Returns {"answer": str, "sources": [{"citation_id": int, "doc_title": str,
    "chunk_index": int, "text": str}], "had_evidence": bool}
    """
    if not evidence_chunks:
        # No evidence retrieved at all — don't call the API. Answering anyway would mean
        # the LLM falls back to its own training knowledge, defeating the point of "grounded."
        return {
            "answer": "I don't have any relevant evidence to answer this question.",
            "sources": [],
            "had_evidence": False,
        }

    evidence_block = _format_evidence(evidence_chunks)
    sources = [
        {"citation_id": i, "doc_title": c.doc_title, "chunk_index": c.chunk_index, "text": c.text}
        for i, c in enumerate(evidence_chunks, 1)
    ]

    client = _get_client()
    user_message = f"Question: {question}\n\nEvidence:\n{evidence_block}"

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=600,
                system=SYNTHESIS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            answer_text = response.content[0].text.strip()
            return {"answer": answer_text, "sources": sources, "had_evidence": True}
        except Exception as e:
            if "overloaded" in str(e).lower() or "rate" in str(e).lower() or "429" in str(e):
                wait = 10 * (attempt + 1)
                print(f"  [synthesis] API issue, waiting {wait}s and retrying ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Exceeded max retries during answer synthesis.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from chunking import chunk_directory
    from hybrid_retrieval import bm25_rank, vector_rank, reciprocal_rank_fusion
    from reranking import rerank

    here = os.path.dirname(__file__)
    docs_dir = os.path.join(here, "..", "data", "docs")
    chunks = chunk_directory(docs_dir)
    chunk_by_id = {c.chunk_id: c for c in chunks}

    def retrieve_and_rerank(question, top_k=5):
        bm25_ranking = bm25_rank(question, chunks)
        vec_ranking = vector_rank(question, chunks)
        fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking])
        candidate_ids = [cid for cid, _ in fused[:15]]
        candidates = [chunk_by_id[cid] for cid in candidate_ids]
        reranked = rerank(question, candidates, top_k=top_k)
        return [c for c, _ in reranked]

    test_cases = [
        "How many days of paid annual leave do employees get?",
        "Why did European revenue decline in Q2?",
        "What was the financial cost of the Electronics recall, and how many units were affected?",
    ]

    for q in test_cases:
        print(f"\n{'=' * 70}\nQuestion: {q}\n{'=' * 70}")
        evidence = retrieve_and_rerank(q)
        result = synthesize_answer(q, evidence)
        print(f"\nAnswer:\n{result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['citation_id']}] {s['doc_title']} (section {s['chunk_index']})")

    print(f"\n{'=' * 70}\nTesting empty-evidence guard (no API call should happen):\n{'=' * 70}")
    result = synthesize_answer("Some question", [])
    print(f"Answer: {result['answer']}")
    print(f"had_evidence: {result['had_evidence']}")
