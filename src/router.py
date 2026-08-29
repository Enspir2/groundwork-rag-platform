"""
Query router: the actual front door of the system. Classifies a question as needing
structured data (SQL), documents, or both, gathers evidence from whichever sources
are needed, applies permission filtering, and passes everything into synthesis
together — so one question can draw on both the database and the documents.
"""
import os
import json
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(__file__))
from chunking import chunk_directory, filter_chunks_by_role, Chunk
from hybrid_retrieval import bm25_rank, vector_rank, reciprocal_rank_fusion
from reranking import rerank
from synthesis import synthesize_answer
from sql_tool import answer_from_sql, SQLValidationError, SCHEMA_DESCRIPTION

load_dotenv()

ROUTER_MODEL = "gemini-flash-lite-latest"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


ROUTER_SYSTEM_PROMPT = f"""You classify questions for a retrieval system with two data sources:

1. A structured database with this schema:
{SCHEMA_DESCRIPTION}

2. A set of internal documents (finance reports, HR policy, product notes) containing
   narrative explanations, policies, and context that aren't in the database.

Given a question, decide which source(s) are needed to answer it well.
Respond with ONLY a JSON object: {{"needs_sql": true/false, "needs_documents": true/false}}
No explanation, no markdown fences — just the raw JSON object.

Examples:
Question: "What was Europe's revenue in Q2?"
Response: {{"needs_sql": true, "needs_documents": false}}

Question: "How many days of annual leave do employees get?"
Response: {{"needs_sql": false, "needs_documents": true}}

Question: "Why did European revenue decline in Q2 compared to Q1?"
Response: {{"needs_sql": true, "needs_documents": true}}
"""


def classify_question(question: str) -> dict:
    """Returns {"needs_sql": bool, "needs_documents": bool}. Defaults to documents-only
    (the safer fallback) if classification fails for any reason."""
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=ROUTER_MODEL,
            contents=f"Question: {question}",
            config=types.GenerateContentConfig(system_instruction=ROUTER_SYSTEM_PROMPT),
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        result = json.loads(raw)
        return {
            "needs_sql": bool(result.get("needs_sql", False)),
            "needs_documents": bool(result.get("needs_documents", True)),
        }
    except Exception as e:
        print(f"  [router] Classification failed ({e}), defaulting to documents-only.")
        return {"needs_sql": False, "needs_documents": True}


def _sql_result_to_pseudo_chunk(sql_result: dict) -> Chunk:
    """Wraps a SQL query result as a Chunk so synthesis can treat it identically
    to document evidence — same citation mechanism, same evidence format."""
    rows_text = "\n".join(str(row) for row in sql_result["rows"]) or "(no matching rows)"
    text = f"Query: {sql_result['question']}\nSQL executed: {sql_result['sql']}\nResults:\n{rows_text}"
    return Chunk(
        chunk_id="sql_result",
        doc_id="revenue_database",
        doc_title="Structured Revenue Database (live SQL query)",
        text=text,
        chunk_index=0,
        metadata={"allowed_roles": ["all"]},  # structured revenue numbers treated as broadly visible;
                                                # a real system might restrict this per-column instead
    )


def route_and_answer(question: str, user_role: str = "all", top_k: int = 5) -> dict:
    """
    Full pipeline: classify -> gather evidence from needed source(s) -> permission-filter
    documents -> retrieve+rerank -> synthesize a grounded, cited answer.
    """
    classification = classify_question(question)
    print(f"  [router] needs_sql={classification['needs_sql']}, needs_documents={classification['needs_documents']}")

    evidence: list[Chunk] = []
    sql_used = None

    if classification["needs_sql"]:
        try:
            sql_result = answer_from_sql(question)
            sql_used = sql_result["sql"]
            evidence.append(_sql_result_to_pseudo_chunk(sql_result))
        except SQLValidationError as e:
            print(f"  [router] SQL generation was blocked by validation: {e}")
        except Exception as e:
            print(f"  [router] SQL path failed ({e}), continuing with documents only.")

    if classification["needs_documents"]:
        here = os.path.dirname(__file__)
        docs_dir = os.path.join(here, "..", "data", "docs")
        all_chunks = chunk_directory(docs_dir)
        # Permission filtering happens BEFORE retrieval — see Phase 8 notes.
        allowed_chunks = filter_chunks_by_role(all_chunks, user_role)

        bm25_ranking = bm25_rank(question, allowed_chunks)
        vec_ranking = vector_rank(question, allowed_chunks)
        fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking])
        chunk_by_id = {c.chunk_id: c for c in allowed_chunks}
        candidate_ids = [cid for cid, _ in fused[:15]]
        candidates = [chunk_by_id[cid] for cid in candidate_ids]

        if candidates:
            reranked = rerank(question, candidates, top_k=top_k)
            evidence.extend([c for c, _ in reranked])

    result = synthesize_answer(question, evidence)
    result["sql_used"] = sql_used
    result["classification"] = classification
    return result


if __name__ == "__main__":
    test_questions = [
        ("What was Europe's revenue in Q2?", "all"),
        ("Why did European revenue decline in Q2 compared to Q1?", "manager"),
        ("How many days of paid annual leave do employees get?", "hr"),
    ]

    for question, role in test_questions:
        print(f"\n{'=' * 70}\nQuestion: {question}  (role={role})\n{'=' * 70}")
        result = route_and_answer(question, user_role=role)
        if result.get("sql_used"):
            print(f"  SQL used: {result['sql_used']}")
        print(f"\nAnswer:\n{result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['citation_id']}] {s['doc_title']} (section {s['chunk_index']})")
