"""
Controlled SQL tool: translates a natural-language question into SQL against our
known, restricted schema, validates it defensively, and executes it against a
READ-ONLY database connection. The LLM never gets direct, unrestricted database
access — three independent layers of defense:
  1. Prompt-level: told exactly what schema exists, instructed SELECT-only.
  2. Static validation: reject anything destructive, multi-statement, or
     referencing tables outside the allowlist, BEFORE execution.
  3. Connection-level: the database connection itself is opened read-only, so
     even a validation bypass cannot cause a write.

Note: in a production Postgres setup, layer 3 would be a dedicated least-privilege
database role (GRANT SELECT only), not just a read-only connection flag — SQLite
has no per-user roles, so a read-only connection URI is our approximation of the
same principle.
"""
import os
import re
import sqlite3
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-flash-lite-latest"
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "revenue.db")

ALLOWED_TABLES = {"revenue"}

SCHEMA_DESCRIPTION = """
Table: revenue
Columns:
  - id (integer, primary key)
  - quarter (text, values: 'Q1' or 'Q2')
  - region (text, values: 'North America', 'Europe', 'India', 'Rest of World')
  - revenue_usd_millions (real, revenue in millions of USD)
  - qoq_growth_pct (real, quarter-over-quarter growth percentage; negative means decline)
"""

# Static validation: keywords that should NEVER appear in a generated query,
# regardless of what the LLM was instructed to do. Defense-in-depth — don't
# rely on the prompt instruction alone.
FORBIDDEN_KEYWORDS = [
    "drop", "delete", "insert", "update", "alter", "attach", "detach",
    "pragma", "create", "replace", "vacuum", "reindex",
]

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
        _client = genai.Client(api_key=api_key)
    return _client


SQL_SYSTEM_PROMPT = f"""You translate natural language questions into a single SQLite SELECT
statement against this exact schema:

{SCHEMA_DESCRIPTION}

Rules:
- Generate ONLY a single SELECT statement. Never generate INSERT, UPDATE, DELETE, DROP,
  ALTER, or any other statement type.
- Only reference the 'revenue' table. No other tables exist.
- Respond with ONLY the raw SQL. No markdown code fences, no explanation, no preamble.
"""


class SQLValidationError(Exception):
    pass


def generate_sql(question: str) -> str:
    client = _get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Question: {question}",
        config=types.GenerateContentConfig(system_instruction=SQL_SYSTEM_PROMPT),
    )
    sql = response.text.strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()
    return sql.strip().rstrip(";")


def validate_sql(sql: str) -> None:
    """Raises SQLValidationError if the query fails any safety check. Returns None if safe."""
    lowered = sql.lower()

    # 1. Must be a single statement — reject if a semicolon appears with more content after it
    #    (we already stripped a single trailing semicolon in generate_sql, so any remaining
    #    semicolon indicates multiple statements)
    if ";" in sql:
        raise SQLValidationError("Multiple statements detected — only one SELECT is allowed.")

    # 2. Must start with SELECT
    if not lowered.strip().startswith("select"):
        raise SQLValidationError(f"Only SELECT statements are allowed. Got: {sql!r}")

    # 3. No forbidden keywords anywhere in the query
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", lowered):
            raise SQLValidationError(f"Forbidden keyword '{kw}' detected in query.")

    # 4. Only allowlisted tables may be referenced.
    #    Simple heuristic: look for tokens after FROM/JOIN and check against the allowlist.
    #    Not a full SQL parser, but sufficient for our single-table, no-join use case —
    #    a production system with more tables would use a real SQL parser (e.g. sqlglot)
    #    instead of regex for this step.
    referenced_tables = re.findall(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered)
    referenced_tables += re.findall(r"\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered)
    for table in referenced_tables:
        if table not in ALLOWED_TABLES:
            raise SQLValidationError(f"Query references disallowed table: '{table}'")


def execute_sql(sql: str, timeout_seconds: float = 5.0) -> list[dict]:
    """Executes against a READ-ONLY connection. Raises if the DB path doesn't exist yet."""
    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"{DB_PATH} not found — run data/build_revenue_db.py first.")

    # file: URI with mode=ro opens the connection read-only at the SQLite driver level —
    # this is our read-only "role" equivalent, since SQLite has no user/grant system.
    uri = f"file:{os.path.abspath(DB_PATH)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    conn.row_factory = sqlite3.Row

    # Abort the query if it runs too long — a crude timeout via progress handler,
    # since SQLite has no native per-query timeout. Called periodically during execution.
    import time
    start = time.monotonic()

    def _timeout_check():
        return 1 if (time.monotonic() - start) > timeout_seconds else 0

    conn.set_progress_handler(_timeout_check, 1000)

    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        return rows
    finally:
        conn.close()


def answer_from_sql(question: str) -> dict:
    """
    Full pipeline: NL question -> generated SQL -> validated -> executed.
    Returns both the SQL used and the results, so the caller can show its work
    rather than presenting a number with no visible provenance.
    """
    sql = generate_sql(question)
    validate_sql(sql)  # raises SQLValidationError if unsafe
    rows = execute_sql(sql)
    return {"question": question, "sql": sql, "rows": rows}


if __name__ == "__main__":
    test_questions = [
        "What was Europe's revenue in Q2?",
        "Which region had the biggest revenue decline in Q2?",
        "Compare India's revenue between Q1 and Q2",
    ]
    for q in test_questions:
        print(f"\nQuestion: {q}")
        try:
            result = answer_from_sql(q)
            print(f"  SQL: {result['sql']}")
            print(f"  Rows: {result['rows']}")
        except SQLValidationError as e:
            print(f"  [BLOCKED] {e}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    # Deliberately test that validation actually blocks something dangerous —
    # simulating what would happen if the LLM were tricked or malfunctioned.
    print("\n--- Testing validation against a deliberately dangerous query ---")
    dangerous_sql = "DELETE FROM revenue WHERE region = 'Europe'"
    try:
        validate_sql(dangerous_sql)
        print("  [PROBLEM] Dangerous query was NOT blocked!")
    except SQLValidationError as e:
        print(f"  [BLOCKED correctly] {e}")
