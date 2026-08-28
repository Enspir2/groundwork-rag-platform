"""
Builds data/revenue.db — a small structured dataset the SQL tool will query.
Numbers are consistent with finance_q1_report.md and finance_q2_report.md so that
a query like "why did European revenue decline in Q2 vs Q1" has a real, checkable answer
that spans BOTH the structured table and the unstructured documents.

Run this once: python3 data/build_revenue_db.py
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "revenue.db")

# Wipe and rebuild each time so it's easy to iterate during development
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE revenue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quarter TEXT NOT NULL,
    region TEXT NOT NULL,
    revenue_usd_millions REAL NOT NULL,
    qoq_growth_pct REAL NOT NULL
)
""")

rows = [
    ("Q1", "North America", 18.1, 11.0),
    ("Q1", "Europe", 11.4, 3.0),
    ("Q1", "India", 7.2, 19.0),
    ("Q1", "Rest of World", 5.6, 4.0),
    ("Q2", "North America", 19.0, 5.0),
    ("Q2", "Europe", 8.9, -22.0),
    ("Q2", "India", 7.9, 10.0),
    ("Q2", "Rest of World", 5.0, -11.0),
]

cur.executemany(
    "INSERT INTO revenue (quarter, region, revenue_usd_millions, qoq_growth_pct) VALUES (?, ?, ?, ?)",
    rows,
)

conn.commit()

# Sanity check
print("revenue.db built. Contents:")
for row in cur.execute("SELECT quarter, region, revenue_usd_millions, qoq_growth_pct FROM revenue ORDER BY quarter, region"):
    print(" ", row)

conn.close()
