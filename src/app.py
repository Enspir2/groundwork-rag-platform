"""
Streamlit demo UI for Groundwork. Wires together everything built so far: role-based
permission filtering, query routing (SQL vs documents vs both), and grounded answer
synthesis with citations.

Run with: streamlit run app.py
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from router import route_and_answer

st.set_page_config(page_title="Groundwork — Enterprise RAG Platform", page_icon="📊", layout="centered")

st.title("📊 Groundwork")
st.caption("Grounded RAG platform with hybrid retrieval, reranking, permission-aware access, and query routing across SQL + documents.")

with st.expander("ℹ️ How this works"):
    st.markdown("""
    1. Your question is **classified** (Gemini) as needing structured data (SQL), documents, or both.
    2. Documents are **filtered by your selected role** before retrieval — an unauthorized
       chunk is never scored or ranked, not just hidden afterward.
    3. Document evidence is retrieved via **hybrid search** (BM25 + vector, combined via
       Reciprocal Rank Fusion) and **reranked** by a cross-encoder.
    4. SQL evidence (if needed) is generated, validated (read-only, SELECT-only,
       schema-restricted), and executed against the structured revenue database.
    5. All evidence is passed to **Claude**, which must cite every claim — and must say
       so explicitly if the evidence doesn't fully answer the question.
    """)

col1, col2 = st.columns([2, 1])
with col1:
    question = st.text_input("Ask a question", placeholder="e.g. Why did European revenue decline in Q2?")
with col2:
    role = st.selectbox("Your role", ["all", "hr", "finance", "manager"])

st.caption("Try switching roles on the same HR/finance question to see permission filtering in action.")

example_questions = [
    "Why did European revenue decline in Q2 compared to Q1?",
    "How many days of paid annual leave do employees get?",
    "What was the financial cost of the Electronics recall?",
    "What was India's revenue in Q1 vs Q2?",
]

st.write("**Try an example:**")
example_cols = st.columns(len(example_questions))
for i, eq in enumerate(example_questions):
    if example_cols[i].button(eq[:28] + "...", key=f"ex_{i}"):
        question = eq
        st.session_state["pending_question"] = eq

if "pending_question" in st.session_state:
    question = st.session_state["pending_question"]

ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question:
    with st.spinner("Routing, retrieving, and synthesizing an answer..."):
        try:
            result = route_and_answer(question, user_role=role)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            result = None

    if result:
        st.subheader("Answer")
        st.write(result["answer"])

        classification = result.get("classification", {})
        badges = []
        if classification.get("needs_sql"):
            badges.append("🗄️ SQL")
        if classification.get("needs_documents"):
            badges.append("📄 Documents")
        if badges:
            st.caption(f"Sources used: {' + '.join(badges)}  |  Role: `{role}`")

        if result.get("sql_used"):
            with st.expander("🔍 SQL query executed"):
                st.code(result["sql_used"], language="sql")

        if result["sources"]:
            st.subheader("Sources")
            for s in result["sources"]:
                with st.expander(f"[{s['citation_id']}] {s['doc_title']} — section {s['chunk_index']}"):
                    st.text(s["text"])
        elif not result["had_evidence"]:
            st.info("No evidence was found for this question with your current role — this may be a permissions restriction or the question may be outside the knowledge base.")

elif ask_clicked and not question:
    st.warning("Enter a question first.")
