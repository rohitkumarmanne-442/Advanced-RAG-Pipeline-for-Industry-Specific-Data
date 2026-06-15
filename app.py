"""Main Streamlit application entry point for the Advanced RAG Pipeline."""
from __future__ import annotations

import streamlit as st

# ── Page config (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Advanced RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (existing design tokens) ─────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
    }
    .stApp { background-color: #f8f9fa; }
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Lazy import of contact form (avoids loading pipeline deps on contact page) ─
from src.contact import render_contact_form  # noqa: E402

# ── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='
            background: linear-gradient(135deg, var(--primary-color, #667eea),
                        var(--secondary-color, #764ba2));
            padding: 1rem;
            border-radius: 8px;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
        '>
            <h2 style='margin:0;'>🔍 RAG Pipeline</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        options=["🏠 RAG Query", "📬 Contact Us"],
        label_visibility="collapsed",
    )

# ── Page routing ─────────────────────────────────────────────────────────────
if page == "📬 Contact Us":
    render_contact_form()
else:
    # ── Main RAG Query page ───────────────────────────────────────────────
    st.markdown(
        """
        <div style='
            background: linear-gradient(135deg, var(--primary-color, #667eea) 0%,
                        var(--secondary-color, #764ba2) 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
        '>
            <h1 style='margin:0; font-size:2rem;'>🔍 Advanced RAG Pipeline</h1>
            <p style='margin:0.5rem 0 0; opacity:0.9;'>
                Industry-specific document retrieval and question answering.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Enter your query",
        placeholder="Ask anything about your documents…",
    )

    if st.button("Search", type="primary"):
        if query.strip():
            with st.spinner("Retrieving relevant documents…"):
                st.info("🚧 Pipeline integration coming soon.")
        else:
            st.warning("Please enter a query before searching.")
