"""Main Streamlit entry-point for the Advanced RAG Pipeline app."""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS (design tokens used by all pages)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --card-border: 1px solid #e0e0e0;
        --card-radius: 10px;
    }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .block-container { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔍 RAG Pipeline")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        options=["Query", "Contact Us"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Advanced RAG Pipeline v1.0")

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if page == "Contact Us":
    from src.contact import render_contact_form
    render_contact_form()
else:
    # ---- Main Query page (existing pipeline UI) --------------------------
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, var(--primary-color, #667eea) 0%,
                        var(--secondary-color, #764ba2) 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        ">
            <h1 style="color: white; margin: 0;">🔍 Advanced RAG Pipeline</h1>
            <p style="color: rgba(255,255,255,0.85); margin: 0.5rem 0 0;">
                Industry-Specific Document Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Enter your query below to search across the indexed document corpus.",
        icon="💡",
    )
    query = st.text_input("Your question", placeholder="What does policy X say about Y?")
    if st.button("Search", type="primary") and query:
        st.warning("Pipeline integration coming soon.", icon="🚧")
