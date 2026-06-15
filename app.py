"""Main entry point for the Advanced RAG Pipeline Streamlit app."""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS (gradient header variables + card borders)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --border-color: #e0e0e0;
        }
        .stApp { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Lazy imports (keep pipeline resources out of the contact page path)
# ---------------------------------------------------------------------------

def _render_rag_page() -> None:  # pragma: no cover
    """Render the main RAG query interface."""
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, var(--primary-color) 0%,
                        var(--secondary-color) 100%);
            padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
        ">
            <h1 style='color:white; margin:0;'>🔍 Advanced RAG Pipeline</h1>
            <p style='color:rgba(255,255,255,0.85); margin:0.4rem 0 0;'>
                Query your industry-specific data with AI.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Enter your query below to search the knowledge base.")
    query = st.text_input("Your question", placeholder="Ask anything…")
    if st.button("Search", use_container_width=True) and query:
        st.write(f"*(Pipeline response for: {query!r} would appear here.)*")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        label="Go to",
        options=["RAG Query", "Contact Us"],
        index=0,
        key="nav_page",
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#888;'>Advanced RAG Pipeline v1.0</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if page == "Contact Us":
    from src.contact import render_contact_form  # imported lazily
    render_contact_form()
else:
    _render_rag_page()
