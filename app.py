"""
app.py — Advanced RAG Pipeline for Industry-Specific Data
Streamlit front-end
"""

from __future__ import annotations

import streamlit as st

# Local module — contact form rendering & submission
from contact_form import render_contact_form

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced RAG Pipeline",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --indigo: #4f46e5;
        --radius: 12px;
    }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #fff;
        padding: 3rem 2rem;
        border-radius: var(--radius);
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.4rem; margin: 0 0 .5rem; }
    .hero p  { font-size: 1.1rem; opacity: .9; margin: 0; }

    /* ── Section cards ── */
    .card {
        background: #fff;
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,.07);
        margin-bottom: 1.5rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        font-size: .85rem;
        color: #888;
        padding: 1.5rem 0 .5rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
    .footer a { color: var(--indigo); text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🔍 Advanced RAG Pipeline</h1>
        <p>Industry-Specific Retrieval-Augmented Generation — ask your documents anything.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="card">',
    unsafe_allow_html=True,
)
st.subheader("Query your knowledge base")
query = st.text_input("Enter your question:", placeholder="e.g. What are the key risk factors?")

if query:
    with st.spinner("Retrieving and generating answer…"):
        # TODO: wire up actual RAG pipeline
        st.info(
            "🚧 Pipeline integration coming soon. "
            f'Your query — *"{query}"* — has been received.'
        )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# RRF / retrieval details section (placeholder)
# ---------------------------------------------------------------------------
with st.expander("📊 Retrieval details (RRF scores)", expanded=False):
    st.markdown(
        "Reciprocal Rank Fusion scores and per-source breakdown will appear here "
        "once a query is submitted."
    )

# ---------------------------------------------------------------------------
# Contact form  ← injected just above the footer
# ---------------------------------------------------------------------------
render_contact_form()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Streamlit &nbsp;·&nbsp;
        <a href="https://github.com/rohitkumarmanne-442/Advanced-RAG-Pipeline-for-Industry-Specific-Data"
           target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
