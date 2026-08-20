"""
Advanced RAG Pipeline - Visual Web Application
Streamlit-based interface that visualizes the entire RAG pipeline process.
"""

import streamlit as st
import time
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import numpy as np
import pandas as pd
from loguru import logger

# Suppress loguru output in web app
logger.remove()

# Detect cloud deployment
IS_CLOUD = os.environ.get("STREAMLIT_CLOUD", "false").lower() == "true" or os.environ.get("GROQ_API_KEY", "") != ""

# Load secrets from Streamlit Cloud if available
try:
    if hasattr(st, "secrets"):
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
            IS_CLOUD = True
        if "STREAMLIT_CLOUD" in st.secrets:
            IS_CLOUD = True
except Exception:
    pass

CONFIG_PATH = "config/settings_cloud.yaml" if IS_CLOUD else "config/settings.yaml"

# ─── Page Configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Advanced RAG Pipeline | SEC Filings",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Hide default streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Root variables */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --secondary: #ec4899;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border: #334155;
        --radius-sm: 12px;
        --radius-lg: 16px;
    }

    /* Base line-height bump for readability */
    html, body, [class*="css"] { line-height: 1.75; }

    /* Main header styling */
    .hero-container {
        text-align: center;
        padding: 1.25rem 1rem 0.75rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #ec4899, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }

    /* Hero badge — animated shimmer gradient */
    .hero-badge {
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-top: 0.35rem;
        background: linear-gradient(120deg,
            #6366f1 0%, #8b5cf6 25%, #ec4899 50%, #8b5cf6 75%, #6366f1 100%);
        background-size: 300% 100%;
        animation: hero-shimmer 3.5s linear infinite;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
    .hero-badge::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 50px;
        padding: 1px;
        background: linear-gradient(120deg,
            rgba(255,255,255,0.6), rgba(255,255,255,0) 40%,
            rgba(255,255,255,0) 60%, rgba(255,255,255,0.6));
        background-size: 200% 100%;
        animation: hero-badge-sheen 2.5s ease-in-out infinite;
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite: xor;
                mask-composite: exclude;
        pointer-events: none;
    }
    @keyframes hero-shimmer {
        0%   { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }
    @keyframes hero-badge-sheen {
        0%   { background-position: -100% 0; }
        60%  { background-position: 200% 0; }
        100% { background-position: 200% 0; }
    }

    /* Pipeline flow */
    .pipeline-container {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
    }
    .pipeline-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.3rem;
        flex-wrap: wrap;
    }
    .flow-step {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 0.6rem 1.1rem;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #475569;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        transition: all 0.2s;
    }
    .flow-step:hover {
        border-color: #6366f1;
        color: #6366f1;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
    }
    .flow-arrow {
        color: #cbd5e1;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Metric cards */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.8rem;
        margin: 1rem 0;
        grid-auto-rows: 1fr;
        align-items: stretch;
    }
    @media (max-width: 768px) {
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .metric-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 1.1rem 1rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);
        transition: transform 0.2s;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 108px;
        height: 100%;
        overflow: hidden;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card.green { background: linear-gradient(135deg, #10b981 0%, #059669 100%); box-shadow: 0 4px 20px rgba(16, 185, 129, 0.25); }
    .metric-card.cyan  { background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); box-shadow: 0 4px 20px rgba(6, 182, 212, 0.25); }
    .metric-card.pink  { background: linear-gradient(135deg, #ec4899 0%, #db2777 100%); box-shadow: 0 4px 20px rgba(236, 72, 153, 0.25); }
    .metric-card.amber { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); box-shadow: 0 4px 20px rgba(245, 158, 11, 0.25); }
    .metric-value-row {
        display: flex;
        align-items: baseline;
        justify-content: flex-end;
        gap: 0.35rem;
        line-height: 1.1;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.1;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }
    .metric-trend {
        font-size: 0.9rem;
        opacity: 0.85;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.72rem;
        opacity: 0.9;
        margin-top: 0.4rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Answer card */
    .answer-container {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1px solid #bbf7d0;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .answer-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .answer-header-text {
        font-size: 1rem;
        font-weight: 700;
        color: #166534;
    }
    .answer-text {
        font-size: 1.05rem;
        line-height: 1.75;
        color: #1e293b;
    }

    /* Source cards */
    .source-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.15rem 1.4rem;
        margin: 0.7rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.2s;
    }
    .source-card.tier-high   { border-left-color: #10b981; }
    .source-card.tier-medium { border-left-color: #f59e0b; }
    .source-card.tier-low    { border-left-color: #ef4444; }
    .source-card:hover {
        border-color: #6366f1;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
    }
    .source-card.tier-high:hover   { border-left-color: #10b981; }
    .source-card.tier-medium:hover { border-left-color: #f59e0b; }
    .source-card.tier-low:hover    { border-left-color: #ef4444; }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .source-rank {
        font-weight: 700;
        color: #6366f1;
        font-size: 0.9rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
    }
    .score-high   { background: linear-gradient(135deg, #10b981, #059669); }
    .score-medium { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .score-low    { background: linear-gradient(135deg, #ef4444, #dc2626); }
    .source-content {
        color: #334155;
        font-size: 0.96rem;
        line-height: 1.75;
        margin: 0.5rem 0;
    }
    .source-meta {
        color: #94a3b8;
        font-size: 0.78rem;
        padding-top: 0.5rem;
        border-top: 1px solid #f1f5f9;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 2rem 0 1rem;
    }
    .section-icon {
        width: 36px;
        height: 36px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .section-icon.purple { background: #ede9fe; }
    .section-icon.green  { background: #d1fae5; }
    .section-icon.blue   { background: #dbeafe; }
    .section-icon.pink   { background: #fce7f3; }
    .section-icon.amber  { background: #fef3c7; }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
    }

    /* Latency bar */
    .latency-bar {
        background: linear-gradient(135deg, #ede9fe 0%, #fae8ff 100%);
        border: 1px solid #e9d5ff;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.8rem;
        margin: 1rem 0;
    }
    .latency-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #6366f1;
    }
    .latency-label {
        color: #7c3aed;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* RRF formula box */
    .formula-box {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
    }
    .footer-text {
        color: #94a3b8;
        font-size: 0.82rem;
    }
    .footer-links {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        margin-top: 0.8rem;
    }
    .footer-link {
        color: #6366f1;
        text-decoration: none;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .footer-link:hover { text-decoration: underline; }

    /* Sidebar styling — accent left border on config cards */
    .sidebar-section {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #6366f1;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin: 0.7rem 0;
    }
    .sidebar-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .sidebar-value {
        font-size: 0.88rem;
        font-weight: 500;
        color: #1e293b;
        line-height: 1.5;
    }

    /* Sidebar status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
    .status-pill .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-pill.ready {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }
    .status-pill.ready .status-dot {
        background: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.25);
    }
    .status-pill.loading {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }
    .status-pill.loading .status-dot {
        background: #f59e0b;
        animation: pulse-dot 1.2s ease-in-out infinite;
    }
    .status-pill.error {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .status-pill.error .status-dot { background: #ef4444; }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.4; transform: scale(0.85); }
    }
    .status-row {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin: 0.5rem 0 0.75rem;
        font-size: 0.9rem;
        font-weight: 600;
        color: #1e293b;
    }

    /* Query input area */
    .query-container {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: border-color 0.2s;
    }
    .query-container:focus-within {
        border-color: #6366f1;
    }

    /* Sample question buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-size: 0.82rem !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
    }

    /* Pill-style tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: #f1f5f9;
        padding: 0.35rem;
        border-radius: 12px;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 0.5rem 1rem !important;
        background: transparent !important;
        color: #475569 !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {
        color: #6366f1 !important;
        background: rgba(99, 102, 241, 0.08) !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: white !important;
        color: #6366f1 !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ─── Pipeline State Management ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Load and initialize the RAG pipeline (cached across reruns)."""
    from src.pipeline.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(config_path=CONFIG_PATH)
    pipeline.initialize()
    return pipeline


@st.cache_resource
def ingest_documents(_pipeline):
    """Ingest documents (cached - only runs once)."""
    num_chunks = _pipeline.ingest_documents("data/raw/")
    return num_chunks


def score_tier(score: float) -> str:
    """Map a score to a visual tier class: high (>0.5), medium (0.2–0.5), low (<0.2)."""
    if score is None:
        return "tier-low"
    if score > 0.5:
        return "tier-high"
    if score >= 0.2:
        return "tier-medium"
    return "tier-low"


def get_detailed_retrieval(pipeline, query):
    """Get detailed retrieval results showing each step."""
    results = {}

    # Step 1: Encode query
    embedding_manager = pipeline._components["embedding_manager"]
    start = time.time()
    query_embedding = embedding_manager.encode_query(query)
    results["encoding_time"] = time.time() - start

    # Step 2: Dense retrieval
    vector_store = pipeline._components["vector_store"]
    start = time.time()
    dense_results = vector_store.query(
        query_embedding=query_embedding.tolist(),
        top_k=10,
    )
    results["dense_time"] = time.time() - start
    results["dense_results"] = [
        {
            "content": dense_results["documents"][i][:200],
            "score": 1 - dense_results["distances"][i],
            "metadata": dense_results["metadatas"][i] if dense_results.get("metadatas") else {},
        }
        for i in range(len(dense_results.get("documents", [])))
    ]

    # Step 3: Sparse retrieval (BM25)
    retriever = pipeline._components["retriever"]
    start = time.time()
    if retriever._bm25_index is not None:
        tokenized_query = retriever._tokenize(query)
        bm25_scores = retriever._bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:10]
        results["sparse_results"] = [
            {
                "content": retriever._corpus[idx][:200],
                "score": float(bm25_scores[idx]),
                "index": int(idx),
            }
            for idx in top_indices
            if bm25_scores[idx] > 0
        ]
    else:
        results["sparse_results"] = []
    results["sparse_time"] = time.time() - start

    # Step 4: Hybrid retrieval with RRF
    start = time.time()
    hybrid_results = retriever.retrieve(query, top_k=10)
    results["fusion_time"] = time.time() - start
    results["fused_results"] = hybrid_results

    # Step 5: Generate answer
    context = pipeline._build_context(hybrid_results[:5])
    start = time.time()
    answer = pipeline._generate_answer(query, context)
    results["generation_time"] = time.time() - start
    results["answer"] = answer
    results["context"] = context

    results["total_time"] = (
        results["encoding_time"]
        + results["dense_time"]
        + results["sparse_time"]
        + results["fusion_time"]
        + results["generation_time"]
    )

    return results


# ─── Sidebar ──────────────────────────────────────────────────────────────────────────

import yaml
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

with st.sidebar:
    st.markdown("## ⚡ Pipeline Config")

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-label">Embedding Model</div>
        <div class="sidebar-value">{config["embedding"]["model_name"].split("/")[-1]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-label">LLM</div>
        <div class="sidebar-value">{config['llm']['provider'].title()} / {config['llm']['model_name']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-label">Vector Store</div>
        <div class="sidebar-value">{config["vectorstore"]["backend"].title()} ({config["vectorstore"]["chroma"]["distance_metric"]})</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-label">Retrieval Strategy</div>
        <div class="sidebar-value">Hybrid (Dense + BM25) → RRF (k={config["retrieval"]["fusion"]["rrf_k"]})</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-label">Chunking</div>
        <div class="sidebar-value">{config["chunking"]["strategy"].title()} (threshold: {config["chunking"]["semantic"]["breakpoint_threshold"]})</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Show a loading pill while the pipeline initialises, then swap to ready/error.
    status_slot = st.empty()
    status_slot.markdown(
        '<div class="status-row">Pipeline status'
        '<span class="status-pill loading"><span class="status-dot"></span>Loading</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Initialize pipeline
    with st.spinner("Initializing pipeline..."):
        try:
            pipeline = load_pipeline()
            num_chunks = ingest_documents(pipeline)
            stats = pipeline.get_pipeline_stats()
            doc_count = stats.get("vector_store", {}).get("document_count", 0)
            status_slot.markdown(
                '<div class="status-row">Pipeline ready'
                '<span class="status-pill ready"><span class="status-dot"></span>Ready</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            col_a, col_b = st.columns(2)
            col_a.metric("Chunks", num_chunks)
            col_b.metric("Documents", doc_count)
        except Exception as e:
            status_slot.markdown(
                '<div class="status-row">Pipeline status'
                '<span class="status-pill error"><span class="status-dot"></span>Error</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.error(f"Pipeline error: {str(e)[:80]}")
            pipeline = None

    st.markdown("---")
    st.markdown("##### Built by [Rohit Manne](https://github.com/rohitkumarmanne-442)")
    if IS_CLOUD:
        st.caption("☁️ Cloud mode (Groq API)")
    else:
        st.caption("🖥️ Local mode (Ollama)")


# ─── Hero Section ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-container">
    <div class="hero-title">Advanced RAG Pipeline</div>
    <div class="hero-subtitle">Production-grade Retrieval-Augmented Generation for SEC Financial Filings</div>
    <div class="hero-badge">⚡ Hybrid Retrieval + Reciprocal Rank Fusion</div>
</div>
""", unsafe_allow_html=True)

# Pipeline flow
st.markdown("""
<div class="pipeline-container">
    <div class="pipeline-flow">
        <span class="flow-step">📄 Ingest</span>
        <span class="flow-arrow">›</span>
        <span class="flow-step">✂️ Semantic Chunk</span>
        <span class="flow-arrow">›</span>
        <span class="flow-step">🧢 Embed (384d)</span>
        <span class="flow-arrow">›</span>
        <span class="flow-step">🔎 Dense + Sparse</span>
        <span class="flow-arrow">›</span>
        <span class="flow-step">⚡ RRF Fusion</span>
        <span class="flow-arrow">›</span>
        <span class="flow-step">🤖 Generate</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Query Section ──────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="section-header">
    <div class="section-icon purple">💬</div>
    <div class="section-title">Query the Pipeline</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Enter your question about the SEC filings:",
        placeholder="Ask anything about the SEC financial filings...",
        label_visibility="collapsed",
    )
with col2:
    search_clicked = st.button("⚡ Search", type="primary", use_container_width=True)

# Sample questions
st.markdown("")
sample_cols = st.columns(4)
sample_questions = [
    "What was the total revenue for FY2023?",
    "What are the supply chain risk factors?",
    "How did R&D spending change?",
    "What is the company's market position?",
]
for i, sq in enumerate(sample_questions):
    if sample_cols[i].button(sq, key=f"sample_{i}", use_container_width=True):
        query = sq
        search_clicked = True

# ─── Results Section ──────────────────────────────────────────────────────────────────────────

if search_clicked and query and pipeline:

    # Process query with detailed tracking
    with st.spinner(""):
        progress_bar = st.progress(0, text="Encoding query...")
        results = {}

        # Step 1: Encode query
        embedding_manager = pipeline._components["embedding_manager"]
        start = time.time()
        query_embedding = embedding_manager.encode_query(query)
        results["encoding_time"] = time.time() - start
        progress_bar.progress(20, text="Running dense retrieval...")

        # Step 2: Dense retrieval
        vector_store = pipeline._components["vector_store"]
        start = time.time()
        dense_results = vector_store.query(query_embedding=query_embedding.tolist(), top_k=10)
        results["dense_time"] = time.time() - start
        results["dense_results"] = [
            {
                "content": dense_results["documents"][i][:200],
                "score": 1 - dense_results["distances"][i],
                "metadata": dense_results["metadatas"][i] if dense_results.get("metadatas") else {},
            }
            for i in range(len(dense_results.get("documents", [])))
        ]
        progress_bar.progress(40, text="Running sparse retrieval (BM25)...")

        # Step 3: Sparse retrieval
        retriever = pipeline._components["retriever"]
        start = time.time()
        if retriever._bm25_index is not None:
            tokenized_query = retriever._tokenize(query)
            bm25_scores = retriever._bm25_index.get_scores(tokenized_query)
            top_indices = np.argsort(bm25_scores)[::-1][:10]
            results["sparse_results"] = [
                {"content": retriever._corpus[idx][:200], "score": float(bm25_scores[idx]), "index": int(idx)}
                for idx in top_indices if bm25_scores[idx] > 0
            ]
        else:
            results["sparse_results"] = []
        results["sparse_time"] = time.time() - start
        progress_bar.progress(60, text="Applying Reciprocal Rank Fusion...")

        # Step 4: RRF Fusion
        start = time.time()
        hybrid_results = retriever.retrieve(query, top_k=10)
        results["fusion_time"] = time.time() - start
        results["fused_results"] = hybrid_results
        progress_bar.progress(80, text="Generating answer with LLM...")

        # Step 5: LLM Generation
        context = pipeline._build_context(hybrid_results[:5])
        start = time.time()
        answer = pipeline._generate_answer(query, context)
        results["generation_time"] = time.time() - start
        results["answer"] = answer
        results["context"] = context

        results["total_time"] = (
            results["encoding_time"] + results["dense_time"] +
            results["sparse_time"] + results["fusion_time"] + results["generation_time"]
        )
        progress_bar.progress(100, text="Done!")
        time.sleep(0.3)
        progress_bar.empty()

    # ─── Timing Metrics ──────────────────────────────────────────────────────────────────

    colors = ["", "green", "cyan", "pink", "amber"]
    step_names = ["Encoding", "Dense Search", "BM25 Search", "RRF Fusion", "LLM Gen"]
    step_times = [
        results["encoding_time"], results["dense_time"],
        results["sparse_time"], results["fusion_time"], results["generation_time"]
    ]

    st.markdown(f"""
    <div class="latency-bar">
        <span class="latency-label">Total Pipeline Latency</span>
        <span class="latency-value">{results['total_time']:.2f}s</span>
    </div>
    """, unsafe_allow_html=True)

    # Median timing = simple baseline for a trend arrow (▲ slower / ▼ faster).
    median_time = float(np.median(step_times)) if step_times else 0.0
    metrics_html = '<div class="metrics-grid">'
    for name, t, color in zip(step_names, step_times, colors):
        if t > median_time * 1.15:
            arrow = "▲"
        elif t < median_time * 0.85:
            arrow = "▼"
        else:
            arrow = "▶"
        metrics_html += f"""
        <div class="metric-card {color}">
            <div class="metric-value-row">
                <span class="metric-trend">{arrow}</span>
                <span class="metric-value">{t:.3f}s</span>
            </div>
            <div class="metric-label" title="{name}">{name}</div>
        </div>"""
    metrics_html += '</div>'
    st.markdown(metrics_html, unsafe_allow_html=True)

    st.markdown("")

    # ─── Answer Section ───────────────────────────────────────────────────────────────────

    st.markdown(f"""
    <div class="answer-container">
        <div class="answer-header">
            <span style="font-size:1.3rem;">🤖</span>
            <span class="answer-header-text">Generated Answer</span>
        </div>
        <div class="answer-text">{results['answer']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Retrieval Deep Dive ──────────────────────────────────────────────────────────────────

    st.markdown("""
    <div class="section-header">
        <div class="section-icon blue">📊</div>
        <div class="section-title">Retrieval Deep Dive</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⚡ Fused Results (RRF)", "🎯 Dense (Semantic)", "📝 Sparse (BM25)"])

    with tab1:
        if results["fused_results"]:
            for i, result in enumerate(results["fused_results"][:5]):
                score = result.get("score", 0)
                score_class = "score-high" if score > 0.5 else "score-medium" if score >= 0.2 else "score-low"
                tier_class = score_tier(score)

                st.markdown(f"""
                <div class="source-card {tier_class}">
                    <div class="source-header">
                        <span class="source-rank">#{i+1}</span>
                        <span class="score-badge {score_class}">RRF: {score:.4f}</span>
                    </div>
                    <div class="source-content">{result['content'][:300]}...</div>
                    <div class="source-meta">
                        Strategy: {result.get('source', 'rrf_fusion')} &nbsp;|&nbsp;
                        Section: {result.get('metadata', {}).get('section', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No fused results found.")

    with tab2:
        if results["dense_results"]:
            dense_scores = [r["score"] for r in results["dense_results"]]
            chart_df = pd.DataFrame({
                "Document": [f"Doc {i+1}" for i in range(len(dense_scores))],
                "Cosine Similarity": dense_scores,
            })
            st.bar_chart(chart_df.set_index("Document"), color="#6366f1")

            for i, result in enumerate(results["dense_results"][:5]):
                tier_class = score_tier(result["score"])
                st.markdown(f"""
                <div class="source-card {tier_class}">
                    <div class="source-header">
                        <span class="source-rank">#{i+1}</span>
                        <span class="score-badge score-high">Sim: {result['score']:.4f}</span>
                    </div>
                    <div class="source-content">{result['content'][:200]}...</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No dense results.")

    with tab3:
        if results["sparse_results"]:
            sparse_scores = [r["score"] for r in results["sparse_results"]]
            chart_df = pd.DataFrame({
                "Document": [f"Doc {i+1}" for i in range(len(sparse_scores))],
                "BM25 Score": sparse_scores,
            })
            st.bar_chart(chart_df.set_index("Document"), color="#ec4899")

            # Normalise BM25 to a 0–1 tier band for consistent visual coding.
            max_bm25 = max(sparse_scores) if sparse_scores else 1.0
            for i, result in enumerate(results["sparse_results"][:5]):
                normalised = (result["score"] / max_bm25) if max_bm25 else 0.0
                tier_class = score_tier(normalised)
                st.markdown(f"""
                <div class="source-card {tier_class}">
                    <div class="source-header">
                        <span class="source-rank">#{i+1}</span>
                        <span class="score-badge score-medium">BM25: {result['score']:.2f}</span>
                    </div>
                    <div class="source-content">{result['content'][:200]}...</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No sparse results. BM25 index may not be built.")

    # ─── RRF Fusion Explanation ──────────────────────────────────────────────────────────────────

    st.markdown("""
    <div class="section-header">
        <div class="section-icon pink">⚡</div>
        <div class="section-title">Reciprocal Rank Fusion</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <strong>How it works:</strong> RRF combines rankings from multiple retrieval strategies into a single unified ranking.
        Unlike score-based fusion, RRF is robust to differences in score scales between dense and sparse retrievers.
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)} \quad \text{where } k = 60")

    if results["fused_results"]:
        fusion_data = []
        for i, r in enumerate(results["fused_results"][:7]):
            fusion_data.append({
                "Rank": i + 1,
                "Content Preview": r["content"][:100] + "...",
                "RRF Score": f"{r.get('score', 0):.4f}",
                "Source": r.get("source", "fusion"),
            })

        st.dataframe(
            pd.DataFrame(fusion_data),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "RRF Score": st.column_config.TextColumn("RRF Score", width="small"),
                "Source": st.column_config.TextColumn("Source", width="small"),
            }
        )

    # ─── Context Window ─────────────────────────────────────────────────────────────────────────

    st.markdown("""
    <div class="section-header">
        <div class="section-icon green">📖</div>
        <div class="section-title">Context Window (sent to LLM)</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View full context sent to the language model", expanded=False):
        st.code(results["context"], language=None)

elif search_clicked and not query:
    st.warning("Please enter a question to search.")

# ─── Contact Form ────────────────────────────────────────────────────────────────────────

try:
    from src.contact.ui import render_contact_form
    render_contact_form()
except Exception as _contact_exc:  # pragma: no cover - defensive: never break the page
    st.warning(f"Contact form unavailable: {_contact_exc}")

# ─── Footer ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    <div class="footer-text">
        <strong>Advanced RAG Pipeline</strong> &mdash; Built with LlamaIndex &bull; ChromaDB &bull; HuggingFace &bull; Groq<br>
        Semantic Chunking &bull; Hybrid Retrieval &bull; Reciprocal Rank Fusion &bull; Production-grade Evaluation
    </div>
    <div class="footer-links">
        <a class="footer-link" href="https://github.com/rohitkumarmanne-442/Advanced-RAG-Pipeline-for-Industry-Specific-Data" target="_blank">GitHub Repository</a>
        <a class="footer-link" href="https://github.com/rohitkumarmanne-442" target="_blank">About the Author</a>
    </div>
</div>
""", unsafe_allow_html=True)
