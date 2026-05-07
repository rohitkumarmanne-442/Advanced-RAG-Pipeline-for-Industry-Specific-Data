"""
Advanced RAG Pipeline - Visual Web Application
Streamlit-based interface that visualizes the entire RAG pipeline process:
- Document ingestion status
- Query processing steps
- Dense vs Sparse retrieval comparison
- Reciprocal Rank Fusion visualization
- Source document highlighting
- Pipeline performance metrics
"""

import streamlit as st
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from loguru import logger

# Suppress loguru output in web app
logger.remove()

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Advanced RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1a73e8, #8e24aa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .step-box {
        border-left: 4px solid #1a73e8;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
        border-radius: 0 8px 8px 0;
    }
    .step-box-active {
        border-left: 4px solid #34a853;
        background: #e8f5e9;
    }
    .source-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }
    .score-high { background: #34a853; }
    .score-medium { background: #fbbc04; }
    .score-low { background: #ea4335; }
    .pipeline-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 1rem;
        flex-wrap: wrap;
    }
    .flow-step {
        background: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #1565c0;
    }
    .flow-arrow {
        color: #90a4ae;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Pipeline State Management ────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Load and initialize the RAG pipeline (cached across reruns)."""
    from src.pipeline.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(config_path="config/settings.yaml")
    pipeline.initialize()
    return pipeline


@st.cache_resource
def ingest_documents(_pipeline):
    """Ingest documents (cached - only runs once)."""
    num_chunks = _pipeline.ingest_documents("data/raw/")
    return num_chunks


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


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Configuration")

    st.markdown("---")

    # Load config
    import yaml
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    st.markdown("**Embedding Model**")
    st.code(config["embedding"]["model_name"], language=None)

    st.markdown("**LLM Provider**")
    st.code(f"{config['llm']['provider']} / {config['llm']['model_name']}", language=None)

    st.markdown("**Vector Store**")
    st.code(config["vectorstore"]["backend"], language=None)

    st.markdown("**Chunking Strategy**")
    st.code(config["chunking"]["strategy"], language=None)

    st.markdown("**Fusion Method**")
    st.code(config["retrieval"]["fusion"]["method"], language=None)

    st.markdown("---")

    st.markdown("## 📊 Pipeline Status")

    # Initialize pipeline
    with st.spinner("Loading pipeline..."):
        try:
            pipeline = load_pipeline()
            num_chunks = ingest_documents(pipeline)
            st.success(f"✅ Pipeline ready ({num_chunks} chunks indexed)")
        except Exception as e:
            st.error(f"❌ Pipeline error: {str(e)[:100]}")
            pipeline = None

    if pipeline:
        stats = pipeline.get_pipeline_stats()
        if "vector_store" in stats:
            st.metric("Documents Indexed", stats["vector_store"].get("document_count", 0))


# ─── Main Content ────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🔍 Advanced RAG Pipeline</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Production-grade Retrieval-Augmented Generation for SEC Financial Filings</p>',
    unsafe_allow_html=True,
)

# Pipeline flow visualization
st.markdown("""
<div class="pipeline-flow">
    <span class="flow-step">📄 Document Ingestion</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">✂️ Semantic Chunking</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">🧮 Embedding</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">🔎 Hybrid Retrieval</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">⚡ RRF Fusion</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">🤖 LLM Generation</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Query Section ────────────────────────────────────────────────────────────

st.markdown("### 💬 Ask a Question")

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Enter your question about the SEC filings:",
        placeholder="e.g., What was the total revenue for FY2023?",
        label_visibility="collapsed",
    )
with col2:
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

# Sample questions
st.markdown("**Try these:**")
sample_cols = st.columns(3)
sample_questions = [
    "What was the total revenue for FY2023?",
    "What are the supply chain risk factors?",
    "How did R&D spending change between 2022 and 2023?",
]
for i, sq in enumerate(sample_questions):
    if sample_cols[i].button(sq, key=f"sample_{i}", use_container_width=True):
        query = sq
        search_clicked = True

# ─── Results Section ──────────────────────────────────────────────────────────

if search_clicked and query and pipeline:
    st.markdown("---")

    # Process query with detailed tracking
    with st.spinner("Processing query through RAG pipeline..."):
        results = get_detailed_retrieval(pipeline, query)

    # ─── Step-by-Step Visualization ───────────────────────────────────────

    st.markdown("### 📋 Pipeline Execution Steps")

    steps = [
        ("1️⃣ Query Encoding", f"Encoded query to {config['embedding'].get('dimension', 384)}-dim vector", results["encoding_time"]),
        ("2️⃣ Dense Retrieval", f"Found {len(results['dense_results'])} results via semantic search", results["dense_time"]),
        ("3️⃣ Sparse Retrieval (BM25)", f"Found {len(results['sparse_results'])} results via keyword match", results["sparse_time"]),
        ("4️⃣ Reciprocal Rank Fusion", f"Fused into {len(results['fused_results'])} ranked results", results["fusion_time"]),
        ("5️⃣ LLM Generation", f"Generated answer using {config['llm']['model_name']}", results["generation_time"]),
    ]

    step_cols = st.columns(5)
    for i, (step_name, desc, duration) in enumerate(steps):
        with step_cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{duration:.2f}s</div>
                <div class="metric-label">{step_name}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Total time
    st.info(f"⏱️ **Total pipeline latency: {results['total_time']:.2f}s**")

    st.markdown("---")

    # ─── Answer Section ───────────────────────────────────────────────────

    st.markdown("### 🤖 Generated Answer")
    st.markdown(f"> {results['answer']}")

    st.markdown("---")

    # ─── Retrieval Comparison ─────────────────────────────────────────────

    st.markdown("### 📊 Retrieval Strategy Comparison")

    tab1, tab2, tab3 = st.tabs(["🔀 Fused Results (RRF)", "🎯 Dense Retrieval", "📝 Sparse (BM25)"])

    with tab1:
        if results["fused_results"]:
            for i, result in enumerate(results["fused_results"][:5]):
                score = result.get("score", 0)
                score_class = "score-high" if score > 0.5 else "score-medium" if score > 0.2 else "score-low"

                st.markdown(f"""
                <div class="source-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong>Result #{i+1}</strong>
                        <span class="score-badge {score_class}">RRF Score: {score:.4f}</span>
                    </div>
                    <p style="margin-top:0.5rem; color:#555; font-size:0.9rem;">{result['content'][:300]}...</p>
                    <small style="color:#888;">Source: {result.get('source', 'rrf_fusion')} | Section: {result.get('metadata', {}).get('section', 'N/A')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No fused results found.")

    with tab2:
        if results["dense_results"]:
            # Score distribution chart
            dense_scores = [r["score"] for r in results["dense_results"]]
            chart_df = pd.DataFrame({
                "Document": [f"Doc {i+1}" for i in range(len(dense_scores))],
                "Cosine Similarity": dense_scores,
            })
            st.bar_chart(chart_df.set_index("Document"))

            for i, result in enumerate(results["dense_results"][:5]):
                st.markdown(f"**[{i+1}]** (sim: {result['score']:.4f}) {result['content'][:150]}...")
        else:
            st.warning("No dense results.")

    with tab3:
        if results["sparse_results"]:
            sparse_scores = [r["score"] for r in results["sparse_results"]]
            chart_df = pd.DataFrame({
                "Document": [f"Doc {i+1}" for i in range(len(sparse_scores))],
                "BM25 Score": sparse_scores,
            })
            st.bar_chart(chart_df.set_index("Document"))

            for i, result in enumerate(results["sparse_results"][:5]):
                st.markdown(f"**[{i+1}]** (BM25: {result['score']:.2f}) {result['content'][:150]}...")
        else:
            st.warning("No sparse results. BM25 index may not be built.")

    st.markdown("---")

    # ─── Fusion Visualization ─────────────────────────────────────────────

    st.markdown("### ⚡ Reciprocal Rank Fusion Breakdown")

    st.markdown("""
    **RRF Formula:** $\\text{RRF}(d) = \\sum_{r \\in R} \\frac{1}{k + \\text{rank}_r(d)}$ where $k = 60$

    This combines rankings from both dense (semantic) and sparse (keyword) retrieval,
    producing a unified ranking that is robust to score scale differences.
    """)

    if results["fused_results"]:
        fusion_data = []
        for i, r in enumerate(results["fused_results"][:7]):
            fusion_data.append({
                "Rank": i + 1,
                "Content Preview": r["content"][:80] + "...",
                "RRF Score": f"{r.get('score', 0):.4f}",
                "Source Strategy": r.get("source", "fusion"),
            })

        st.dataframe(
            pd.DataFrame(fusion_data),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ─── Context Window ───────────────────────────────────────────────────

    st.markdown("### 📖 Context Sent to LLM")
    with st.expander("View full context window", expanded=False):
        st.text(results["context"])

elif search_clicked and not query:
    st.warning("Please enter a question.")

# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:1rem;">
    <strong>Advanced RAG Pipeline</strong> | Built with LlamaIndex, ChromaDB, HuggingFace, Ragas<br>
    Semantic Chunking • Reciprocal Rank Fusion • Cross-Encoder Reranking • Hallucination Detection
</div>
""", unsafe_allow_html=True)
