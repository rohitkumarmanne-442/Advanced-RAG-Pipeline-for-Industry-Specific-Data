from setuptools import setup, find_packages

setup(
    name="advanced-rag-pipeline",
    version="1.0.0",
    description="Production-grade RAG pipeline for industry-specific document analysis",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "llama-index>=0.11.0",
        "sentence-transformers>=3.0.0",
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "chromadb>=0.5.0",
        "faiss-cpu>=1.7.0",
        "unstructured[pdf]>=0.15.0",
        "ragas>=0.2.0",
        "pyyaml>=6.0",
        "pydantic>=2.0",
        "loguru>=0.7.0",
        "rank-bm25>=0.2.2",
    ],
)
