"""
Document Ingestion Script
Run this to ingest documents into the RAG pipeline.

Usage:
    python scripts/ingest.py --source data/raw/ --config config/settings.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.pipeline.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG pipeline")
    parser.add_argument(
        "--source",
        type=str,
        default="data/raw",
        help="Path to source documents directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add("logs/ingestion.log", rotation="10 MB", level="DEBUG")

    logger.info(f"Starting document ingestion from: {args.source}")
    logger.info(f"Using config: {args.config}")

    # Initialize pipeline
    pipeline = RAGPipeline(config_path=args.config)
    pipeline.initialize()

    # Ingest documents
    num_chunks = pipeline.ingest_documents(args.source)

    logger.info(f"Ingestion complete! Indexed {num_chunks} chunks.")
    logger.info(f"Pipeline stats: {pipeline.get_pipeline_stats()}")


if __name__ == "__main__":
    main()
