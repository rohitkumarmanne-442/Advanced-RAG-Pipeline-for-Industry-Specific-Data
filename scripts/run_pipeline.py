"""
Interactive RAG Pipeline Script
Run this to query the pipeline interactively or with single questions.

Usage:
    python scripts/run_pipeline.py --query "What was the total revenue?"
    python scripts/run_pipeline.py --interactive
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.pipeline.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single question to ask",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking",
    )
    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    # Initialize pipeline
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(config_path=args.config)
    pipeline.initialize()
    print("Pipeline ready!\n")

    if args.query:
        # Single query mode
        response = pipeline.query(
            args.query,
            top_k=args.top_k,
            use_reranker=not args.no_rerank,
        )
        _print_response(response)

    elif args.interactive:
        # Interactive mode
        print("Interactive RAG Pipeline (type 'quit' to exit)")
        print("-" * 50)

        while True:
            try:
                question = input("\nQuestion: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    break
                if not question:
                    continue

                response = pipeline.query(
                    question,
                    top_k=args.top_k,
                    use_reranker=not args.no_rerank,
                )
                _print_response(response)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        parser.print_help()


def _print_response(response):
    """Pretty-print a RAG response."""
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("-" * 60)
    print(response.answer)
    print("\n" + "-" * 60)
    print(f"Sources: {len(response.source_documents)} documents retrieved")

    for i, doc in enumerate(response.source_documents[:3]):
        source = doc.get("metadata", {}).get("source", "Unknown")
        score = doc.get("score", 0)
        print(f"  [{i+1}] {Path(source).name} (score: {score:.3f})")

    print("=" * 60)


if __name__ == "__main__":
    main()
