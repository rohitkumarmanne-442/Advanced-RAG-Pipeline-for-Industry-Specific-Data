"""
Pipeline Evaluation Script
Run this to evaluate the RAG pipeline against ground truth.

Usage:
    python scripts/evaluate.py --config config/settings.yaml --output results/evaluation.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.evaluator import RAGEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate only retrieval (no LLM generation)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate",
    )
    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/evaluation.log", rotation="10 MB", level="DEBUG")

    logger.info("Starting RAG pipeline evaluation...")

    # Initialize pipeline
    pipeline = RAGPipeline(config_path=args.config)
    pipeline.initialize()

    # Initialize evaluator
    config = pipeline.config
    if args.num_samples:
        config["evaluation"]["num_samples"] = args.num_samples

    evaluator = RAGEvaluator(config)

    if args.retrieval_only:
        logger.info("Running retrieval-only evaluation...")
        results = evaluator.evaluate_retrieval_only(pipeline)
    else:
        logger.info("Running full pipeline evaluation...")
        results = evaluator.evaluate(pipeline, output_path=args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if "ragas_scores" in results:
        print("\nRagas Metrics:")
        for metric, score in results["ragas_scores"].items():
            print(f"  {metric}: {score:.4f}")

    if "custom_metrics" in results:
        print("\nCustom Metrics:")
        for metric, value in results["custom_metrics"].items():
            print(f"  {metric}: {value}")

    if "passes_threshold" in results:
        status = "PASS" if results["passes_threshold"] else "FAIL"
        print(f"\nHallucination Threshold: {status}")

    print("=" * 60)


if __name__ == "__main__":
    main()
