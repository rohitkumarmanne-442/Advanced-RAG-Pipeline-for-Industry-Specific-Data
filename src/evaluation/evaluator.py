"""
RAG Evaluator Module
Comprehensive evaluation of RAG pipeline quality using the
Ragas framework to measure faithfulness, relevancy, and hallucination.
"""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from loguru import logger


class RAGEvaluator:
    """
    Evaluates RAG pipeline performance using multiple metrics:
    - Faithfulness: Does the answer align with retrieved context?
    - Answer Relevancy: Is the answer relevant to the question?
    - Context Precision: Are the retrieved docs precise?
    - Context Recall: Are all relevant docs retrieved?

    Also computes custom metrics:
    - Hallucination Rate: % of claims not supported by context
    - Citation Accuracy: % of citations that are correct
    """

    def __init__(self, config: dict):
        eval_config = config.get("evaluation", {})
        self.metrics_list = eval_config.get(
            "metrics",
            ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        )
        self.ground_truth_path = eval_config.get(
            "ground_truth_path", "data/ground_truth/qa_pairs.json"
        )
        self.num_samples = eval_config.get("num_samples", 50)
        self.hallucination_threshold = eval_config.get("hallucination_threshold", 0.05)
        logger.info(f"RAGEvaluator initialized with metrics: {self.metrics_list}")

    def evaluate(
        self,
        pipeline,
        test_dataset: Optional[list[dict]] = None,
        output_path: Optional[str] = None,
    ) -> dict:
        """
        Run full evaluation of the RAG pipeline.

        Args:
            pipeline: The RAGPipeline instance to evaluate.
            test_dataset: Optional list of QA pairs. Loaded from config path if None.
            output_path: Optional path to save evaluation results.

        Returns:
            Dict with metric scores and detailed results.
        """
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        # Load test dataset
        if test_dataset is None:
            test_dataset = self._load_ground_truth()

        if not test_dataset:
            raise ValueError("No test dataset available for evaluation")

        # Run pipeline on test questions
        evaluation_data = self._generate_evaluation_data(pipeline, test_dataset)

        # Prepare Ragas dataset
        ragas_dataset = Dataset.from_dict(
            {
                "question": [d["question"] for d in evaluation_data],
                "answer": [d["answer"] for d in evaluation_data],
                "contexts": [d["contexts"] for d in evaluation_data],
                "ground_truth": [d["ground_truth"] for d in evaluation_data],
            }
        )

        # Select metrics
        metrics = self._get_metrics()

        # Run Ragas evaluation
        logger.info(f"Running Ragas evaluation on {len(evaluation_data)} samples...")
        results = evaluate(ragas_dataset, metrics=metrics)

        # Compute additional custom metrics
        custom_metrics = self._compute_custom_metrics(evaluation_data)

        # Compile final results
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(evaluation_data),
            "ragas_scores": {k: float(v) for k, v in results.items()},
            "custom_metrics": custom_metrics,
            "hallucination_rate": custom_metrics.get("hallucination_rate", 0),
            "passes_threshold": custom_metrics.get("hallucination_rate", 1)
            < self.hallucination_threshold,
            "detailed_results": evaluation_data,
        }

        # Save results
        if output_path:
            self._save_results(final_results, output_path)

        logger.info(
            f"Evaluation complete. Faithfulness: {results.get('faithfulness', 'N/A'):.3f}, "
            f"Relevancy: {results.get('answer_relevancy', 'N/A'):.3f}"
        )
        return final_results

    def evaluate_retrieval_only(
        self, pipeline, test_dataset: Optional[list[dict]] = None
    ) -> dict:
        """
        Evaluate only the retrieval component (without LLM generation).
        Useful for testing chunking/embedding/retrieval improvements.
        """
        if test_dataset is None:
            test_dataset = self._load_ground_truth()

        results = {
            "precision_at_k": [],
            "recall_at_k": [],
            "mrr": [],
            "ndcg": [],
        }

        for sample in test_dataset:
            question = sample["question"]
            expected_contexts = sample.get("expected_contexts", [])

            # Get retrieval results
            retriever = pipeline._components.get("retriever")
            if retriever:
                retrieved = retriever.retrieve(question)
                retrieved_texts = [r["content"] for r in retrieved]

                # Compute metrics
                precision = self._precision_at_k(retrieved_texts, expected_contexts)
                recall = self._recall_at_k(retrieved_texts, expected_contexts)
                mrr = self._mean_reciprocal_rank(retrieved_texts, expected_contexts)

                results["precision_at_k"].append(precision)
                results["recall_at_k"].append(recall)
                results["mrr"].append(mrr)

        # Average metrics
        avg_results = {}
        for metric, values in results.items():
            if values:
                avg_results[f"avg_{metric}"] = sum(values) / len(values)

        logger.info(f"Retrieval evaluation: {avg_results}")
        return avg_results

    def _generate_evaluation_data(
        self, pipeline, test_dataset: list[dict]
    ) -> list[dict]:
        """Generate evaluation data by running pipeline on test questions."""
        evaluation_data = []

        for i, sample in enumerate(test_dataset[: self.num_samples]):
            question = sample["question"]
            ground_truth = sample["answer"]

            try:
                response = pipeline.query(question)

                evaluation_data.append(
                    {
                        "question": question,
                        "answer": response.answer,
                        "contexts": [
                            doc["content"] for doc in response.source_documents
                        ],
                        "ground_truth": ground_truth,
                        "retrieval_scores": response.retrieval_scores,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to process question {i}: {e}")
                evaluation_data.append(
                    {
                        "question": question,
                        "answer": f"Error: {str(e)}",
                        "contexts": [],
                        "ground_truth": ground_truth,
                        "retrieval_scores": [],
                    }
                )

        return evaluation_data

    def _get_metrics(self):
        """Get Ragas metric objects based on configuration."""
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }

        return [metric_map[m] for m in self.metrics_list if m in metric_map]

    def _compute_custom_metrics(self, evaluation_data: list[dict]) -> dict:
        """Compute custom evaluation metrics beyond Ragas."""
        hallucination_count = 0
        total_claims = 0
        empty_responses = 0

        for item in evaluation_data:
            answer = item["answer"]
            contexts = item["contexts"]

            if not answer or answer.startswith("Error"):
                empty_responses += 1
                continue

            # Simple hallucination detection: check if answer claims
            # are grounded in context
            claims = self._extract_claims(answer)
            total_claims += len(claims)

            for claim in claims:
                if not self._is_grounded(claim, contexts):
                    hallucination_count += 1

        hallucination_rate = (
            hallucination_count / total_claims if total_claims > 0 else 0
        )

        return {
            "hallucination_rate": hallucination_rate,
            "total_claims_evaluated": total_claims,
            "ungrounded_claims": hallucination_count,
            "empty_responses": empty_responses,
            "response_rate": 1 - (empty_responses / len(evaluation_data))
            if evaluation_data
            else 0,
        }

    def _extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from generated text."""
        import re

        # Split into sentences as proxy for claims
        sentences = re.split(r"[.!?]+", text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        return claims

    def _is_grounded(self, claim: str, contexts: list[str]) -> bool:
        """Check if a claim is supported by any context passage."""
        claim_lower = claim.lower()
        # Check keyword overlap as simple grounding heuristic
        claim_words = set(claim_lower.split())
        significant_words = {w for w in claim_words if len(w) > 4}

        for context in contexts:
            context_lower = context.lower()
            context_words = set(context_lower.split())

            # If >50% of significant claim words appear in context
            if significant_words:
                overlap = significant_words.intersection(context_words)
                if len(overlap) / len(significant_words) > 0.5:
                    return True

        return False

    def _precision_at_k(
        self, retrieved: list[str], relevant: list[str], k: int = 5
    ) -> float:
        """Compute Precision@K."""
        if not retrieved or not relevant:
            return 0.0

        retrieved_k = retrieved[:k]
        hits = sum(
            1 for doc in retrieved_k if any(rel in doc for rel in relevant)
        )
        return hits / len(retrieved_k)

    def _recall_at_k(
        self, retrieved: list[str], relevant: list[str], k: int = 5
    ) -> float:
        """Compute Recall@K."""
        if not relevant:
            return 0.0

        retrieved_k = retrieved[:k]
        hits = sum(
            1 for rel in relevant if any(rel in doc for doc in retrieved_k)
        )
        return hits / len(relevant)

    def _mean_reciprocal_rank(
        self, retrieved: list[str], relevant: list[str]
    ) -> float:
        """Compute Mean Reciprocal Rank (MRR)."""
        for i, doc in enumerate(retrieved):
            if any(rel in doc for rel in relevant):
                return 1.0 / (i + 1)
        return 0.0

    def _load_ground_truth(self) -> list[dict]:
        """Load ground truth QA pairs from file."""
        path = Path(self.ground_truth_path)
        if not path.exists():
            logger.warning(f"Ground truth file not found: {self.ground_truth_path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} ground truth QA pairs")
        return data

    def _save_results(self, results: dict, path: str):
        """Save evaluation results to JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Evaluation results saved to {output_path}")
