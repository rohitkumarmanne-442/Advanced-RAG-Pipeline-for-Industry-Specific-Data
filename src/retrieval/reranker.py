"""
Reranker Module
Implements cross-encoder based reranking to improve retrieval
precision after initial candidate retrieval.
"""

from typing import Optional

import numpy as np
from loguru import logger


class Reranker:
    """
    Cross-encoder reranker that scores query-document pairs
    for more accurate relevance estimation than bi-encoder similarity.

    Uses models like BAAI/bge-reranker-large for state-of-the-art
    reranking performance.
    """

    def __init__(self, config: dict):
        reranker_config = config.get("retrieval", {}).get("reranker", {})
        self.enabled = reranker_config.get("enabled", True)
        self.model_name = reranker_config.get("model", "BAAI/bge-reranker-large")
        self.top_n = reranker_config.get("top_n", 5)
        self._model = None
        logger.info(f"Reranker configured: model={self.model_name}, enabled={self.enabled}")

    @property
    def model(self):
        """Lazy-load the cross-encoder reranker model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info(f"Loaded reranker model: {self.model_name}")
        return self._model

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_n: Optional[int] = None,
    ) -> list[dict]:
        """
        Rerank retrieval results using cross-encoder scoring.

        The cross-encoder processes (query, document) pairs jointly,
        enabling deeper semantic understanding compared to bi-encoder
        cosine similarity.

        Args:
            query: The search query.
            results: List of retrieval results with 'content' key.
            top_n: Number of top results to return after reranking.

        Returns:
            Reranked results with updated scores.
        """
        if not self.enabled or not results:
            return results

        n = top_n or self.top_n

        # Create query-document pairs for cross-encoder
        pairs = [[query, result["content"]] for result in results]

        # Score all pairs
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        reranked = []
        for result, score in zip(results, scores):
            reranked_result = result.copy()
            reranked_result["rerank_score"] = float(score)
            reranked_result["original_score"] = result.get("score", 0)
            reranked.append(reranked_result)

        # Sort by reranker score (descending)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Update ranks
        for i, result in enumerate(reranked):
            result["rank"] = i + 1
            result["score"] = result["rerank_score"]

        final_results = reranked[:n]

        logger.debug(
            f"Reranked {len(results)} results → top {len(final_results)} "
            f"(score range: {final_results[-1]['score']:.3f} - {final_results[0]['score']:.3f})"
        )
        return final_results
