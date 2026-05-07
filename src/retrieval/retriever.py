"""
Hybrid Retriever Module
Implements multi-strategy retrieval with Reciprocal Rank Fusion (RRF)
combining dense vector search and sparse BM25 retrieval.
"""

from typing import Optional

import numpy as np
from loguru import logger


class HybridRetriever:
    """
    Production-grade hybrid retriever that combines:
    1. Dense retrieval (embedding-based semantic search)
    2. Sparse retrieval (BM25 keyword matching)

    Results are fused using Reciprocal Rank Fusion (RRF) which
    consistently outperforms simple score combination methods.
    """

    def __init__(self, config: dict, vector_store, embedding_manager):
        retrieval_config = config.get("retrieval", {})
        self.top_k = retrieval_config.get("top_k", 10)
        self.final_top_k = retrieval_config.get("final_top_k", 5)

        fusion_config = retrieval_config.get("fusion", {})
        self.fusion_method = fusion_config.get("method", "reciprocal_rank")
        self.rrf_k = fusion_config.get("rrf_k", 60)
        self.weights = fusion_config.get("weights", {"dense": 0.7, "sparse": 0.3})

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self._bm25_index = None
        self._corpus = None
        self._corpus_metadata = None

        logger.info(
            f"HybridRetriever initialized: fusion={self.fusion_method}, "
            f"top_k={self.top_k}, rrf_k={self.rrf_k}"
        )

    def build_sparse_index(self, documents: list[str], metadatas: list[dict] = None):
        """
        Build BM25 index for sparse retrieval.

        Args:
            documents: List of document texts.
            metadatas: Optional parallel metadata list.
        """
        from rank_bm25 import BM25Okapi

        # Tokenize documents for BM25
        tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        self._corpus = documents
        self._corpus_metadata = metadatas or [{} for _ in documents]

        logger.info(f"BM25 index built with {len(documents)} documents")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Perform hybrid retrieval combining dense and sparse search
        with Reciprocal Rank Fusion.

        Args:
            query: The search query.
            top_k: Override default number of results.
            metadata_filter: Optional metadata filter for dense search.

        Returns:
            List of result dicts with 'content', 'metadata', 'score', 'rank'.
        """
        k = top_k or self.final_top_k

        # Dense retrieval
        dense_results = self._dense_retrieve(query, metadata_filter)

        # Sparse retrieval (BM25)
        sparse_results = self._sparse_retrieve(query)

        # Fuse results
        if self.fusion_method == "reciprocal_rank":
            fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        elif self.fusion_method == "weighted":
            fused = self._weighted_fusion(dense_results, sparse_results)
        elif self.fusion_method == "relative_score":
            fused = self._relative_score_fusion(dense_results, sparse_results)
        else:
            fused = dense_results  # Fallback to dense only

        # Return top-k fused results
        final_results = fused[:k]

        logger.debug(
            f"Retrieved {len(final_results)} results "
            f"(dense={len(dense_results)}, sparse={len(sparse_results)})"
        )
        return final_results

    def _dense_retrieve(
        self, query: str, metadata_filter: Optional[dict] = None
    ) -> list[dict]:
        """Perform dense vector retrieval."""
        query_embedding = self.embedding_manager.encode_query(query)

        # Query vector store
        if hasattr(self.vector_store, "query"):
            results = self.vector_store.query(
                query_embedding=query_embedding.tolist()
                if isinstance(query_embedding, np.ndarray)
                else query_embedding,
                top_k=self.top_k,
                where=metadata_filter,
            )
        else:
            results = self.vector_store.query(
                query_embedding=query_embedding, top_k=self.top_k
            )

        # Normalize to standard format
        dense_results = []
        for i in range(len(results.get("documents", []))):
            dense_results.append(
                {
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                    "score": 1 - results["distances"][i]
                    if results.get("distances")
                    else 0.0,
                    "source": "dense",
                    "id": results["ids"][i] if results.get("ids") else str(i),
                }
            )

        return dense_results

    def _sparse_retrieve(self, query: str) -> list[dict]:
        """Perform BM25 sparse retrieval."""
        if self._bm25_index is None or self._corpus is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        sparse_results = []
        for idx in top_indices:
            if scores[idx] > 0:
                sparse_results.append(
                    {
                        "content": self._corpus[idx],
                        "metadata": self._corpus_metadata[idx],
                        "score": float(scores[idx]),
                        "source": "sparse",
                        "id": str(idx),
                    }
                )

        return sparse_results

    def _reciprocal_rank_fusion(
        self, *result_lists: list[dict]
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF) combining multiple ranked lists.

        RRF score = Σ 1 / (k + rank_i) for each list i

        This method is robust to score scale differences between
        retrieval methods and consistently outperforms linear combination.
        """
        # Track fused scores by document content (as unique key)
        fused_scores = {}
        doc_map = {}  # content -> full result dict

        for result_list in result_lists:
            for rank, result in enumerate(result_list):
                doc_key = result["content"][:200]  # Use first 200 chars as key
                rrf_score = 1.0 / (self.rrf_k + rank + 1)

                if doc_key in fused_scores:
                    fused_scores[doc_key] += rrf_score
                else:
                    fused_scores[doc_key] = rrf_score
                    doc_map[doc_key] = result

        # Sort by fused score
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

        fused_results = []
        for rank, key in enumerate(sorted_keys):
            result = doc_map[key].copy()
            result["score"] = fused_scores[key]
            result["rank"] = rank + 1
            result["source"] = "rrf_fusion"
            fused_results.append(result)

        return fused_results

    def _weighted_fusion(
        self, dense_results: list[dict], sparse_results: list[dict]
    ) -> list[dict]:
        """
        Weighted score fusion with configurable weights.
        Normalizes scores before combining.
        """
        dense_weight = self.weights.get("dense", 0.7)
        sparse_weight = self.weights.get("sparse", 0.3)

        # Normalize scores to [0, 1]
        dense_results = self._normalize_scores(dense_results)
        sparse_results = self._normalize_scores(sparse_results)

        # Combine with weights
        fused_scores = {}
        doc_map = {}

        for result in dense_results:
            key = result["content"][:200]
            fused_scores[key] = result["score"] * dense_weight
            doc_map[key] = result

        for result in sparse_results:
            key = result["content"][:200]
            if key in fused_scores:
                fused_scores[key] += result["score"] * sparse_weight
            else:
                fused_scores[key] = result["score"] * sparse_weight
                doc_map[key] = result

        # Sort by fused score
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

        return [
            {**doc_map[key], "score": fused_scores[key], "source": "weighted_fusion", "rank": i + 1}
            for i, key in enumerate(sorted_keys)
        ]

    def _relative_score_fusion(
        self, dense_results: list[dict], sparse_results: list[dict]
    ) -> list[dict]:
        """
        Relative Score Fusion - normalizes by min-max within each list
        then combines, giving more weight to confident retrievals.
        """
        dense_norm = self._min_max_normalize(dense_results)
        sparse_norm = self._min_max_normalize(sparse_results)

        # Merge
        all_results = {}
        doc_map = {}

        for result in dense_norm:
            key = result["content"][:200]
            all_results[key] = result["score"]
            doc_map[key] = result

        for result in sparse_norm:
            key = result["content"][:200]
            if key in all_results:
                all_results[key] = max(all_results[key], result["score"])
            else:
                all_results[key] = result["score"]
                doc_map[key] = result

        sorted_keys = sorted(all_results.keys(), key=lambda k: all_results[k], reverse=True)

        return [
            {**doc_map[key], "score": all_results[key], "source": "relative_fusion", "rank": i + 1}
            for i, key in enumerate(sorted_keys)
        ]

    def _normalize_scores(self, results: list[dict]) -> list[dict]:
        """Normalize scores to [0, 1] range."""
        if not results:
            return results

        scores = [r["score"] for r in results]
        max_score = max(scores) if scores else 1
        if max_score == 0:
            return results

        normalized = []
        for result in results:
            r = result.copy()
            r["score"] = r["score"] / max_score
            normalized.append(r)
        return normalized

    def _min_max_normalize(self, results: list[dict]) -> list[dict]:
        """Min-max normalize scores."""
        if not results:
            return results

        scores = [r["score"] for r in results]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s

        if range_s == 0:
            return [{**r, "score": 1.0} for r in results]

        return [{**r, "score": (r["score"] - min_s) / range_s} for r in results]

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        import re

        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        # Remove very short tokens
        return [t for t in tokens if len(t) > 2]
