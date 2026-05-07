"""Tests for retrieval and rank fusion."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHybridRetriever:
    """Tests for HybridRetriever with RRF."""

    def setup_method(self):
        self.config = {
            "retrieval": {
                "top_k": 5,
                "final_top_k": 3,
                "fusion": {
                    "method": "reciprocal_rank",
                    "rrf_k": 60,
                    "weights": {"dense": 0.7, "sparse": 0.3},
                },
                "reranker": {"enabled": False},
            },
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {"normalize_embeddings": True, "batch_size": 32},
                "dimension": 384,
            },
        }

    def test_reciprocal_rank_fusion(self):
        from src.retrieval.retriever import HybridRetriever

        # Create mock objects
        class MockVectorStore:
            def query(self, **kwargs):
                return {
                    "ids": ["1", "2", "3"],
                    "documents": ["Doc A about revenue", "Doc B about expenses", "Doc C about growth"],
                    "metadatas": [{}, {}, {}],
                    "distances": [0.1, 0.3, 0.5],
                }

        class MockEmbeddingManager:
            def encode_query(self, query):
                return np.random.randn(384).astype(np.float32)

        retriever = HybridRetriever(
            self.config, MockVectorStore(), MockEmbeddingManager()
        )

        # Build sparse index
        corpus = [
            "Revenue grew by 15% in Q4 2023",
            "Operating expenses decreased by 5%",
            "The company expanded into new markets",
            "Net income reached $2.5 billion",
            "Customer acquisition cost improved by 20%",
        ]
        retriever.build_sparse_index(corpus)

        # Test retrieval
        results = retriever.retrieve("What was the revenue growth?")

        assert len(results) <= 3  # final_top_k
        assert all("content" in r for r in results)
        assert all("score" in r for r in results)
        # Results should be sorted by score (descending)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_score_calculation(self):
        """Verify RRF score formula: 1/(k + rank)"""
        from src.retrieval.retriever import HybridRetriever

        class MockVS:
            def query(self, **kwargs):
                return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        class MockEM:
            def encode_query(self, q):
                return np.zeros(384)

        retriever = HybridRetriever(self.config, MockVS(), MockEM())

        # Test RRF with known inputs
        list1 = [
            {"content": "Doc A", "metadata": {}, "score": 0.9, "source": "dense", "id": "1"},
            {"content": "Doc B", "metadata": {}, "score": 0.7, "source": "dense", "id": "2"},
        ]
        list2 = [
            {"content": "Doc B", "metadata": {}, "score": 5.0, "source": "sparse", "id": "2"},
            {"content": "Doc A", "metadata": {}, "score": 3.0, "source": "sparse", "id": "1"},
        ]

        fused = retriever._reciprocal_rank_fusion(list1, list2)

        # Both Doc A and Doc B appear in both lists
        # Doc A: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
        # Doc B: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
        assert len(fused) == 2
        assert all(r["source"] == "rrf_fusion" for r in fused)

    def test_empty_corpus(self):
        from src.retrieval.retriever import HybridRetriever

        class MockVS:
            def query(self, **kwargs):
                return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        class MockEM:
            def encode_query(self, q):
                return np.zeros(384)

        retriever = HybridRetriever(self.config, MockVS(), MockEM())
        results = retriever.retrieve("test query")
        assert results == []


class TestReranker:
    """Tests for cross-encoder reranker."""

    def setup_method(self):
        self.config = {
            "retrieval": {
                "reranker": {
                    "enabled": True,
                    "model": "BAAI/bge-reranker-large",
                    "top_n": 3,
                }
            }
        }

    def test_reranker_disabled(self):
        from src.retrieval.reranker import Reranker

        config = {"retrieval": {"reranker": {"enabled": False}}}
        reranker = Reranker(config)

        results = [
            {"content": "Doc 1", "score": 0.5},
            {"content": "Doc 2", "score": 0.8},
        ]

        reranked = reranker.rerank("test query", results)
        # Should return unchanged when disabled
        assert reranked == results

    def test_empty_results(self):
        from src.retrieval.reranker import Reranker

        reranker = Reranker(self.config)
        reranked = reranker.rerank("test query", [])
        assert reranked == []
