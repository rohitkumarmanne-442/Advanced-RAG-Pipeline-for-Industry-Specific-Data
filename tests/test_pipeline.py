"""Tests for the end-to-end RAG pipeline."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRAGPipeline:
    """Integration tests for the RAG pipeline."""

    def test_config_loading(self, tmp_path):
        """Test configuration file loading."""
        import yaml

        config = {
            "project": {"name": "test"},
            "chunking": {"strategy": "recursive", "recursive": {"chunk_size": 512}},
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {"normalize_embeddings": True, "batch_size": 32},
                "dimension": 384,
            },
            "vectorstore": {
                "backend": "chroma",
                "chroma": {
                    "collection_name": "test",
                    "persist_directory": str(tmp_path / "chroma"),
                },
            },
            "retrieval": {
                "top_k": 5,
                "final_top_k": 3,
                "fusion": {"method": "reciprocal_rank", "rrf_k": 60},
                "reranker": {"enabled": False},
            },
            "llm": {"provider": "ollama", "model_name": "mistral"},
            "evaluation": {"metrics": ["faithfulness"]},
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from src.pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(config_path=str(config_path))
        assert pipeline.config["project"]["name"] == "test"

    def test_config_not_found(self):
        """Test error on missing config."""
        from src.pipeline.rag_pipeline import RAGPipeline

        with pytest.raises(FileNotFoundError):
            RAGPipeline(config_path="nonexistent.yaml")

    def test_pipeline_stats(self, tmp_path):
        """Test pipeline statistics reporting."""
        import yaml

        config = {
            "project": {"name": "test"},
            "chunking": {"strategy": "semantic"},
            "embedding": {"model_name": "test-model", "dimension": 384},
            "vectorstore": {
                "backend": "chroma",
                "chroma": {"persist_directory": str(tmp_path / "chroma")},
            },
            "retrieval": {"fusion": {"method": "reciprocal_rank"}},
            "llm": {"provider": "ollama"},
            "evaluation": {},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from src.pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(config_path=str(config_path))
        stats = pipeline.get_pipeline_stats()

        assert stats["initialized"] is False
        assert stats["config"]["chunking_strategy"] == "semantic"
        assert stats["config"]["vector_backend"] == "chroma"


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_response_creation(self):
        from src.pipeline.rag_pipeline import RAGResponse

        response = RAGResponse(
            answer="Test answer",
            source_documents=[{"content": "source"}],
            retrieval_scores=[0.95],
            metadata={"query": "test"},
        )

        assert response.answer == "Test answer"
        assert len(response.source_documents) == 1
        assert response.retrieval_scores[0] == 0.95

    def test_empty_response(self):
        from src.pipeline.rag_pipeline import RAGResponse

        response = RAGResponse(answer="No results found.")
        assert response.source_documents == []
        assert response.retrieval_scores == []
