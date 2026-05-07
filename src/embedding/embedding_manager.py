"""
Embedding Manager Module
Manages embedding model loading, encoding, and caching
with support for multiple HuggingFace models.
"""

import numpy as np
from typing import Optional
from pathlib import Path

import torch
from loguru import logger


class EmbeddingManager:
    """
    Manages embedding generation using HuggingFace models with
    support for batched encoding, normalization, and caching.

    Supports models like:
    - BAAI/bge-large-en-v1.5 (best quality)
    - sentence-transformers/all-MiniLM-L6-v2 (fast)
    - BAAI/bge-small-en-v1.5 (balanced)
    """

    def __init__(self, config: dict):
        embedding_config = config.get("embedding", {})
        self.model_name = embedding_config.get("model_name", "BAAI/bge-large-en-v1.5")
        self.model_kwargs = embedding_config.get("model_kwargs", {"device": "cuda"})
        self.encode_kwargs = embedding_config.get(
            "encode_kwargs", {"normalize_embeddings": True, "batch_size": 32}
        )
        self.dimension = embedding_config.get("dimension", 1024)
        self._model = None
        self._cache = {}

        # Fallback to CPU if CUDA not available
        if self.model_kwargs.get("device") == "cuda" and not torch.cuda.is_available():
            self.model_kwargs["device"] = "cpu"
            logger.warning("CUDA not available, falling back to CPU")

        logger.info(f"EmbeddingManager configured with model: {self.model_name}")

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, device=self.model_kwargs.get("device", "cpu")
            )
            logger.info(
                f"Loaded embedding model: {self.model_name} "
                f"on {self.model_kwargs.get('device', 'cpu')}"
            )
        return self._model

    def encode(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts into embedding vectors.

        Args:
            texts: List of text strings to encode.
            batch_size: Override default batch size.
            show_progress: Show progress bar.

        Returns:
            Numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        bs = batch_size or self.encode_kwargs.get("batch_size", 32)
        normalize = self.encode_kwargs.get("normalize_embeddings", True)

        # Add query instruction for BGE models
        processed_texts = self._prepare_texts(texts)

        embeddings = self.model.encode(
            processed_texts,
            batch_size=bs,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query with appropriate instruction prefix.
        BGE models require 'Represent this sentence:' prefix for queries.
        """
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.encode_kwargs.get("normalize_embeddings", True),
            convert_to_numpy=True,
        )
        return embedding[0]

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        """Encode documents for storage in vector store."""
        return self.encode(documents, show_progress=True)

    def _prepare_texts(self, texts: list[str]) -> list[str]:
        """Prepare texts with model-specific instructions."""
        # BGE models benefit from instruction prefixes
        if "bge" in self.model_name.lower():
            return [
                f"Represent this sentence for searching relevant passages: {t}"
                for t in texts
            ]
        return texts

    def compute_similarity(
        self, query_embedding: np.ndarray, doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and document embeddings."""
        # Normalize if not already normalized
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(
            doc_embeddings, axis=1, keepdims=True
        )
        similarities = np.dot(doc_norms, query_norm)
        return similarities

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()

    def save_model(self, path: str):
        """Save the current model to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        logger.info(f"Model saved to {save_path}")

    def load_custom_model(self, path: str):
        """Load a fine-tuned model from disk."""
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            path, device=self.model_kwargs.get("device", "cpu")
        )
        logger.info(f"Loaded custom model from {path}")
