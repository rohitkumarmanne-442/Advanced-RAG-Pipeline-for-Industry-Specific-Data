"""
FAISS Vector Store Module
Implements high-performance vector storage using Facebook's FAISS
library with multiple index types for different scale requirements.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class FAISSVectorStore:
    """
    FAISS-backed vector store supporting multiple index types:
    - Flat: Exact search (best for < 100k vectors)
    - IVFFlat: Approximate search with inverted file index
    - IVFPQ: Product quantization for memory efficiency
    - HNSW: Graph-based approximate nearest neighbor

    Includes metadata storage and persistence.
    """

    def __init__(self, config: dict):
        faiss_config = config.get("vectorstore", {}).get("faiss", {})
        self.index_type = faiss_config.get("index_type", "IVFFlat")
        self.nlist = faiss_config.get("nlist", 100)
        self.nprobe = faiss_config.get("nprobe", 10)
        self.persist_path = faiss_config.get("persist_path", "data/vectorstore/faiss")
        self.dimension = config.get("embedding", {}).get("dimension", 1024)

        self._index = None
        self._documents = []  # Parallel list of document texts
        self._metadatas = []  # Parallel list of metadata dicts
        self._ids = []  # Parallel list of IDs

        logger.info(
            f"FAISSVectorStore configured: type={self.index_type}, dim={self.dimension}"
        )

    @property
    def index(self):
        """Lazy-initialize FAISS index."""
        if self._index is None:
            self._index = self._create_index()
        return self._index

    def _create_index(self):
        """Create a FAISS index based on configuration."""
        import faiss

        if self.index_type == "Flat":
            index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vectors)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            index.nprobe = self.nprobe
        elif self.index_type == "IVFPQ":
            quantizer = faiss.IndexFlatIP(self.dimension)
            # m must divide dimension; use 8 sub-quantizers
            m = min(8, self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, m, 8)
            index.nprobe = self.nprobe
        elif self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 connections per node
            index.hnsw.efSearch = 64
            index.hnsw.efConstruction = 128
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        logger.info(f"Created FAISS index: {self.index_type}")
        return index

    def add_documents(
        self,
        documents: list[str],
        embeddings: np.ndarray,
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ):
        """
        Add documents with embeddings to the FAISS index.

        For IVF-based indices, training is performed automatically
        if the index hasn't been trained yet.
        """
        import faiss

        if len(documents) == 0:
            return

        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Train index if needed (IVF indices require training)
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            logger.info(f"Training FAISS index with {len(embeddings)} vectors...")
            self.index.train(embeddings)

        # Add vectors to index
        self.index.add(embeddings)

        # Store associated data
        self._documents.extend(documents)
        self._metadatas.extend(metadatas or [{} for _ in documents])
        self._ids.extend(ids or [str(i) for i in range(len(self._ids), len(self._ids) + len(documents))])

        logger.info(f"Added {len(documents)} documents. Total: {self.index.ntotal}")

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> dict:
        """
        Query the FAISS index for similar documents.

        Args:
            query_embedding: Query vector (1D numpy array).
            top_k: Number of results.
            filter_fn: Optional function to filter results by metadata.

        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'distances'.
        """
        # Reshape for FAISS (expects 2D array)
        query_vec = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )

        # Search with extra results if filtering
        search_k = top_k * 3 if filter_fn else top_k
        distances, indices = self.index.search(query_vec, search_k)

        # Gather results
        results = {"ids": [], "documents": [], "metadatas": [], "distances": []}

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for insufficient results
                continue

            if idx >= len(self._documents):
                continue

            metadata = self._metadatas[idx]

            # Apply filter if provided
            if filter_fn and not filter_fn(metadata):
                continue

            results["ids"].append(self._ids[idx])
            results["documents"].append(self._documents[idx])
            results["metadatas"].append(metadata)
            results["distances"].append(float(dist))

            if len(results["ids"]) >= top_k:
                break

        return results

    def save(self, path: Optional[str] = None):
        """Persist the FAISS index and metadata to disk."""
        import faiss

        save_path = Path(path or self.persist_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))

        # Save metadata
        metadata = {
            "documents": self._documents,
            "metadatas": self._metadatas,
            "ids": self._ids,
            "index_type": self.index_type,
            "dimension": self.dimension,
        }
        with open(save_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"FAISS index saved to {save_path}")

    def load(self, path: Optional[str] = None):
        """Load a persisted FAISS index and metadata."""
        import faiss

        load_path = Path(path or self.persist_path)

        if not (load_path / "index.faiss").exists():
            raise FileNotFoundError(f"No FAISS index found at {load_path}")

        # Load FAISS index
        self._index = faiss.read_index(str(load_path / "index.faiss"))

        # Load metadata
        with open(load_path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        self._documents = metadata["documents"]
        self._metadatas = metadata["metadatas"]
        self._ids = metadata["ids"]

        logger.info(
            f"Loaded FAISS index from {load_path} "
            f"({self._index.ntotal} vectors)"
        )

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "index_type": self.index_type,
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal if self._index else 0,
            "is_trained": self.index.is_trained if self._index and hasattr(self.index, "is_trained") else True,
            "persist_path": self.persist_path,
        }
