"""
ChromaDB Vector Store Module
Implements persistent vector storage using ChromaDB with
metadata filtering and collection management.
"""

from pathlib import Path
from typing import Optional
import uuid

from loguru import logger


class ChromaVectorStore:
    """
    ChromaDB-backed vector store with persistent storage,
    metadata-aware filtering, and batch operations.
    """

    def __init__(self, config: dict):
        chroma_config = config.get("vectorstore", {}).get("chroma", {})
        self.collection_name = chroma_config.get("collection_name", "sec_filings")
        self.persist_directory = chroma_config.get(
            "persist_directory", "data/vectorstore/chroma"
        )
        self.distance_metric = chroma_config.get("distance_metric", "cosine")
        self._client = None
        self._collection = None
        logger.info(
            f"ChromaVectorStore configured: collection={self.collection_name}, "
            f"persist={self.persist_directory}"
        )

    @property
    def client(self):
        """Lazy-initialize ChromaDB client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB client initialized at {self.persist_directory}")
        return self._client

    @property
    def collection(self):
        """Get or create the vector collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
            )
            logger.info(
                f"Collection '{self.collection_name}' ready with "
                f"{self._collection.count()} documents"
            )
        return self._collection

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ):
        """
        Add documents with their embeddings to the vector store.

        Args:
            documents: List of document text content.
            embeddings: List of embedding vectors.
            metadatas: Optional metadata for each document.
            ids: Optional unique IDs. Generated if not provided.
        """
        if not documents:
            return

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Sanitize metadata (ChromaDB only supports str, int, float, bool)
        if metadatas:
            metadatas = [self._sanitize_metadata(m) for m in metadatas]

        # Batch insert (ChromaDB has a limit of ~41666 per batch)
        batch_size = 5000
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end] if metadatas else None,
                ids=ids[i:batch_end],
            )

        logger.info(f"Added {len(documents)} documents to collection")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> dict:
        """
        Query the vector store for similar documents.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            where: Metadata filter conditions.
            where_document: Document content filter.

        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'distances'.
        """
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            query_params["where"] = where
        if where_document:
            query_params["where_document"] = where_document

        results = self.collection.query(**query_params)

        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.warning(f"Deleted collection: {self.collection_name}")

    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self.persist_directory,
        }

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Sanitize metadata to ChromaDB-compatible types."""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                sanitized[key] = str(value)
            elif value is None:
                sanitized[key] = ""
            else:
                sanitized[key] = str(value)
        return sanitized
