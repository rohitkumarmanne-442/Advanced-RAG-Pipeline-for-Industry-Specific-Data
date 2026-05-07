"""
Semantic Chunker Module
Implements embedding-based semantic chunking that splits documents
at natural topic boundaries rather than fixed character counts.
"""

import numpy as np
from typing import Optional

from loguru import logger


class SemanticChunker:
    """
    Splits documents into semantically coherent chunks by detecting
    topic shifts using embedding similarity between consecutive sentences.

    This approach preserves context integrity far better than naive
    fixed-size chunking, especially for complex financial/legal documents.
    """

    def __init__(self, config: dict):
        chunking_config = config.get("chunking", {}).get("semantic", {})
        self.embedding_model_name = chunking_config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.breakpoint_threshold = chunking_config.get("breakpoint_threshold", 0.3)
        self.min_chunk_size = chunking_config.get("min_chunk_size", 100)
        self.max_chunk_size = chunking_config.get("max_chunk_size", 1500)
        self._encoder = None
        logger.info(
            f"SemanticChunker initialized with threshold={self.breakpoint_threshold}"
        )

    @property
    def encoder(self):
        """Lazy-load the sentence transformer model."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        return self._encoder

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[dict]:
        """
        Split text into semantic chunks based on embedding similarity.

        Args:
            text: The document text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of chunk dicts with 'content', 'metadata', and 'chunk_index'.
        """
        if not text.strip():
            return []

        # Split into sentences
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [
                {
                    "content": text.strip(),
                    "metadata": metadata or {},
                    "chunk_index": 0,
                }
            ]

        # Compute embeddings for all sentences
        embeddings = self.encoder.encode(sentences, show_progress_bar=False)

        # Calculate cosine distances between consecutive sentences
        distances = self._calculate_distances(embeddings)

        # Find breakpoints where topic shifts occur
        breakpoints = self._find_breakpoints(distances)

        # Create chunks from breakpoints
        chunks = self._create_chunks(sentences, breakpoints, metadata)

        # Post-process: merge small chunks and split oversized ones
        chunks = self._post_process_chunks(chunks)

        logger.debug(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using NLTK."""
        import nltk

        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)

        # Filter out very short sentences (likely artifacts)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _calculate_distances(self, embeddings: np.ndarray) -> list[float]:
        """
        Calculate cosine distances between consecutive sentence embeddings.
        Higher distance = more semantic difference = potential breakpoint.
        """
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def _find_breakpoints(self, distances: list[float]) -> list[int]:
        """
        Identify indices where semantic breakpoints occur using
        percentile-based thresholding on distance values.
        """
        if not distances:
            return []

        # Use dynamic threshold based on document statistics
        threshold = np.percentile(distances, (1 - self.breakpoint_threshold) * 100)

        breakpoints = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                breakpoints.append(i + 1)  # Break AFTER this sentence

        return breakpoints

    def _create_chunks(
        self,
        sentences: list[str],
        breakpoints: list[int],
        metadata: Optional[dict],
    ) -> list[dict]:
        """Create chunk dicts from sentences and breakpoint indices."""
        chunks = []
        start_idx = 0

        all_breaks = breakpoints + [len(sentences)]

        for chunk_idx, break_idx in enumerate(all_breaks):
            chunk_sentences = sentences[start_idx:break_idx]
            if chunk_sentences:
                content = " ".join(chunk_sentences)
                chunk_meta = dict(metadata) if metadata else {}
                chunk_meta["sentence_range"] = (start_idx, break_idx - 1)
                chunk_meta["num_sentences"] = len(chunk_sentences)

                chunks.append(
                    {
                        "content": content,
                        "metadata": chunk_meta,
                        "chunk_index": chunk_idx,
                    }
                )
            start_idx = break_idx

        return chunks

    def _post_process_chunks(self, chunks: list[dict]) -> list[dict]:
        """Merge chunks that are too small and split chunks that are too large."""
        if not chunks:
            return chunks

        processed = []
        buffer = None

        for chunk in chunks:
            content_len = len(chunk["content"])

            if content_len < self.min_chunk_size:
                # Merge with buffer or start new buffer
                if buffer is None:
                    buffer = chunk
                else:
                    buffer["content"] += " " + chunk["content"]
                    buffer["metadata"]["num_sentences"] = (
                        buffer["metadata"].get("num_sentences", 0)
                        + chunk["metadata"].get("num_sentences", 0)
                    )
            else:
                # Flush buffer first
                if buffer is not None:
                    buffer["content"] += " " + chunk["content"]
                    buffer["metadata"]["num_sentences"] = (
                        buffer["metadata"].get("num_sentences", 0)
                        + chunk["metadata"].get("num_sentences", 0)
                    )
                    processed.append(buffer)
                    buffer = None
                elif content_len > self.max_chunk_size:
                    # Split oversized chunk
                    split_chunks = self._split_large_chunk(chunk)
                    processed.extend(split_chunks)
                else:
                    processed.append(chunk)

        # Don't forget remaining buffer
        if buffer is not None:
            if processed:
                processed[-1]["content"] += " " + buffer["content"]
            else:
                processed.append(buffer)

        # Re-index chunks
        for i, chunk in enumerate(processed):
            chunk["chunk_index"] = i

        return processed

    def _split_large_chunk(self, chunk: dict) -> list[dict]:
        """Split an oversized chunk at sentence boundaries."""
        sentences = self._split_into_sentences(chunk["content"])
        sub_chunks = []
        current_content = []
        current_len = 0

        for sentence in sentences:
            if current_len + len(sentence) > self.max_chunk_size and current_content:
                sub_chunks.append(
                    {
                        "content": " ".join(current_content),
                        "metadata": dict(chunk["metadata"]),
                        "chunk_index": 0,
                    }
                )
                current_content = [sentence]
                current_len = len(sentence)
            else:
                current_content.append(sentence)
                current_len += len(sentence)

        if current_content:
            sub_chunks.append(
                {
                    "content": " ".join(current_content),
                    "metadata": dict(chunk["metadata"]),
                    "chunk_index": 0,
                }
            )

        return sub_chunks
