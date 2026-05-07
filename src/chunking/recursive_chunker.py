"""
Recursive Chunker Module
Implements recursive character-based chunking with configurable
separators and overlap for maintaining context continuity.
"""

from typing import Optional

from loguru import logger


class RecursiveChunker:
    """
    Recursively splits documents using a hierarchy of separators,
    attempting to split at the most natural boundaries first
    (paragraphs > sentences > words).
    """

    def __init__(self, config: dict):
        chunking_config = config.get("chunking", {}).get("recursive", {})
        self.chunk_size = chunking_config.get("chunk_size", 512)
        self.chunk_overlap = chunking_config.get("chunk_overlap", 50)
        self.separators = chunking_config.get(
            "separators", ["\n\n", "\n", ". ", " "]
        )
        logger.info(
            f"RecursiveChunker initialized: size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[dict]:
        """
        Split text into overlapping chunks using recursive separation.

        Args:
            text: The document text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of chunk dicts with 'content', 'metadata', and 'chunk_index'.
        """
        if not text.strip():
            return []

        # Recursively split the text
        splits = self._recursive_split(text, self.separators)

        # Merge splits into chunks with overlap
        chunks = self._merge_with_overlap(splits)

        # Create chunk objects with metadata
        result = []
        for i, chunk_content in enumerate(chunks):
            chunk_meta = dict(metadata) if metadata else {}
            chunk_meta["char_count"] = len(chunk_content)
            chunk_meta["chunking_strategy"] = "recursive"

            result.append(
                {
                    "content": chunk_content,
                    "metadata": chunk_meta,
                    "chunk_index": i,
                }
            )

        logger.debug(f"Created {len(result)} recursive chunks")
        return result

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separator hierarchy."""
        if not separators:
            # Base case: split by character count
            return self._split_by_chars(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = text.split(separator)

        # Filter empty splits
        splits = [s for s in splits if s.strip()]

        # Check if any split is still too large
        final_splits = []
        for split in splits:
            if len(split) <= self.chunk_size:
                final_splits.append(split)
            else:
                # Recursively split with next separator
                sub_splits = self._recursive_split(split, remaining_separators)
                final_splits.extend(sub_splits)

        return final_splits

    def _split_by_chars(self, text: str) -> list[str]:
        """Last-resort splitting by character count."""
        splits = []
        for i in range(0, len(text), self.chunk_size):
            splits.append(text[i : i + self.chunk_size])
        return splits

    def _merge_with_overlap(self, splits: list[str]) -> list[str]:
        """
        Merge small splits into chunks of target size,
        maintaining overlap between consecutive chunks.
        """
        if not splits:
            return []

        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            # If adding this split exceeds chunk size, finalize current chunk
            if current_length + split_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Keep overlap from the end of current chunk
                overlap_text = chunk_text[-self.chunk_overlap :]
                current_chunk = [overlap_text, split]
                current_length = len(overlap_text) + split_len
            else:
                current_chunk.append(split)
                current_length += split_len

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
