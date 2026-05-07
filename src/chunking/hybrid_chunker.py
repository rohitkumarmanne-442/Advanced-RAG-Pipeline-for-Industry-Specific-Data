"""
Hybrid Chunker Module
Combines semantic and recursive chunking strategies with
special handling for structured content like tables and lists.
"""

from typing import Optional
import re

from loguru import logger

from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.recursive_chunker import RecursiveChunker


class HybridChunker:
    """
    Intelligent chunking strategy that routes different content types
    to the most appropriate chunking method:
    - Tables → preserved as complete units
    - Narrative text → semantic chunking
    - Lists/structured data → recursive chunking with custom separators

    This approach maximizes retrieval accuracy for heterogeneous documents
    like SEC filings that mix prose, tables, and structured data.
    """

    # Patterns for identifying content types
    TABLE_PATTERN = re.compile(r"\[TABLES?\].*?(?=\n\n[A-Z]|\Z)", re.DOTALL)
    LIST_PATTERN = re.compile(
        r"(?:^|\n)(?:[\s]*[-•*]\s+.+\n?){3,}", re.MULTILINE
    )
    HEADER_PATTERN = re.compile(
        r"^(?:ITEM\s+\d+|PART\s+[IVX]+|Section\s+\d+).*$",
        re.MULTILINE | re.IGNORECASE,
    )

    def __init__(self, config: dict):
        self.config = config
        hybrid_config = config.get("chunking", {}).get("hybrid", {})
        self.table_strategy = hybrid_config.get(
            "table_chunk_strategy", "preserve_complete"
        )
        self.semantic_chunker = SemanticChunker(config)
        self.recursive_chunker = RecursiveChunker(config)
        logger.info("HybridChunker initialized with semantic + recursive strategies")

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[dict]:
        """
        Intelligently chunk text by content type.

        Process:
        1. Identify and extract tables (preserve as complete chunks)
        2. Identify structured content (lists, enumerations)
        3. Apply semantic chunking to narrative sections
        4. Apply recursive chunking to structured sections
        5. Reassemble in document order
        """
        if not text.strip():
            return []

        # Segment the document into content blocks
        segments = self._segment_document(text)

        all_chunks = []
        chunk_index = 0

        for segment in segments:
            content = segment["content"]
            seg_type = segment["type"]
            seg_metadata = dict(metadata) if metadata else {}
            seg_metadata["content_type"] = seg_type
            seg_metadata["section_header"] = segment.get("header")

            if seg_type == "table":
                # Preserve tables as complete chunks
                chunks = self._chunk_table(content, seg_metadata)
            elif seg_type == "list":
                # Use recursive chunking for lists
                chunks = self.recursive_chunker.chunk(content, seg_metadata)
            else:
                # Use semantic chunking for narrative text
                chunks = self.semantic_chunker.chunk(content, seg_metadata)

            # Update chunk indices
            for chunk in chunks:
                chunk["chunk_index"] = chunk_index
                chunk_index += 1
                all_chunks.append(chunk)

        logger.info(
            f"HybridChunker produced {len(all_chunks)} chunks from "
            f"{len(segments)} segments"
        )
        return all_chunks

    def _segment_document(self, text: str) -> list[dict]:
        """
        Segment document into typed content blocks while
        preserving document order.
        """
        segments = []
        remaining = text
        current_header = None

        # Track positions of special content
        while remaining:
            # Check for table content
            table_match = self.TABLE_PATTERN.search(remaining)
            list_match = self.LIST_PATTERN.search(remaining)
            header_match = self.HEADER_PATTERN.search(remaining)

            # Find the earliest match
            matches = []
            if table_match:
                matches.append(("table", table_match))
            if list_match:
                matches.append(("list", list_match))

            if not matches:
                # No special content found; treat remainder as narrative
                if remaining.strip():
                    segments.append(
                        {
                            "type": "narrative",
                            "content": remaining.strip(),
                            "header": current_header,
                        }
                    )
                break

            # Process the earliest match
            earliest_type, earliest_match = min(matches, key=lambda m: m[1].start())

            # Add narrative content before the match
            pre_content = remaining[: earliest_match.start()].strip()
            if pre_content:
                # Check for header in pre-content
                h_match = self.HEADER_PATTERN.search(pre_content)
                if h_match:
                    current_header = h_match.group().strip()

                segments.append(
                    {
                        "type": "narrative",
                        "content": pre_content,
                        "header": current_header,
                    }
                )

            # Add the special content
            segments.append(
                {
                    "type": earliest_type,
                    "content": earliest_match.group().strip(),
                    "header": current_header,
                }
            )

            # Move past this match
            remaining = remaining[earliest_match.end():]

        return segments if segments else [{"type": "narrative", "content": text, "header": None}]

    def _chunk_table(self, table_text: str, metadata: dict) -> list[dict]:
        """
        Handle table chunking based on configured strategy.
        Options: preserve_complete, split_by_rows, summarize
        """
        if self.table_strategy == "preserve_complete":
            # Keep the entire table as one chunk
            return [
                {
                    "content": table_text,
                    "metadata": {**metadata, "is_table": True},
                    "chunk_index": 0,
                }
            ]
        elif self.table_strategy == "split_by_rows":
            # Split large tables into row groups
            lines = table_text.split("\n")
            header_lines = lines[:3]  # Header + separator
            data_lines = lines[3:]

            if len(data_lines) <= 10:
                return [
                    {
                        "content": table_text,
                        "metadata": {**metadata, "is_table": True},
                        "chunk_index": 0,
                    }
                ]

            chunks = []
            row_group_size = 10
            for i in range(0, len(data_lines), row_group_size):
                group = data_lines[i : i + row_group_size]
                chunk_content = "\n".join(header_lines + group)
                chunks.append(
                    {
                        "content": chunk_content,
                        "metadata": {
                            **metadata,
                            "is_table": True,
                            "row_range": (i, i + len(group)),
                        },
                        "chunk_index": 0,
                    }
                )
            return chunks
        else:
            return [
                {
                    "content": table_text,
                    "metadata": {**metadata, "is_table": True},
                    "chunk_index": 0,
                }
            ]
