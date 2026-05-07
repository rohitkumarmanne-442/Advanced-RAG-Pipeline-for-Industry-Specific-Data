"""Tests for chunking strategies."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def setup_method(self):
        self.config = {
            "chunking": {
                "recursive": {
                    "chunk_size": 200,
                    "chunk_overlap": 30,
                    "separators": ["\n\n", "\n", ". ", " "],
                }
            }
        }

    def test_basic_chunking(self):
        from src.chunking.recursive_chunker import RecursiveChunker

        chunker = RecursiveChunker(self.config)
        text = "First paragraph with some content. " * 20
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        assert all("content" in chunk for chunk in chunks)
        assert all("chunk_index" in chunk for chunk in chunks)

    def test_empty_text(self):
        from src.chunking.recursive_chunker import RecursiveChunker

        chunker = RecursiveChunker(self.config)
        chunks = chunker.chunk("")

        assert chunks == []

    def test_short_text(self):
        from src.chunking.recursive_chunker import RecursiveChunker

        chunker = RecursiveChunker(self.config)
        text = "Short text."
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0]["content"] == text

    def test_chunk_overlap(self):
        from src.chunking.recursive_chunker import RecursiveChunker

        chunker = RecursiveChunker(self.config)
        text = "Sentence one. " * 50
        chunks = chunker.chunk(text)

        # Check that chunks have some overlap
        if len(chunks) > 1:
            # The end of chunk N should overlap with start of chunk N+1
            for i in range(len(chunks) - 1):
                end_words = set(chunks[i]["content"].split()[-5:])
                start_words = set(chunks[i + 1]["content"].split()[:10])
                assert len(end_words.intersection(start_words)) > 0

    def test_metadata_preservation(self):
        from src.chunking.recursive_chunker import RecursiveChunker

        chunker = RecursiveChunker(self.config)
        text = "Some text content. " * 30
        metadata = {"source": "test.pdf", "page": 1}
        chunks = chunker.chunk(text, metadata=metadata)

        for chunk in chunks:
            assert chunk["metadata"]["source"] == "test.pdf"
            assert chunk["metadata"]["page"] == 1


class TestSemanticChunker:
    """Tests for SemanticChunker (requires model download)."""

    def setup_method(self):
        self.config = {
            "chunking": {
                "semantic": {
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "breakpoint_threshold": 0.3,
                    "min_chunk_size": 50,
                    "max_chunk_size": 500,
                }
            }
        }

    @pytest.mark.slow
    def test_semantic_chunking(self):
        from src.chunking.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(self.config)
        text = (
            "The company reported strong financial results in Q4. "
            "Revenue grew by 15% year over year. "
            "Operating margins expanded to 25%. "
            "The CEO announced a new strategic initiative. "
            "The initiative focuses on AI and machine learning. "
            "New products will be launched in 2024. "
            "In other news, the board approved a stock buyback program. "
            "The program authorizes up to $5 billion in repurchases."
        )
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        assert all("content" in chunk for chunk in chunks)

    def test_empty_input(self):
        from src.chunking.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(self.config)
        chunks = chunker.chunk("")
        assert chunks == []


class TestHybridChunker:
    """Tests for HybridChunker."""

    def setup_method(self):
        self.config = {
            "chunking": {
                "hybrid": {
                    "table_chunk_strategy": "preserve_complete",
                },
                "semantic": {
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "breakpoint_threshold": 0.3,
                    "min_chunk_size": 50,
                    "max_chunk_size": 500,
                },
                "recursive": {
                    "chunk_size": 200,
                    "chunk_overlap": 30,
                },
            }
        }

    def test_table_preservation(self):
        from src.chunking.hybrid_chunker import HybridChunker

        chunker = HybridChunker(self.config)
        text = (
            "Some narrative text before the table.\n\n"
            "[TABLES]\n"
            "Header1 | Header2 | Header3\n"
            "--- | --- | ---\n"
            "Value1 | Value2 | Value3\n"
            "Value4 | Value5 | Value6\n\n"
            "More narrative text after."
        )

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

        # Find table chunk
        table_chunks = [
            c for c in chunks if c.get("metadata", {}).get("content_type") == "table"
        ]
        # Tables should be preserved as complete units
        if table_chunks:
            assert "Header1" in table_chunks[0]["content"]
