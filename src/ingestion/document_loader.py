"""
Document Loader Module
Handles loading and initial processing of various document formats
including PDFs, HTML, JSON, and plain text files.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.table_extractor import TableExtractor


@dataclass
class DocumentMetadata:
    """Metadata associated with a loaded document."""

    source: str
    file_type: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    filing_type: Optional[str] = None  # e.g., 10-K, 10-Q, 8-K
    company_name: Optional[str] = None
    filing_date: Optional[str] = None
    has_tables: bool = False
    has_images: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class Document:
    """Represents a processed document with content and metadata."""

    content: str
    metadata: DocumentMetadata
    tables: list = field(default_factory=list)
    images: list = field(default_factory=list)


class DocumentLoader:
    """
    Unified document loader that handles multiple file formats
    and extracts structured content including tables and images.
    """

    SUPPORTED_FORMATS = {".pdf", ".txt", ".html", ".json", ".htm"}

    def __init__(self, config: dict):
        self.config = config
        self.pdf_parser = PDFParser(config.get("ingestion", {}))
        self.table_extractor = TableExtractor(config.get("ingestion", {}))
        logger.info("DocumentLoader initialized with config")

    def load_directory(self, directory_path: str) -> list[Document]:
        """Load all supported documents from a directory."""
        documents = []
        dir_path = Path(directory_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        for file_path in sorted(dir_path.rglob("*")):
            if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    docs = self.load_file(str(file_path))
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} document(s) from {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def load_file(self, file_path: str) -> list[Document]:
        """Load a single file and return document(s)."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {suffix}")

        if suffix == ".pdf":
            return self._load_pdf(path)
        elif suffix == ".txt":
            return self._load_text(path)
        elif suffix in (".html", ".htm"):
            return self._load_html(path)
        elif suffix == ".json":
            return self._load_json(path)

        return []

    def _load_pdf(self, path: Path) -> list[Document]:
        """Load and parse a PDF document with table/image extraction."""
        documents = []

        # Parse PDF pages
        parsed_pages = self.pdf_parser.parse(str(path))

        # Extract tables if configured
        tables_by_page = {}
        if self.config.get("ingestion", {}).get("extract_tables", True):
            tables_by_page = self.table_extractor.extract_tables(str(path))

        for page in parsed_pages:
            page_tables = tables_by_page.get(page["page_number"], [])

            metadata = DocumentMetadata(
                source=str(path),
                file_type="pdf",
                page_number=page["page_number"],
                section=page.get("section"),
                has_tables=len(page_tables) > 0,
                has_images=len(page.get("images", [])) > 0,
                extra={
                    "total_pages": page.get("total_pages"),
                    "char_count": len(page["content"]),
                },
            )

            # Integrate table content into the document text
            content = page["content"]
            if page_tables:
                table_text = self._format_tables(page_tables)
                content = f"{content}\n\n[TABLES]\n{table_text}"

            documents.append(
                Document(
                    content=content,
                    metadata=metadata,
                    tables=page_tables,
                    images=page.get("images", []),
                )
            )

        return documents

    def _load_text(self, path: Path) -> list[Document]:
        """Load a plain text file."""
        content = path.read_text(encoding="utf-8")
        metadata = DocumentMetadata(
            source=str(path),
            file_type="txt",
        )
        return [Document(content=content, metadata=metadata)]

    def _load_html(self, path: Path) -> list[Document]:
        """Load an HTML file and extract text content."""
        from unstructured.partition.html import partition_html

        elements = partition_html(filename=str(path))
        content = "\n\n".join(str(el) for el in elements)

        metadata = DocumentMetadata(
            source=str(path),
            file_type="html",
        )
        return [Document(content=content, metadata=metadata)]

    def _load_json(self, path: Path) -> list[Document]:
        """Load a JSON file (e.g., structured SEC filing data)."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both single documents and arrays
        if isinstance(data, list):
            documents = []
            for idx, item in enumerate(data):
                content = item.get("content", item.get("text", str(item)))
                metadata = DocumentMetadata(
                    source=str(path),
                    file_type="json",
                    section=item.get("section"),
                    company_name=item.get("company_name"),
                    filing_type=item.get("filing_type"),
                    filing_date=item.get("filing_date"),
                    extra={"index": idx},
                )
                documents.append(Document(content=content, metadata=metadata))
            return documents
        else:
            content = data.get("content", data.get("text", str(data)))
            metadata = DocumentMetadata(
                source=str(path),
                file_type="json",
                company_name=data.get("company_name"),
                filing_type=data.get("filing_type"),
                filing_date=data.get("filing_date"),
            )
            return [Document(content=content, metadata=metadata)]

    def _format_tables(self, tables: list) -> str:
        """Format extracted tables as structured text."""
        formatted = []
        for i, table in enumerate(tables):
            if isinstance(table, dict):
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                header_str = " | ".join(str(h) for h in headers)
                separator = " | ".join("---" for _ in headers)
                row_strs = [" | ".join(str(cell) for cell in row) for row in rows]
                table_str = f"Table {i + 1}:\n{header_str}\n{separator}\n" + "\n".join(
                    row_strs
                )
            else:
                table_str = f"Table {i + 1}:\n{str(table)}"
            formatted.append(table_str)
        return "\n\n".join(formatted)
