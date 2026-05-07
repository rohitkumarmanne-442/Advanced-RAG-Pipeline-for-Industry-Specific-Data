"""
Table Extractor Module
Extracts structured table data from PDF documents using
multiple extraction strategies for maximum accuracy.
"""

from pathlib import Path
from typing import Optional

from loguru import logger


class TableExtractor:
    """
    Extracts tables from PDF documents and converts them
    to structured format for inclusion in RAG context.
    """

    def __init__(self, config: dict):
        self.config = config
        self.extract_tables = config.get("extract_tables", True)
        logger.info("TableExtractor initialized")

    def extract_tables(self, file_path: str) -> dict[int, list]:
        """
        Extract tables from a PDF file.

        Returns:
            Dict mapping page numbers to lists of table dicts.
            Each table dict has 'headers' and 'rows' keys.
        """
        if not self.extract_tables:
            return {}

        try:
            return self._extract_with_camelot(file_path)
        except Exception as e:
            logger.warning(f"Camelot extraction failed: {e}. Falling back to pdfplumber.")
            try:
                return self._extract_with_pdfplumber(file_path)
            except Exception as e2:
                logger.error(f"All table extraction methods failed: {e2}")
                return {}

    def _extract_with_camelot(self, file_path: str) -> dict[int, list]:
        """Extract tables using Camelot (lattice + stream detection)."""
        import camelot

        tables_by_page = {}

        # Try lattice-based detection first (for bordered tables)
        tables = camelot.read_pdf(file_path, pages="all", flavor="lattice")

        if len(tables) == 0:
            # Fall back to stream-based detection (for borderless tables)
            tables = camelot.read_pdf(file_path, pages="all", flavor="stream")

        for table in tables:
            page_num = table.page
            df = table.df

            if df.empty:
                continue

            # Use first row as headers if it looks like a header
            headers = list(df.iloc[0])
            rows = [list(row) for _, row in df.iloc[1:].iterrows()]

            if page_num not in tables_by_page:
                tables_by_page[page_num] = []

            tables_by_page[page_num].append(
                {
                    "headers": headers,
                    "rows": rows,
                    "accuracy": table.accuracy,
                    "bbox": table._bbox,
                }
            )

        logger.info(
            f"Extracted {sum(len(t) for t in tables_by_page.values())} tables "
            f"from {len(tables_by_page)} pages"
        )
        return tables_by_page

    def _extract_with_pdfplumber(self, file_path: str) -> dict[int, list]:
        """Extract tables using pdfplumber as fallback."""
        import pdfplumber

        tables_by_page = {}

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()

                if not tables:
                    continue

                tables_by_page[page_num] = []
                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    headers = [str(cell) if cell else "" for cell in table[0]]
                    rows = [
                        [str(cell) if cell else "" for cell in row]
                        for row in table[1:]
                    ]

                    tables_by_page[page_num].append(
                        {"headers": headers, "rows": rows}
                    )

        return tables_by_page

    def table_to_markdown(self, table: dict) -> str:
        """Convert a table dict to markdown format for embedding."""
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        if not headers:
            return ""

        # Build markdown table
        header_str = "| " + " | ".join(str(h) for h in headers) + " |"
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        row_strs = [
            "| " + " | ".join(str(cell) for cell in row) + " |" for row in rows
        ]

        return "\n".join([header_str, separator] + row_strs)

    def table_to_natural_language(self, table: dict) -> str:
        """
        Convert a table to natural language description
        for better semantic search performance.
        """
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        if not headers or not rows:
            return ""

        descriptions = []
        for row in rows:
            pairs = []
            for header, value in zip(headers, row):
                if value and str(value).strip():
                    pairs.append(f"{header}: {value}")
            if pairs:
                descriptions.append("; ".join(pairs))

        return "\n".join(descriptions)
