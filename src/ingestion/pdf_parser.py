"""
PDF Parser Module
Handles parsing of PDF documents using multiple backends
with support for OCR, image extraction, and layout analysis.
"""

from pathlib import Path
from typing import Optional

from loguru import logger


class PDFParser:
    """
    Multi-backend PDF parser that extracts text, images, and
    structural information from PDF documents.
    """

    def __init__(self, config: dict):
        self.config = config
        self.parser_backend = config.get("pdf_parser", "unstructured")
        self.ocr_enabled = config.get("ocr_enabled", True)
        self.extract_images = config.get("extract_images", True)
        logger.info(f"PDFParser initialized with backend: {self.parser_backend}")

    def parse(self, file_path: str) -> list[dict]:
        """
        Parse a PDF file and return structured page data.

        Returns:
            List of dicts with keys: page_number, content, section, images, total_pages
        """
        if self.parser_backend == "unstructured":
            return self._parse_with_unstructured(file_path)
        elif self.parser_backend == "pymupdf":
            return self._parse_with_pymupdf(file_path)
        else:
            raise ValueError(f"Unsupported PDF parser backend: {self.parser_backend}")

    def _parse_with_unstructured(self, file_path: str) -> list[dict]:
        """Parse PDF using the Unstructured library for layout-aware extraction."""
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res" if self.ocr_enabled else "fast",
            extract_images_in_pdf=self.extract_images,
            infer_table_structure=True,
        )

        # Group elements by page
        pages = {}
        for element in elements:
            page_num = element.metadata.page_number if element.metadata.page_number else 1
            if page_num not in pages:
                pages[page_num] = {
                    "page_number": page_num,
                    "content": "",
                    "section": None,
                    "images": [],
                    "elements": [],
                }

            # Track section headers
            element_type = type(element).__name__
            if element_type == "Title":
                pages[page_num]["section"] = str(element)

            pages[page_num]["content"] += str(element) + "\n"
            pages[page_num]["elements"].append(
                {"type": element_type, "text": str(element)}
            )

        total_pages = max(pages.keys()) if pages else 0
        result = []
        for page_num in sorted(pages.keys()):
            page_data = pages[page_num]
            page_data["total_pages"] = total_pages
            # Remove internal elements tracking from output
            del page_data["elements"]
            result.append(page_data)

        logger.info(f"Parsed {len(result)} pages from {Path(file_path).name}")
        return result

    def _parse_with_pymupdf(self, file_path: str) -> list[dict]:
        """Parse PDF using PyMuPDF for fast text extraction."""
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        total_pages = len(doc)
        pages = []

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")

            # Extract images
            images = []
            if self.extract_images:
                image_list = page.get_images(full=True)
                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    images.append(
                        {
                            "index": img_idx,
                            "width": pix.width,
                            "height": pix.height,
                            "xref": xref,
                        }
                    )

            # Detect section headers (lines in larger font or bold)
            blocks = page.get_text("dict")["blocks"]
            section = None
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["size"] > 14 or "bold" in span["font"].lower():
                                section = span["text"].strip()
                                break
                        if section:
                            break
                if section:
                    break

            pages.append(
                {
                    "page_number": page_num + 1,
                    "content": text,
                    "section": section,
                    "images": images,
                    "total_pages": total_pages,
                }
            )

        doc.close()
        logger.info(f"Parsed {total_pages} pages from {Path(file_path).name}")
        return pages
