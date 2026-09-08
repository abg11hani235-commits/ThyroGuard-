import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class PDFDocumentLoader:
    """Loads and extracts text page-by-page from medical PDFs using pypdf."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

    def load_pages(self) -> List[Dict[str, Any]]:
        import pypdf
        reader = pypdf.PdfReader(self.file_path)
        pages_data = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages_data.append({
                "page_number": idx + 1,
                "text": text
            })
        return pages_data


class SmartClinicalLoader:
    """
    SOTA Intelligent Document Parser for Clinical Guidelines.
    Preserves clinical tables, dosage matrices, and algorithms:
      - Primary Engine: Docling (converts PDF pages to structured Markdown tables).
      - Fallback Engine: pypdf (deterministic text extraction).
    """
    def __init__(self, file_path: str, prefer_docling: bool = True):
        self.file_path = file_path
        self.prefer_docling = prefer_docling
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

    def load_markdown_or_pages(self) -> Dict[str, Any]:
        """
        Attempts Docling conversion to preserve tables; falls back to pypdf.
        Returns a dictionary containing mode ('docling_markdown' or 'pypdf_pages')
        and extracted content.
        """
        if self.prefer_docling:
            try:
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                result = converter.convert(self.file_path)
                markdown_text = result.document.export_to_markdown()
                logger.info(f"Successfully parsed {self.file_path} via Docling.")
                return {
                    "mode": "docling_markdown",
                    "content": markdown_text,
                    "tables_preserved": True
                }
            except Exception as e:
                logger.info(f"Docling engine unavailable or error ({e}). Falling back to pypdf.")

        # Fallback to pypdf
        legacy_loader = PDFDocumentLoader(self.file_path)
        pages = legacy_loader.load_pages()
        return {
            "mode": "pypdf_pages",
            "content": pages,
            "tables_preserved": False
        }
