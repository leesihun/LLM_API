"""
PDF File Handler
=================
Handler for analyzing PDF files with text extraction.

Version: 1.0.0
Created: 2025-01-13
"""

from typing import Dict, Any, List

from backend.utils.logging_utils import get_logger
from ..base_handler import BaseFileHandler

logger = get_logger(__name__)


class PDFHandler(BaseFileHandler):
    """
    Handler for PDF file analysis.

    Features:
    - Page count extraction
    - Text extraction from pages
    - First page preview
    - Metadata extraction
    """

    def __init__(self):
        """Initialize PDF handler."""
        self.supported_extensions = ['pdf']

    def supports(self, file_path: str) -> bool:
        """
        Check if this is a PDF file.

        Args:
            file_path: Path to the file

        Returns:
            True if file has .pdf extension
        """
        extension = self.get_file_extension(file_path)
        return extension in self.supported_extensions

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported extensions."""
        return self.supported_extensions

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with analysis results including:
            - format: 'PDF'
            - total_pages: Number of pages
            - first_page_preview: Text preview from first page
        """
        try:
            import PyPDF2

            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)

                # Extract text from first page
                first_page_text = ""
                if num_pages > 0:
                    first_page_text = pdf_reader.pages[0].extract_text()

                return {
                    "format": "PDF",
                    "total_pages": num_pages,
                    "first_page_preview": first_page_text[:500]
                }

        except ImportError:
            return {"format": "PDF", "error": "PyPDF2 not installed"}
        except Exception as e:
            logger.error(f"PDF analysis error: {e}", exc_info=True)
            return {"format": "PDF", "error": str(e)}
