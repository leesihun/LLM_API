"""
Unified PDF File Handler
=========================
Handler for PDF files merging functionality from python_coder and file_analyzer.

Features:
- Page count extraction
- Text extraction from pages
- First page preview
- Metadata extraction
- Both extract_metadata() and analyze() support

Version: 2.0.0 (Unified)
Created: 2025-01-20
"""

from typing import Dict, Any

from .base import UnifiedFileHandler


class PDFHandler(UnifiedFileHandler):
    """Unified handler for PDF files."""

    def __init__(self):
        """Initialize PDF handler."""
        super().__init__()
        self.supported_extensions = ['pdf']

    def supports(self, file_path: str) -> bool:
        """
        Check if this is a PDF file.

        Args:
            file_path: Path to the file

        Returns:
            True if file has .pdf extension
        """
        return self.supports_file(file_path)

    def extract_metadata(
        self,
        file_path: str,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Extract metadata from PDF file for code generation.

        Args:
            file_path: Path to the PDF file
            quick_mode: If True, extract only essential metadata

        Returns:
            Dictionary containing PDF metadata
        """
        metadata = {
            'file_type': 'pdf',
            'file_size': self.get_file_size(file_path),
            'error': None
        }

        try:
            import PyPDF2

            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)

                metadata['total_pages'] = num_pages

                if not quick_mode and num_pages > 0:
                    # Extract text from first page
                    first_page_text = pdf_reader.pages[0].extract_text()
                    metadata['first_page_preview'] = first_page_text[:500]
                else:
                    metadata['first_page_preview'] = "(not extracted in quick mode)"

        except ImportError:
            metadata['error'] = "PyPDF2 not installed"
        except Exception as e:
            metadata['error'] = str(e)
            self.logger.error(f"PDF metadata extraction error: {e}", exc_info=True)

        return metadata

    def analyze(self, file_path: str, query: str = "") -> Dict[str, Any]:
        """
        Perform comprehensive PDF analysis.

        Args:
            file_path: Path to the PDF file
            query: Optional query (not used for PDF)

        Returns:
            Dictionary with comprehensive analysis results
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
            self.logger.error(f"PDF analysis error: {e}", exc_info=True)
            return {"format": "PDF", "error": str(e)}

    def build_context_section(
        self,
        filename: str,
        metadata: Dict[str, Any],
        index: int
    ) -> str:
        """
        Build context section for PDF file.

        Args:
            filename: Name of the PDF file
            metadata: Metadata from extract_metadata()
            index: File index for numbering

        Returns:
            Formatted context string
        """
        lines = []
        lines.append(f"{index}. {filename} (PDF)")

        if metadata.get('error'):
            lines.append(f"   Error: {metadata['error']}")
            return '\n'.join(lines)

        # Basic info
        total_pages = metadata.get('total_pages')
        if total_pages is not None:
            lines.append(f"   Pages: {total_pages}")

        # Preview
        preview = metadata.get('first_page_preview')
        if preview and preview != "(not extracted in quick mode)":
            preview_text = preview[:200]
            lines.append(f"   First page preview: {preview_text}...")

        return '\n'.join(lines)
