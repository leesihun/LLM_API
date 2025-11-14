"""
Text File Handler
==================
Handler for analyzing text files (.txt, .md, .log, .py, .js, etc.).

Version: 1.0.0
Created: 2025-01-13
"""

from typing import Dict, Any, List

from backend.utils.logging_utils import get_logger
from ..base_handler import BaseFileHandler

logger = get_logger(__name__)


class TextHandler(BaseFileHandler):
    """
    Handler for text file analysis.

    Features:
    - Multi-encoding support
    - Line and word counting
    - Character statistics
    - Text preview extraction
    - Support for various text formats
    """

    def __init__(self):
        """Initialize text handler."""
        self.supported_extensions = ['txt', 'md', 'log', 'py', 'js', 'java', 'c', 'cpp', 'h', 'css', 'html', 'xml', 'yaml', 'yml', 'ini', 'cfg', 'conf', 'sh', 'bash']

    def supports(self, file_path: str) -> bool:
        """
        Check if this is a text file.

        Args:
            file_path: Path to the file

        Returns:
            True if file has text extension
        """
        extension = self.get_file_extension(file_path)
        return extension in self.supported_extensions

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported extensions."""
        return self.supported_extensions

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze text file.

        Args:
            file_path: Path to the text file

        Returns:
            Dictionary with analysis results including:
            - format: 'Text'
            - encoding: Detected encoding
            - total_lines: Number of lines
            - total_characters: Total character count
            - total_words: Total word count
            - preview: First 10 lines of content
        """
        try:
            # Try multiple encodings
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']
            content = None
            used_encoding = None

            for enc in encodings:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        content = f.read()
                    used_encoding = enc
                    break
                except Exception:
                    continue

            if content is None:
                return {
                    "format": "Text",
                    "error": "Failed to read text file with any encoding"
                }

            # Analyze content
            lines = content.split('\n')

            return {
                "format": "Text",
                "encoding": used_encoding,
                "total_lines": len(lines),
                "total_characters": len(content),
                "total_words": len(content.split()),
                "preview": '\n'.join(lines[:10])
            }

        except Exception as e:
            logger.error(f"Text analysis error: {e}", exc_info=True)
            return {"format": "Text", "error": str(e)}
