"""
File Handler Registry
=====================
Singleton registry for all file handlers.

Version: 2.0.0
Created: December 3, 2025
"""

from pathlib import Path
from typing import Dict, Optional, List
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FileHandlerRegistry:
    """
    Registry for file handlers with caching support.

    Manages all file type handlers and provides unified access.
    """

    def __init__(self):
        self.handlers: List = []
        self._metadata_cache: Dict[str, dict] = {}
        self._init_handlers()

    def _init_handlers(self):
        """Initialize all handlers in priority order"""
        from .csv_handler import CSVHandler
        from .excel_handler import ExcelHandler
        from .json_handler import JSONHandler
        from .docx_handler import DOCXHandler
        from .pdf_handler import PDFHandler
        from .image_handler import ImageHandler
        from .text_handler import TextHandler

        # Order matters - more specific handlers first
        self.handlers = [
            CSVHandler(),
            ExcelHandler(),
            JSONHandler(),
            DOCXHandler(),
            PDFHandler(),
            ImageHandler(),
            TextHandler(),  # Fallback
        ]

    def get_handler(self, file_path: Path):
        """
        Get appropriate handler for file.

        Args:
            file_path: Path to file

        Returns:
            Handler instance or None
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        for handler in self.handlers:
            if handler.supports(file_path):
                return handler
        return None

    def extract_metadata(
        self,
        file_path: Path,
        quick_mode: bool = False,
        use_cache: bool = True
    ) -> dict:
        """
        Extract metadata using appropriate handler.

        Args:
            file_path: Path to file
            quick_mode: Quick mode (essential metadata only)
            use_cache: Use cached metadata if available

        Returns:
            Metadata dictionary
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        file_path_str = str(file_path)

        # Check cache
        if use_cache and file_path_str in self._metadata_cache:
            return self._metadata_cache[file_path_str]

        # Get handler
        handler = self.get_handler(file_path)
        if handler:
            metadata = handler.extract_metadata(file_path, quick_mode=quick_mode)
        else:
            metadata = {
                'file_type': 'unknown',
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'error': 'No handler available for this file type'
            }

        # Cache result
        if use_cache:
            self._metadata_cache[file_path_str] = metadata

        return metadata

    def analyze(self, file_path: Path) -> dict:
        """
        Perform comprehensive file analysis.

        Args:
            file_path: Path to file

        Returns:
            Analysis dictionary
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        handler = self.get_handler(file_path)
        if handler:
            return handler.analyze(file_path)
        else:
            return {
                'format': 'unknown',
                'error': 'No handler available for this file type'
            }

    def build_context_section(
        self,
        filename: str,
        file_path: Path,
        metadata: dict,
        index: int
    ) -> str:
        """Build context section for LLM prompt"""
        handler = self.get_handler(file_path)
        if handler:
            return handler.build_context_section(filename, metadata, index)
        else:
            return f"{index}. {filename} (Unknown type)"

    def clear_cache(self):
        """Clear metadata cache"""
        self._metadata_cache.clear()


# Singleton instance
file_handler_registry = FileHandlerRegistry()


def get_handler(file_path: Path):
    """
    Convenience function to get handler.

    Args:
        file_path: Path to file

    Returns:
        Handler instance or None
    """
    return file_handler_registry.get_handler(file_path)
