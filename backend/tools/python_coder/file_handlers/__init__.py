"""File handlers for different file types using Strategy pattern."""

from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseFileHandler
from .csv_handler import CSVFileHandler
from .docx_handler import DOCXFileHandler
from .excel_handler import ExcelFileHandler
from .json_handler import JSONFileHandler
from .text_handler import TextFileHandler


class FileHandlerFactory:
    """Factory for creating appropriate file handlers and managing metadata cache."""

    def __init__(self):
        self.handlers: List[BaseFileHandler] = [
            JSONFileHandler(),
            ExcelFileHandler(),
            CSVFileHandler(),
            DOCXFileHandler(),
            TextFileHandler(),  # Fallback for general text files
        ]
        self._metadata_cache: Dict[str, dict] = {}

    def get_handler(self, file_path: Path) -> Optional[BaseFileHandler]:
        """Get appropriate handler for the given file."""
        for handler in self.handlers:
            if handler.supports_file(file_path):
                return handler
        return None

    def extract_metadata(
        self,
        file_path: Path,
        quick_mode: bool = False,
        use_cache: bool = True
    ) -> dict:
        """
        Extract metadata for a file using the appropriate handler.

        Args:
            file_path: Path to the file
            quick_mode: If True, extract only essential metadata
            use_cache: If True, use cached metadata if available

        Returns:
            Metadata dictionary
        """
        file_path_str = str(file_path)

        # Check cache
        if use_cache and file_path_str in self._metadata_cache:
            return self._metadata_cache[file_path_str]

        # Get handler and extract metadata
        handler = self.get_handler(file_path)
        if handler:
            metadata = handler.extract_metadata(file_path, quick_mode=quick_mode)
        else:
            metadata = {
                'file_type': 'unknown',
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'error': 'No handler available for this file type'
            }

        # Cache the result
        if use_cache:
            self._metadata_cache[file_path_str] = metadata

        return metadata

    def build_context_section(
        self,
        filename: str,
        file_path: Path,
        metadata: dict,
        index: int
    ) -> str:
        """Build context section for a file."""
        handler = self.get_handler(file_path)
        if handler:
            return handler.build_context_section(filename, metadata, index)
        else:
            return f"{index}. {filename} (Unknown type)"

    def clear_cache(self):
        """Clear the metadata cache."""
        self._metadata_cache.clear()


__all__ = [
    'BaseFileHandler',
    'JSONFileHandler',
    'ExcelFileHandler',
    'CSVFileHandler',
    'DOCXFileHandler',
    'TextFileHandler',
    'FileHandlerFactory',
]
