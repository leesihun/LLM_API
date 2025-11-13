"""Base file handler abstract class for the Strategy pattern."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class BaseFileHandler(ABC):
    """Abstract base class for file-specific metadata extraction and context building."""

    def __init__(self):
        self.supported_extensions = []

    @abstractmethod
    def extract_metadata(
        self,
        file_path: Path,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Extract metadata from a file.

        Args:
            file_path: Path to the file
            quick_mode: If True, extract only essential metadata (faster)

        Returns:
            Dictionary containing file metadata
        """
        pass

    @abstractmethod
    def build_context_section(
        self,
        filename: str,
        metadata: Dict[str, Any],
        index: int
    ) -> str:
        """
        Build a context section for this file to include in the prompt.

        Args:
            filename: Name of the file
            metadata: Metadata dictionary from extract_metadata
            index: File index (for numbering in context)

        Returns:
            Formatted string to include in the prompt
        """
        pass

    def supports_file(self, file_path: Path) -> bool:
        """Check if this handler supports the given file."""
        return file_path.suffix.lower() in self.supported_extensions

    def _get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes."""
        return file_path.stat().st_size

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size for display."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
