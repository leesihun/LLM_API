"""
Base Handler for File Analysis
================================
Abstract base class for format-specific file handlers.

Version: 1.0.0
Created: 2025-01-13
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List


class BaseFileHandler(ABC):
    """
    Abstract base class for file format handlers.

    Each handler is responsible for analyzing a specific file format
    and returning structured metadata and analysis results.

    Subclasses must implement:
    - supports(file_path): Check if handler can process this file
    - analyze(file_path): Extract metadata and analyze file content
    """

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """
        Check if this handler supports the given file.

        Args:
            file_path: Path to the file

        Returns:
            True if this handler can process the file, False otherwise
        """
        pass

    @abstractmethod
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze file and return comprehensive metadata.

        Args:
            file_path: Path to the file to analyze

        Returns:
            Dictionary with analysis results including:
            - format: File format type
            - Additional format-specific metadata

        Raises:
            Exception: If analysis fails
        """
        pass

    # Helper methods available to all handlers

    def get_file_extension(self, file_path: str) -> str:
        """
        Get file extension in lowercase.

        Args:
            file_path: Path to the file

        Returns:
            File extension without dot (e.g., 'csv', 'xlsx')
        """
        return Path(file_path).suffix.lstrip('.').lower()

    def get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes.

        Args:
            file_path: Path to the file

        Returns:
            File size in bytes
        """
        return os.path.getsize(file_path)

    def human_readable_size(self, size_bytes: int) -> str:
        """
        Convert bytes to human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Human-readable size string (e.g., '1.23 MB')
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists.

        Args:
            file_path: Path to the file

        Returns:
            True if file exists, False otherwise
        """
        return os.path.exists(file_path)

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this handler.

        Returns:
            List of supported extensions (e.g., ['csv', 'tsv'])

        Note:
            Subclasses should override this to specify their supported formats
        """
        return []
