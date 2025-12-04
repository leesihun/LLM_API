"""
Base File Handler
=================
Abstract base class for all file handlers.

Provides unified interface for:
- Metadata extraction (for code generation)
- File analysis (for file inspection)
- Context building (for LLM prompts)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class FileHandler(ABC):
    """
    Unified abstract base class for all file handlers.

    All file handlers must implement:
    - extract_metadata(): Quick metadata for code generation
    - analyze(): Comprehensive analysis for inspection
    - supports(): File type checking
    """

    def __init__(self):
        """Initialize handler with supported extensions"""
        self.supported_extensions: List[str] = []
        self.file_type: str = "unknown"

    def supports(self, file_path: Path) -> bool:
        """
        Check if this handler supports the given file.

        Args:
            file_path: Path to file

        Returns:
            True if handler supports this file type
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def extract_metadata(
        self,
        file_path: Path,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Extract metadata from file for code generation context.

        Used by python_coder to build file context for LLM prompts.

        Args:
            file_path: Path to file
            quick_mode: If True, extract only essential metadata (faster)

        Returns:
            Dictionary containing:
            - file_type: Type of file
            - file_size: File size in bytes
            - file_size_human: Human-readable size
            - rows: Number of rows (if applicable)
            - columns: List of column names (if applicable)
            - dtypes: Data types (if applicable)
            - preview: Sample data
            - error: Error message if extraction failed
        """
        pass

    @abstractmethod
    def analyze(self, file_path: Path) -> Dict[str, Any]:
        """
        Perform comprehensive file analysis.

        Used by file_analyzer for detailed file inspection.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with comprehensive analysis:
            - format: File format type
            - size_bytes: File size
            - size_human: Human-readable size
            - structure: File structure details
            - statistics: Statistical info (if applicable)
            - preview: Content preview
            - metadata: Additional metadata
        """
        pass

    def build_context_section(
        self,
        filename: str,
        metadata: Dict[str, Any],
        index: int
    ) -> str:
        """
        Build formatted context section for LLM prompt.

        Args:
            filename: Name of file
            metadata: Metadata from extract_metadata()
            index: File index for numbering

        Returns:
            Formatted string for LLM prompt
        """
        lines = []
        lines.append(f"{index}. {filename} ({metadata.get('file_type', 'unknown')})")

        if metadata.get('error'):
            lines.append(f"   Error: {metadata['error']}")
            return '\n'.join(lines)

        # Format based on file type
        if 'rows' in metadata and metadata['rows'] != 'unknown':
            lines.append(f"   Rows: {metadata['rows']:,}")

        if 'columns' in metadata:
            cols = metadata['columns']
            if isinstance(cols, list) and cols:
                lines.append(f"   Columns ({len(cols)}): {', '.join(str(c) for c in cols[:8])}")
                if len(cols) > 8:
                    lines.append(f"   ... and {len(cols) - 8} more columns")

        if 'file_size_human' in metadata:
            lines.append(f"   Size: {metadata['file_size_human']}")

        return '\n'.join(lines)

    # Utility methods

    def _get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes"""
        return file_path.stat().st_size

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size for display"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _safe_str(self, value: Any, max_length: int = 100) -> str:
        """Safely convert value to string with length limit"""
        s = str(value)
        if len(s) > max_length:
            return s[:max_length] + "..."
        return s
