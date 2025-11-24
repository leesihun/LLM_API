"""
File Analyzer
==============
Main file analyzer class that routes to appropriate format-specific handlers.

Uses strategy pattern for extensible file format support.

Version: 1.1.0
Created: 2025-01-13
"""

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.utils.logging_utils import get_logger
from .base_handler import BaseFileHandler
from .summary_generator import SummaryGenerator
from .llm_analyzer import LLMAnalyzer
from .handlers import (
    CSVHandler,
    ExcelHandler,
    JSONHandler,
    TextHandler,
    PDFHandler,
    DOCXHandler,
    ImageHandler,
)

logger = get_logger(__name__)


@dataclass
class FileAnalysisPayload:
    """Normalized per-file analysis payload."""

    file: str
    full_path: str
    extension: str
    size_bytes: Optional[int]
    size_human: str
    success: bool = True
    format: str = "Unknown"
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "file": self.file,
            "full_path": self.full_path,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "size_human": self.size_human,
            "success": self.success,
            "format": self.format,
        }

        if self.warnings:
            payload["warnings"] = self.warnings

        if self.error:
            payload["error"] = self.error

        payload.update(self.details)
        return payload


class FileAnalyzer:
    """
    Main file analyzer that routes to appropriate handler.

    This class coordinates file analysis by:
    1. Identifying the file format
    2. Selecting the appropriate handler
    3. Executing the analysis
    4. Generating summaries

    Supports: CSV, Excel, JSON, Text, PDF, DOCX, Images

    Features:
    - Strategy pattern for extensibility
    - Automatic format detection
    - Batch analysis support
    - Comprehensive error handling
    - Optional LLM-powered deep analysis
    """

    def __init__(self, use_llm_for_complex: bool = False):
        """
        Initialize FileAnalyzer.

        Args:
            use_llm_for_complex: If True, uses LLM-based analysis for complex structures
        """
        self.use_llm_for_complex = use_llm_for_complex

        # Register all handlers
        self.handlers: List[BaseFileHandler] = [
            CSVHandler(),
            ExcelHandler(),
            JSONHandler(),
            TextHandler(),
            PDFHandler(),
            DOCXHandler(),
            ImageHandler(),
        ]

    def analyze(self, file_paths: List[str], user_query: str = "") -> Dict[str, Any]:
        try:
            normalized_paths = self._normalize_inputs(file_paths)

            analysis_id = uuid.uuid4().hex
            results: List[Dict[str, Any]] = []
            missing_files = 0

            for file_path in normalized_paths:
                file_result = self._analyze_single_file(file_path)
                results.append(file_result)

            summary = SummaryGenerator.generate_summary(results, user_query)
            successful = sum(1 for result in results if result.get("success", False))
            overall_success = successful > 0

            return {
                "success": overall_success,
                "files_analyzed": len(normalized_paths),
                "results": results,
                "summary": summary,
                "metadata": {
                    "analysis_id": analysis_id,
                    "successful_files": successful,
                    "failed_files": len(results) - successful,
                    "missing_files": missing_files,
                    "use_llm_for_complex": self.use_llm_for_complex,
                    "user_query_provided": bool(user_query.strip()),
                },
            }

        except Exception as e:
            logger.error(f"File analysis error: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }

    def _analyze_single_file(self, file_path: str) -> Dict[str, Any]:

        try:
            path = Path(file_path)
            size_bytes = os.path.getsize(file_path)
            payload = FileAnalysisPayload(
                file=path.name,
                full_path=str(path.absolute()),
                extension=path.suffix.lstrip('.').lower(),
                size_bytes=size_bytes,
                size_human=self._human_readable_size(size_bytes),
            )

            handler = self._find_handler(file_path)
            if handler:
                detailed_info = self._execute_handler(handler, file_path)
                payload.details.update(detailed_info)
                payload.format = detailed_info.get("format", payload.format)
            else:
                payload.format = "Unknown/Unsupported"
                payload.warnings.append(
                    f"No handler registered for '.{payload.extension}' files"
                )

            return payload.to_dict()

        except Exception as e:
            logger.error(f"Single file analysis error for {file_path}: {e}", exc_info=True)
            return self._build_failure_payload(file_path, str(e))

    def _normalize_inputs(self, file_paths: List[str]) -> List[str]:
        """Normalize input paths (strip None/empty entries)."""
        if not file_paths:
            return []

        normalized = []
        for path in file_paths:
            if not path:
                continue
            normalized.append(str(path))
        return normalized

    def _build_missing_file_result(self, file_path: str) -> Dict[str, Any]:
        """Return a consistent payload for missing files."""
        path = Path(file_path)
        return {
            "file": path.name or str(path),
            "full_path": str(path.absolute()),
            "extension": path.suffix.lstrip('.').lower(),
            "size_bytes": None,
            "size_human": "Unknown",
            "success": False,
            "format": "Unknown/Unsupported",
            "error": f"File not found: {file_path}",
        }

    def _execute_handler(self, handler: BaseFileHandler, file_path: str) -> Dict[str, Any]:
        """Run the handler and normalize its response."""
        details = handler.analyze(file_path) or {}
        if "format" not in details:
            handler_name = handler.__class__.__name__.replace("Handler", "").strip()
            details["format"] = handler_name or "Unknown"
        return details

    def _build_failure_payload(self, file_path: str, error_message: str) -> Dict[str, Any]:
        """Return payload for unexpected handler errors."""
        path = Path(file_path)
        return {
            "file": path.name or file_path,
            "full_path": str(path.absolute()),
            "extension": path.suffix.lstrip('.').lower(),
            "size_bytes": None,
            "size_human": "Unknown",
            "success": False,
            "format": "Unknown",
            "error": error_message,
        }

    def _find_handler(self, file_path: str) -> BaseFileHandler:
        """
        Find appropriate handler for file.

        Args:
            file_path: Path to file

        Returns:
            Handler instance or None if not supported
        """
        for handler in self.handlers:
            if handler.supports(file_path):
                return handler
        return None

    def is_supported(self, file_path: str) -> bool:
        """
        Check if file format is supported.

        Args:
            file_path: Path to file

        Returns:
            True if format is supported, False otherwise
        """
        return self._find_handler(file_path) is not None

    def get_supported_formats(self) -> List[str]:
        """
        Get list of all supported file formats.

        Returns:
            List of supported file extensions
        """
        formats = []
        for handler in self.handlers:
            formats.extend(handler.get_supported_extensions())
        return sorted(set(formats))

    def analyze_multiple(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple files.

        Args:
            file_paths: List of file paths

        Returns:
            List of analysis result dictionaries
        """
        results = []
        for file_path in file_paths:
            result = self._analyze_single_file(file_path)
            results.append(result)
        return results

    def _human_readable_size(self, size_bytes: int) -> str:
        """
        Convert bytes to human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Human-readable size string
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def deep_analyze_with_llm(self, file_path: str, user_query: str = "") -> Dict[str, Any]:
        """
        Use LLM to generate and execute custom analysis code.

        Args:
            file_path: Path to file to analyze
            user_query: Specific user question to guide analysis

        Returns:
            Dict with LLM analysis results
        """
        return LLMAnalyzer.deep_analyze(file_path, user_query)


# Singleton instance
file_analyzer = FileAnalyzer()


def analyze_files(file_paths: List[str], user_query: str = "") -> Dict[str, Any]:
    """
    Convenience function to analyze files.

    Args:
        file_paths: List of file paths to analyze
        user_query: User's question (optional)

    Returns:
        Analysis results dictionary
    """
    return file_analyzer.analyze(file_paths, user_query)
