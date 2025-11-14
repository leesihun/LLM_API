"""
File Analyzer
==============
Main file analyzer class that routes to appropriate format-specific handlers.

Uses strategy pattern for extensible file format support.

Version: 1.0.0
Created: 2025-01-13
"""

import os
import uuid
from typing import Dict, Any, List
from pathlib import Path

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
        """
        Main analysis entry point.

        Args:
            file_paths: List of file paths to analyze
            user_query: User's question (optional, for context)

        Returns:
            Dict with analysis results including:
            - success: Boolean indicating overall success
            - files_analyzed: Number of files processed
            - results: List of per-file analysis results
            - summary: Human-readable summary
        """
        try:
            if not file_paths:
                return {
                    "success": False,
                    "error": "No files provided for analysis"
                }

            results = []
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    results.append({
                        "file": file_path,
                        "success": False,
                        "error": f"File not found: {file_path}"
                    })
                    continue

                file_result = self._analyze_single_file(file_path)
                results.append(file_result)

            # Generate summary
            summary = SummaryGenerator.generate_summary(results, user_query)

            return {
                "success": True,
                "files_analyzed": len(file_paths),
                "results": results,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"File analysis error: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }

    def _analyze_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single file.

        Args:
            file_path: Path to file to analyze

        Returns:
            Dictionary with analysis results
        """
        try:
            path = Path(file_path)

            # Basic file info
            file_info = {
                "file": str(path.name),
                "full_path": str(path.absolute()),
                "extension": path.suffix.lstrip('.').lower(),
                "size_bytes": os.path.getsize(file_path),
                "size_human": self._human_readable_size(os.path.getsize(file_path)),
                "success": True
            }

            # Find appropriate handler and analyze
            handler = self._find_handler(file_path)
            if handler:
                detailed_info = handler.analyze(file_path)
                file_info.update(detailed_info)
            else:
                file_info["format"] = "Unknown/Unsupported"
                file_info["note"] = f"Format '{file_info['extension']}' not specifically supported"

            return file_info

        except Exception as e:
            logger.error(f"Single file analysis error for {file_path}: {e}", exc_info=True)
            return {
                "file": file_path,
                "success": False,
                "error": str(e)
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
