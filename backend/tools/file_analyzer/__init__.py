"""
File Analyzer Module
====================
Modular file analysis system using strategy pattern.

This module provides comprehensive file analysis capabilities with:
- Format-specific handlers for CSV, Excel, JSON, Text, PDF, DOCX, Images
- Extensible architecture using strategy pattern
- Data quality profiling and metadata extraction
- Support for complex nested structures
- Optional LLM-powered deep analysis

Main Components:
- FileAnalyzer: Main analyzer class that orchestrates handlers
- BaseFileHandler: Abstract base class for all handlers
- Format-specific handlers: CSVHandler, ExcelHandler, JSONHandler, etc.

Usage:
    from backend.tools.file_analyzer import FileAnalyzer, analyze_files

    # Using the singleton
    results = analyze_files(["/path/to/file.csv"])

    # Using instance
    analyzer = FileAnalyzer()
    results = analyzer.analyze(["/path/to/file.csv"])

Version: 1.0.0
Created: 2025-01-13
"""

from .analyzer import FileAnalyzer, file_analyzer, analyze_files
from .base_handler import BaseFileHandler

__all__ = [
    'FileAnalyzer',
    'file_analyzer',
    'analyze_files',
    'BaseFileHandler',
]

__version__ = '1.0.0'
