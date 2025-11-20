"""
Backward Compatibility Shim for file_analyzer_tool.py
=====================================================
This file provides backward compatibility for code using the old monolithic file_analyzer_tool.py

DEPRECATED: Please use 'from backend.tools.file_analyzer import ...' instead

The modular implementation is in backend/tools/file_analyzer/
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'backend.tools.file_analyzer_tool' is deprecated. "
    "Use 'from backend.tools.file_analyzer import file_analyzer' instead. "
    "This compatibility shim will be removed in v3.0.0",
    DeprecationWarning,
    stacklevel=2
)

# Import from new modular location
from backend.tools.file_analyzer import (
    FileAnalyzer,
    file_analyzer,
    CSVHandler,
    JSONHandler,
    ExcelHandler,
    TextHandler,
    PDFHandler,
    ImageHandler,
    DOCXHandler,
    LLMAnalyzer,
)

# Legacy aliases (old names for backward compatibility)
CSVAnalyzer = CSVHandler
JSONAnalyzer = JSONHandler
ExcelAnalyzer = ExcelHandler
TextAnalyzer = TextHandler
PDFAnalyzer = PDFHandler
ImageAnalyzer = ImageHandler

# Export everything for backward compatibility
__all__ = [
    'FileAnalyzer',
    'file_analyzer',
    # New names
    'CSVHandler',
    'JSONHandler',
    'ExcelHandler',
    'TextHandler',
    'PDFHandler',
    'ImageHandler',
    'DOCXHandler',
    'LLMAnalyzer',
    # Legacy names (aliases)
    'CSVAnalyzer',
    'JSONAnalyzer',
    'ExcelAnalyzer',
    'TextAnalyzer',
    'PDFAnalyzer',
    'ImageAnalyzer',
]
