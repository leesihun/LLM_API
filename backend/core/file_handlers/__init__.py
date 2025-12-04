"""
Unified File Handler System
============================
Single source of truth for all file handling across the application.

This module consolidates the previous 3 separate file handler systems:
- backend/services/file_handler/ (DEPRECATED)
- backend/tools/python_coder/file_handlers/ (DEPRECATED)
- backend/tools/file_analyzer/handlers/ (DEPRECATED)

Usage:
    >>> from backend.core.file_handlers import get_handler, file_handler_registry
    >>> handler = get_handler("data.csv")
    >>> metadata = handler.extract_metadata(Path("data.csv"))

Version: 2.0.0
Created: December 3, 2025
"""

from .base import FileHandler
from .registry import FileHandlerRegistry, file_handler_registry, get_handler
from .csv_handler import CSVHandler
from .excel_handler import ExcelHandler
from .json_handler import JSONHandler
from .text_handler import TextHandler
from .pdf_handler import PDFHandler
from .docx_handler import DOCXHandler
from .image_handler import ImageHandler

__all__ = [
    # Core classes
    'FileHandler',
    'FileHandlerRegistry',

    # Singleton registry
    'file_handler_registry',
    'get_handler',

    # Individual handlers
    'CSVHandler',
    'ExcelHandler',
    'JSONHandler',
    'TextHandler',
    'PDFHandler',
    'DOCXHandler',
    'ImageHandler',
]

__version__ = '2.0.0'
