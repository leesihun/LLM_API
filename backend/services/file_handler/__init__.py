"""
Unified File Handler System
============================
Centralized file handling system replacing duplicated handlers.

Public API:
- FileHandlerRegistry: Singleton registry for all handlers
- file_handler_registry: Pre-initialized registry instance
- UnifiedFileHandler: Base class for handlers
- Individual handler classes

Usage:
    >>> from backend.services.file_handler import file_handler_registry
    >>> handler = file_handler_registry.get_handler("data.csv")
    >>> metadata = handler.extract_metadata(Path("data.csv"))

Created: 2025-01-20
Version: 1.0.0
"""

from backend.services.file_handler.base import UnifiedFileHandler
from backend.services.file_handler.registry import FileHandlerRegistry, file_handler_registry
from backend.services.file_handler.csv_handler import CSVHandler
from backend.services.file_handler.excel_handler import ExcelHandler
from backend.services.file_handler.json_handler import JSONHandler
from backend.services.file_handler.text_handler import TextHandler

__all__ = [
    'UnifiedFileHandler',
    'FileHandlerRegistry',
    'file_handler_registry',
    'CSVHandler',
    'ExcelHandler',
    'JSONHandler',
    'TextHandler',
]

__version__ = '1.0.0'
