"""
File Handler Registry
=====================
Central registry for all file handlers.

Provides singleton access to file handlers with automatic handler selection
based on file type.

Created: 2025-01-20
Version: 1.0.0
"""

from pathlib import Path
from typing import Dict, Optional, List
import logging

from backend.utils.logging_utils import get_logger
from backend.core.exceptions import UnsupportedFileTypeError

logger = get_logger(__name__)


class FileHandlerRegistry:
    """
    Singleton registry for file handlers.
    
    Automatically selects appropriate handler based on file extension.
    
    Usage:
        >>> registry = FileHandlerRegistry()
        >>> handler = registry.get_handler("/path/to/file.csv")
        >>> metadata = handler.extract_metadata(Path("/path/to/file.csv"))
    """
    
    _instance = None
    _handlers = []
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize registry and load handlers"""
        if not FileHandlerRegistry._initialized:
            self._load_handlers()
            FileHandlerRegistry._initialized = True
            logger.info(f"FileHandlerRegistry initialized with {len(self._handlers)} handlers")
    
    def _load_handlers(self):
        """Load all available handlers"""
        from backend.services.file_handler.csv_handler import CSVHandler
        from backend.services.file_handler.excel_handler import ExcelHandler  
        from backend.services.file_handler.json_handler import JSONHandler
        from backend.services.file_handler.text_handler import TextHandler
        # Import other handlers as they're created
        # from backend.services.file_handler.pdf_handler import PDFHandler
        # from backend.services.file_handler.docx_handler import DOCXHandler
        # from backend.services.file_handler.image_handler import ImageHandler
        
        FileHandlerRegistry._handlers = [
            CSVHandler(),
            ExcelHandler(),
            JSONHandler(),
            TextHandler(),
            # Add others as created
        ]
    
    def get_handler(self, file_path: str):
        """
        Get appropriate handler for file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Handler instance that supports this file type
            
        Raises:
            UnsupportedFileTypeError: If no handler found
        """
        for handler in FileHandlerRegistry._handlers:
            if handler.supports(file_path):
                return handler
        
        # No handler found
        path = Path(file_path)
        raise UnsupportedFileTypeError(
            file_path=str(file_path),
            file_type=path.suffix
        )
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of all supported file extensions.
        
        Returns:
            List of extensions (e.g., ['.csv', '.xlsx', '.json'])
        """
        extensions = set()
        for handler in FileHandlerRegistry._handlers:
            extensions.update(handler.supported_extensions)
        return sorted(list(extensions))
    
    def supports_file(self, file_path: str) -> bool:
        """
        Check if any handler supports this file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file type is supported
        """
        try:
            self.get_handler(file_path)
            return True
        except UnsupportedFileTypeError:
            return False


# Convenience singleton instance
file_handler_registry = FileHandlerRegistry()
