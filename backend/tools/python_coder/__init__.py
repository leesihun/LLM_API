"""
Python Coder Tool - Refactored Modular Implementation

This module provides a clean, modular architecture for Python code generation,
verification, and execution with LLM-based iterative improvement.

Public API exports:
- PythonCoderTool: Main orchestrator class
- python_coder_tool: Global singleton instance
- CodeExecutor: Low-level code execution (backward compatibility)
- PythonExecutor: Alias for CodeExecutor (backward compatibility)
- SUPPORTED_FILE_TYPES: List of supported file extensions
- FileContextStorage: File context persistence for multi-phase workflows
"""

from .orchestrator import (
    PythonCoderTool,
    python_coder_tool,
    PythonExecutor,  # Backward compatibility
    SUPPORTED_FILE_TYPES,
)

from .executor.core import CodeExecutor
from .file_context_storage import FileContextStorage
from .code_generator import CodeGenerator
from .code_verifier import CodeVerifier

__all__ = [
    # Main API
    'PythonCoderTool',
    'python_coder_tool',

    # Components (if needed for advanced usage)
    'CodeExecutor',
    'CodeGenerator',
    'CodeVerifier',
    'FileContextStorage',

    # Backward compatibility
    'PythonExecutor',
    'SUPPORTED_FILE_TYPES',
]

__version__ = '2.0.0'
