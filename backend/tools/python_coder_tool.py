"""
Backward Compatibility Shim for python_coder_tool.py
====================================================
This file provides backward compatibility for code using the old monolithic python_coder_tool.py

DEPRECATED: Please use 'from backend.tools.python_coder import ...' instead

The modular implementation is in backend/tools/python_coder/
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'backend.tools.python_coder_tool' is deprecated. "
    "Use 'from backend.tools.python_coder import python_coder_tool' instead. "
    "This compatibility shim will be removed in v3.0.0",
    DeprecationWarning,
    stacklevel=2
)

# Import from new modular location
from backend.tools.python_coder import (
    PythonCoderTool,
    python_coder_tool,
    CodeGenerator,
    CodeExecutor,
    CodeVerifier,
    SUPPORTED_FILE_TYPES,
)

# Legacy aliases
PythonExecutor = CodeExecutor  # Old name for executor

# Export everything for backward compatibility
__all__ = [
    'PythonCoderTool',
    'python_coder_tool',
    'CodeGenerator',
    'CodeExecutor',
    'CodeVerifier',
    'PythonExecutor',  # Legacy alias
    'SUPPORTED_FILE_TYPES',
]
