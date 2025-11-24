"""
Python code executor module.

This module provides isolated code execution with security restrictions.
Supports both traditional subprocess and persistent REPL modes for performance.

Main Components:
- CodeExecutor: Main execution class
- SandboxConfig: Security and resource configuration
- ImportValidator: Import security validation
- REPLManager: Persistent REPL management
- PersistentREPL: Single REPL instance

Usage:
    from backend.tools.python_coder.executor import CodeExecutor

    executor = CodeExecutor(timeout=30, use_persistent_repl=True)
    result = executor.execute(code, session_id="my-session")
"""

# Core executor
from .core import CodeExecutor, PythonExecutor

# Component classes
from .sandbox import SandboxConfig, BLOCKED_IMPORTS, SUPPORTED_FILE_TYPES
from .import_validator import ImportValidator
from .repl_manager import PersistentREPL, REPLManager

# Utility functions
from . import utils

__all__ = [
    # Main executor (primary API)
    "CodeExecutor",
    "PythonExecutor",  # Backward compatibility alias

    # Component classes
    "SandboxConfig",
    "ImportValidator",
    "PersistentREPL",
    "REPLManager",

    # Constants
    "BLOCKED_IMPORTS",
    "SUPPORTED_FILE_TYPES",

    # Utilities module
    "utils",
]

__version__ = "2.0.0"
__author__ = "LLM_API Team"
