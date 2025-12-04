"""
Core Exceptions Module
=====================
Simplified exception hierarchy for the LLM API system.

Version: 3.0.0 (Simplified)
Created: 2025-01-20
Refactored: 2025-01-XX (Reduced from 11 to 3 exception classes)
"""

from typing import Optional, Dict, Any


class LLMAPIException(Exception):
    """Base exception for all LLM API errors with metadata support."""

    def __init__(self, message: str, **metadata):
        """
        Initialize base exception.

        Args:
            message: Human-readable error message
            **metadata: Additional error context as keyword arguments
        """
        super().__init__(message)
        self.message = message
        self.metadata = metadata


class ToolError(LLMAPIException):
    """All tool-related errors (execution, validation, code execution, etc.)."""
    pass


class ConfigError(LLMAPIException):
    """Configuration errors."""
    pass


# Backward compatibility aliases (deprecated, use ToolError instead)
ToolExecutionError = ToolError
ToolValidationError = ToolError
CodeExecutionError = ToolError
ImportValidationError = ToolError
FileHandlerError = ToolError
UnsupportedFileTypeError = ToolError
AgentExecutionError = ToolError
LLMTimeoutError = ToolError
LLMResponseError = ToolError
ConfigurationError = ConfigError
