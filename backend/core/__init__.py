"""
Core Infrastructure Module
==========================
Core infrastructure components for the LLM API system.

Exports:
- BaseTool: Abstract base class for all tools
- SyncBaseTool: Synchronous version of BaseTool
- ToolResult: Standard result type
- ExecutionMetadata: Execution metadata model
- StepResult: Individual step result
- MultiStepResult: Multi-step execution result
- CodeExecutionResult: Code execution result
- FileAnalysisResult: File analysis result
- All custom exceptions
- Retry utilities

Version: 2.0.0
Created: 2025-01-20
"""

# Base tool classes
from backend.core.base_tool import BaseTool, SyncBaseTool

# Result types
from backend.core.result_types import (
    ToolResult,
    ExecutionMetadata,
    StepResult,
    MultiStepResult,
    CodeExecutionResult,
    FileAnalysisResult,
    ExecutionStatus
)

# Exceptions
from backend.core.exceptions import (
    LLMAPIException,
    ToolExecutionError,
    ToolValidationError,
    LLMTimeoutError,
    LLMResponseError,
    FileHandlerError,
    CodeExecutionError,
    ImportValidationError,
    AgentExecutionError,
    ConfigurationError,
    UnsupportedFileTypeError
)

# Retry utilities
from backend.core.retry import (
    retry_async,
    retry_sync,
    with_retry,
    with_retry_sync,
    RetryConfig
)

__all__ = [
    # Base classes
    "BaseTool",
    "SyncBaseTool",

    # Result types
    "ToolResult",
    "ExecutionMetadata",
    "StepResult",
    "MultiStepResult",
    "CodeExecutionResult",
    "FileAnalysisResult",
    "ExecutionStatus",

    # Exceptions
    "LLMAPIException",
    "ToolExecutionError",
    "ToolValidationError",
    "LLMTimeoutError",
    "LLMResponseError",
    "FileHandlerError",
    "CodeExecutionError",
    "ImportValidationError",
    "AgentExecutionError",
    "ConfigurationError",
    "UnsupportedFileTypeError",

    # Retry utilities
    "retry_async",
    "retry_sync",
    "with_retry",
    "with_retry_sync",
    "RetryConfig"
]
