"""
Core Exceptions Module
=====================
Custom exception hierarchy for the LLM API system.

Provides specialized exceptions for different error scenarios:
- Tool execution errors
- Validation errors
- LLM-related errors
- File handling errors

Version: 2.0.0
Created: 2025-01-20
"""

from typing import Optional, Dict, Any


class LLMAPIException(Exception):
    """Base exception for all LLM API errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize base exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
            error_code: Optional error code for categorization
        """
        self.message = message
        self.details = details or {}
        self.error_code = error_code or self.__class__.__name__
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary format.

        Returns:
            Dictionary representation of the exception
        """
        return {
            "error": self.message,
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "details": self.details
        }


class ToolExecutionError(LLMAPIException):
    """
    Raised when a tool fails to execute properly.

    This is the most common exception for tool-related failures,
    including subprocess errors, API failures, and execution timeouts.

    Examples:
        >>> raise ToolExecutionError(
        ...     "Python code execution failed",
        ...     details={"code": "...", "exit_code": 1}
        ... )
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize tool execution error.

        Args:
            message: Error message
            tool_name: Name of the tool that failed
            details: Additional error context
            error_code: Error code
        """
        self.tool_name = tool_name
        details = details or {}
        if tool_name:
            details["tool_name"] = tool_name

        super().__init__(message, details, error_code or "TOOL_EXECUTION_ERROR")


class ToolValidationError(LLMAPIException):
    """
    Raised when tool input validation fails.

    Used for pre-execution validation failures, such as:
    - Invalid parameters
    - Missing required inputs
    - Type mismatches
    - Constraint violations

    Examples:
        >>> raise ToolValidationError(
        ...     "Query parameter cannot be empty",
        ...     details={"parameter": "query", "value": ""}
        ... )
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize validation error.

        Args:
            message: Error message
            parameter: Name of the invalid parameter
            details: Additional error context
            error_code: Error code
        """
        self.parameter = parameter
        details = details or {}
        if parameter:
            details["parameter"] = parameter

        super().__init__(message, details, error_code or "VALIDATION_ERROR")


class LLMTimeoutError(LLMAPIException):
    """
    Raised when an LLM request times out.

    Used specifically for LLM-related timeouts to distinguish
    from general execution timeouts.

    Examples:
        >>> raise LLMTimeoutError(
        ...     "LLM request timed out after 30s",
        ...     details={"timeout": 30, "model": "gemma3:12b"}
        ... )
    """

    def __init__(
        self,
        message: str,
        timeout: Optional[int] = None,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize LLM timeout error.

        Args:
            message: Error message
            timeout: Timeout value in seconds
            model: Model name that timed out
            details: Additional error context
            error_code: Error code
        """
        self.timeout = timeout
        self.model = model
        details = details or {}
        if timeout:
            details["timeout"] = timeout
        if model:
            details["model"] = model

        super().__init__(message, details, error_code or "LLM_TIMEOUT")


class LLMResponseError(LLMAPIException):
    """
    Raised when LLM returns an invalid or unexpected response.

    Used for:
    - Parsing failures
    - Invalid response format
    - Missing required fields
    - Unexpected response structure

    Examples:
        >>> raise LLMResponseError(
        ...     "Failed to parse LLM response as JSON",
        ...     details={"response": "...", "expected_format": "JSON"}
        ... )
    """

    def __init__(
        self,
        message: str,
        response: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize LLM response error.

        Args:
            message: Error message
            response: The problematic response (truncated if long)
            details: Additional error context
            error_code: Error code
        """
        self.response = response
        details = details or {}
        if response:
            # Truncate response for logging
            truncated = response[:500] + "..." if len(response) > 500 else response
            details["response_preview"] = truncated

        super().__init__(message, details, error_code or "LLM_RESPONSE_ERROR")


class FileHandlerError(LLMAPIException):
    """
    Raised when file handling operations fail.

    Used for:
    - File not found
    - Unsupported file format
    - File parsing errors
    - File access/permission errors

    Examples:
        >>> raise FileHandlerError(
        ...     "Unsupported file format: .xyz",
        ...     file_path="/path/to/file.xyz",
        ...     details={"supported_formats": [".csv", ".json", ".xlsx"]}
        ... )
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize file handler error.

        Args:
            message: Error message
            file_path: Path to the problematic file
            details: Additional error context
            error_code: Error code
        """
        self.file_path = file_path
        details = details or {}
        if file_path:
            details["file_path"] = file_path

        super().__init__(message, details, error_code or "FILE_HANDLER_ERROR")


class CodeExecutionError(ToolExecutionError):
    """
    Raised when Python code execution fails.

    Specialized subclass of ToolExecutionError for code execution
    with additional context about the code and error details.

    Examples:
        >>> raise CodeExecutionError(
        ...     "Runtime error in generated code",
        ...     details={
        ...         "error_type": "ZeroDivisionError",
        ...         "line_number": 10,
        ...         "stderr": "..."
        ...     }
        ... )
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        stderr: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize code execution error.

        Args:
            message: Error message
            code: The code that failed (truncated if long)
            stderr: Standard error output
            details: Additional error context
            error_code: Error code
        """
        details = details or {}
        if code:
            truncated = code[:500] + "..." if len(code) > 500 else code
            details["code_preview"] = truncated
        if stderr:
            truncated_stderr = stderr[:500] + "..." if len(stderr) > 500 else stderr
            details["stderr"] = truncated_stderr

        super().__init__(
            message,
            tool_name="PythonCoder",
            details=details,
            error_code=error_code or "CODE_EXECUTION_ERROR"
        )


class ImportValidationError(ToolValidationError):
    """
    Raised when code contains blocked or invalid imports.

    Specialized validation error for import restrictions
    in the sandboxed code execution environment.

    Examples:
        >>> raise ImportValidationError(
        ...     "Blocked import detected: socket",
        ...     details={"blocked_imports": ["socket", "subprocess"]}
        ... )
    """

    def __init__(
        self,
        message: str,
        blocked_imports: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize import validation error.

        Args:
            message: Error message
            blocked_imports: List of blocked import names
            details: Additional error context
            error_code: Error code
        """
        details = details or {}
        if blocked_imports:
            details["blocked_imports"] = blocked_imports

        super().__init__(
            message,
            parameter="imports",
            details=details,
            error_code=error_code or "IMPORT_VALIDATION_ERROR"
        )


class AgentExecutionError(LLMAPIException):
    """
    Raised when an agent fails to complete its task.

    Used for high-level agent failures, such as:
    - Maximum iterations exceeded
    - Unable to generate valid action
    - Circular reasoning detected
    - Agent stuck in a loop

    Examples:
        >>> raise AgentExecutionError(
        ...     "Agent exceeded maximum iterations",
        ...     details={"max_iterations": 10, "current_iteration": 10}
        ... )
    """

    def __init__(
        self,
        message: str,
        agent_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize agent execution error.

        Args:
            message: Error message
            agent_type: Type of agent (e.g., "ReAct", "PlanExecute")
            details: Additional error context
            error_code: Error code
        """
        self.agent_type = agent_type
        details = details or {}
        if agent_type:
            details["agent_type"] = agent_type

        super().__init__(message, details, error_code or "AGENT_EXECUTION_ERROR")


class ConfigurationError(LLMAPIException):
    """
    Raised when there's a configuration problem.

    Used for:
    - Missing required configuration
    - Invalid configuration values
    - Configuration conflicts

    Examples:
        >>> raise ConfigurationError(
        ...     "Missing required API key: TAVILY_API_KEY",
        ...     details={"config_key": "TAVILY_API_KEY"}
        ... )
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize configuration error.

        Args:
            message: Error message
            config_key: The problematic configuration key
            details: Additional error context
            error_code: Error code
        """
        self.config_key = config_key
        details = details or {}
        if config_key:
            details["config_key"] = config_key

        super().__init__(message, details, error_code or "CONFIGURATION_ERROR")


class UnsupportedFileTypeError(FileHandlerError):
    """
    Raised when attempting to handle an unsupported file type.

    Specialized file handler error for file type/format issues.

    Examples:
        >>> raise UnsupportedFileTypeError(
        ...     "Unsupported file type: .xyz",
        ...     file_path="/path/to/file.xyz",
        ...     details={"supported_types": [".csv", ".json"]}
        ... )
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize unsupported file type error.

        Args:
            message: Error message
            file_path: Path to the file
            file_type: The unsupported file type
            details: Additional error context
            error_code: Error code
        """
        details = details or {}
        if file_type:
            details["file_type"] = file_type

        super().__init__(
            message,
            file_path=file_path,
            details=details,
            error_code=error_code or "UNSUPPORTED_FILE_TYPE"
        )
