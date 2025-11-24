"""
Base Tool Module
===============
Abstract base class for all tools in the LLM API system.

Provides standardized interface for tool execution:
- Consistent execute() signature
- Input validation
- Error handling
- Lazy LLM loading
- Logging

All tools should inherit from BaseTool to ensure consistency.

Version: 2.0.0
Created: 2025-01-20
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time

from backend.core.result_types import ToolResult
from backend.core.exceptions import (
    ToolExecutionError,
    ToolValidationError,
    LLMTimeoutError
)
from backend.utils.logging_utils import get_logger
from backend.utils.llm_factory import LLMFactory


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    All tools must implement:
    - execute(): Main execution logic
    - validate_inputs(): Input validation

    Provides:
    - Standardized error handling
    - Lazy LLM loading
    - Execution timing
    - Consistent logging

    Example:
        >>> class MyTool(BaseTool):
        ...     async def execute(self, query: str, **kwargs) -> ToolResult:
        ...         if not self.validate_inputs(query=query):
        ...             return self._handle_validation_error("Invalid query")
        ...
        ...         # Tool logic here
        ...         result = do_something(query)
        ...
        ...         return ToolResult.success_result(
        ...             output=result,
        ...             execution_time=self._elapsed_time()
        ...         )
        ...
        ...     def validate_inputs(self, **kwargs) -> bool:
        ...         query = kwargs.get("query", "")
        ...         return len(query.strip()) > 0
    """

    def __init__(self):
        """Initialize the base tool with logger and LLM placeholder"""
        self.logger = get_logger(self.__class__.__name__)
        self._llm = None
        self._start_time = None

    @abstractmethod
    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute the tool with given inputs.

        This is the main entry point for tool execution. All tools
        must implement this method with a consistent signature.

        Args:
            query: The main query/task for the tool
            context: Optional context information
            **kwargs: Additional tool-specific parameters

        Returns:
            ToolResult containing execution outcome

        Raises:
            ToolExecutionError: If execution fails
            ToolValidationError: If inputs are invalid
        """
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate tool inputs before execution.

        Args:
            **kwargs: Input parameters to validate

        Returns:
            True if inputs are valid, False otherwise

        Example:
            >>> def validate_inputs(self, **kwargs) -> bool:
            ...     query = kwargs.get("query", "")
            ...     file_paths = kwargs.get("file_paths", [])
            ...     return len(query.strip()) > 0 and isinstance(file_paths, list)
        """
        pass

    @property
    def name(self) -> str:
        """
        Get the tool name.

        Returns:
            The tool's class name
        """
        return self.__class__.__name__

    def _start_timer(self):
        """Start execution timer"""
        self._start_time = time.time()

    def _elapsed_time(self) -> float:
        """
        Get elapsed time since timer started.

        Returns:
            Elapsed time in seconds
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def _get_llm(self, **config):
        """
        Get or create LLM instance with lazy loading.

        Args:
            **config: Configuration overrides for LLM

        Returns:
            ChatOllama instance

        Example:
            >>> llm = self._get_llm(temperature=0.7)
            >>> response = llm.invoke("Generate code...")
        """
        if self._llm is None:
            self.logger.debug(f"[{self.name}] Creating LLM instance")
            self._llm = LLMFactory.create_llm(**config)
        return self._llm

    def _get_coder_llm(self, **config):
        """
        Get LLM optimized for code generation.

        Args:
            **config: Configuration overrides

        Returns:
            ChatOllama instance configured for coding
        """
        return LLMFactory.create_coder_llm(**config)

    def _handle_error(
        self,
        e: Exception,
        context: str,
        include_traceback: bool = True
    ) -> ToolResult:
        """
        Handle errors in a standardized way.

        Logs the error and creates a ToolResult with error information.

        Args:
            e: The exception that occurred
            context: Context string describing where error occurred
            include_traceback: Whether to include traceback in logs

        Returns:
            ToolResult indicating failure

        Example:
            >>> try:
            ...     result = risky_operation()
            ... except Exception as e:
            ...     return self._handle_error(e, "risky_operation")
        """
        self.logger.error(
            f"[{self.name}] Error in {context}: {e}",
            exc_info=include_traceback
        )

        # Get execution time if timer was started
        execution_time = self._elapsed_time() if self._start_time else None

        return ToolResult.failure_result(
            error=str(e),
            error_type=type(e).__name__,
            metadata={
                "context": context,
                "tool_name": self.name
            },
            execution_time=execution_time
        )

    def _handle_validation_error(
        self,
        message: str,
        parameter: Optional[str] = None,
        **details
    ) -> ToolResult:
        """
        Handle validation errors.

        Args:
            message: Error message
            parameter: Name of invalid parameter
            **details: Additional error details

        Returns:
            ToolResult indicating validation failure

        Example:
            >>> if not query:
            ...     return self._handle_validation_error(
            ...         "Query cannot be empty",
            ...         parameter="query"
            ...     )
        """
        self.logger.warning(f"[{self.name}] Validation error: {message}")

        metadata = {"tool_name": self.name}
        if parameter:
            metadata["parameter"] = parameter
        metadata.update(details)

        return ToolResult.failure_result(
            error=message,
            error_type="ToolValidationError",
            metadata=metadata
        )

    def _handle_timeout(
        self,
        operation: str,
        timeout: int,
        **details
    ) -> ToolResult:
        """
        Handle timeout errors.

        Args:
            operation: Name of the operation that timed out
            timeout: Timeout value
            **details: Additional details

        Returns:
            ToolResult indicating timeout

        Example:
            >>> return self._handle_timeout(
            ...     operation="LLM request",
            ...     timeout=30,
            ...     model="gemma3:12b"
            ... )
        """
        message = f"{operation} timed out after {timeout}s"
        self.logger.error(f"[{self.name}] {message}")

        metadata = {
            "tool_name": self.name,
            "operation": operation,
            "timeout": timeout
        }
        metadata.update(details)

        execution_time = self._elapsed_time() if self._start_time else None

        return ToolResult.failure_result(
            error=message,
            error_type="LLMTimeoutError",
            metadata=metadata,
            execution_time=execution_time
        )

    def _log_execution_start(self, **params):
        """
        Log the start of tool execution.

        Args:
            **params: Execution parameters to log
        """
        self._start_timer()
        self.logger.info(f"[{self.name}] Starting execution")

        # Log important parameters (truncate long values)
        if params:
            log_params = {}
            for key, value in params.items():
                if isinstance(value, str) and len(value) > 100:
                    log_params[key] = value[:100] + "..."
                elif isinstance(value, list) and len(value) > 5:
                    log_params[key] = f"[{len(value)} items]"
                else:
                    log_params[key] = value

            self.logger.debug(f"[{self.name}] Parameters: {log_params}")

    def _log_execution_end(self, result: ToolResult):
        """
        Log the end of tool execution.

        Args:
            result: The execution result
        """
        elapsed = self._elapsed_time()

        if result.success:
            self.logger.info(
                f"[{self.name}] Execution completed successfully "
                f"in {elapsed:.2f}s"
            )
        else:
            self.logger.error(
                f"[{self.name}] Execution failed after {elapsed:.2f}s: "
                f"{result.error_type} - {result.error}"
            )

    async def execute_with_logging(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute the tool with automatic logging.

        Wrapper around execute() that adds automatic start/end logging.

        Args:
            query: The main query/task
            context: Optional context
            **kwargs: Additional parameters

        Returns:
            ToolResult from execution

        Example:
            >>> result = await tool.execute_with_logging(
            ...     query="Analyze this data",
            ...     file_paths=["data.csv"]
            ... )
        """
        self._log_execution_start(query=query[:100], **kwargs)

        try:
            result = await self.execute(query, context, **kwargs)
            self._log_execution_end(result)
            return result
        except Exception as e:
            result = self._handle_error(e, "execute_with_logging")
            self._log_execution_end(result)
            return result


class SyncBaseTool(ABC):
    """
    Synchronous version of BaseTool for tools that don't need async.

    Use this for tools with synchronous operations only.

    Example:
        >>> class SimpleCalculator(SyncBaseTool):
        ...     def execute(self, query: str, **kwargs) -> ToolResult:
        ...         # Synchronous calculation
        ...         result = eval(query)
        ...         return ToolResult.success_result(str(result))
    """

    def __init__(self):
        """Initialize the base tool"""
        self.logger = get_logger(self.__class__.__name__)
        self._llm = None
        self._start_time = None

    @abstractmethod
    def execute(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute the tool synchronously"""
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """Validate inputs"""
        pass

    @property
    def name(self) -> str:
        """Get tool name"""
        return self.__class__.__name__

    # Include same helper methods as BaseTool
    def _start_timer(self):
        """Start execution timer"""
        self._start_time = time.time()

    def _elapsed_time(self) -> float:
        """Get elapsed time"""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def _get_llm(self, **config):
        """Get LLM instance"""
        if self._llm is None:
            self.logger.debug(f"[{self.name}] Creating LLM instance")
            self._llm = LLMFactory.create_llm(**config)
        return self._llm

    def _handle_error(self, e: Exception, context: str) -> ToolResult:
        """Handle errors"""
        self.logger.error(f"[{self.name}] Error in {context}: {e}", exc_info=True)
        execution_time = self._elapsed_time() if self._start_time else None

        return ToolResult.failure_result(
            error=str(e),
            error_type=type(e).__name__,
            metadata={"context": context, "tool_name": self.name},
            execution_time=execution_time
        )

    def _handle_validation_error(
        self,
        message: str,
        parameter: Optional[str] = None,
        **details
    ) -> ToolResult:
        """Handle validation errors"""
        self.logger.warning(f"[{self.name}] Validation error: {message}")

        metadata = {"tool_name": self.name}
        if parameter:
            metadata["parameter"] = parameter
        metadata.update(details)

        return ToolResult.failure_result(
            error=message,
            error_type="ToolValidationError",
            metadata=metadata
        )
