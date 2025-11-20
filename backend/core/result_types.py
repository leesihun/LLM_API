"""
Result Types Module
==================
Standard result types for tool execution and API responses.

Provides Pydantic models for:
- ToolResult: Standard return type for all tools
- ExecutionMetadata: Metadata about execution
- StepResult: Individual step results for multi-step processes

Version: 2.0.0
Created: 2025-01-20
"""

from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
from datetime import datetime
from enum import Enum


class ExecutionStatus(str, Enum):
    """Execution status enumeration"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolResult(BaseModel):
    """
    Standard result type returned by all tools.

    This provides a consistent interface for tool execution results,
    making it easier to handle errors, extract outputs, and track metadata.

    Attributes:
        success: Whether the tool execution succeeded
        output: The main output/result from the tool
        error: Error message if execution failed
        error_type: Type/class name of the error
        metadata: Additional metadata about execution
        execution_time: Time taken for execution in seconds
        timestamp: When the result was generated

    Examples:
        >>> # Success case
        >>> result = ToolResult(
        ...     success=True,
        ...     output="The sum is 15",
        ...     metadata={"code": "print(sum([1,2,3,4,5]))"},
        ...     execution_time=0.5
        ... )

        >>> # Failure case
        >>> result = ToolResult(
        ...     success=False,
        ...     error="Division by zero",
        ...     error_type="ZeroDivisionError",
        ...     metadata={"line": 10}
        ... )
    """

    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with proper serialization.

        Returns:
            Dictionary representation of the result
        """
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "error_type": self.error_type,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        """String representation for logging"""
        if self.success:
            return f"ToolResult(success=True, output_length={len(self.output or '')})"
        else:
            return f"ToolResult(success=False, error={self.error_type}: {self.error})"

    @classmethod
    def success_result(
        cls,
        output: str,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ) -> "ToolResult":
        """
        Create a success result.

        Args:
            output: The successful output
            metadata: Optional metadata
            execution_time: Optional execution time

        Returns:
            ToolResult indicating success
        """
        return cls(
            success=True,
            output=output,
            metadata=metadata or {},
            execution_time=execution_time
        )

    @classmethod
    def failure_result(
        cls,
        error: str,
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ) -> "ToolResult":
        """
        Create a failure result.

        Args:
            error: Error message
            error_type: Type of error
            metadata: Optional metadata
            execution_time: Optional execution time

        Returns:
            ToolResult indicating failure
        """
        return cls(
            success=False,
            error=error,
            error_type=error_type or "Error",
            metadata=metadata or {},
            execution_time=execution_time
        )


class ExecutionMetadata(BaseModel):
    """
    Metadata about tool execution.

    Provides detailed information about how a tool was executed,
    useful for debugging, monitoring, and optimization.

    Attributes:
        tool_name: Name of the tool
        model_used: LLM model used (if applicable)
        iterations: Number of iterations/attempts
        tokens_used: Approximate tokens used
        cache_hit: Whether result was cached
        additional_info: Any additional metadata
    """

    tool_name: str
    model_used: Optional[str] = None
    iterations: int = 1
    tokens_used: Optional[int] = None
    cache_hit: bool = False
    additional_info: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tool_name": self.tool_name,
            "model_used": self.model_used,
            "iterations": self.iterations,
            "tokens_used": self.tokens_used,
            "cache_hit": self.cache_hit,
            "additional_info": self.additional_info
        }


class StepResult(BaseModel):
    """
    Result of an individual step in a multi-step process.

    Used by agents and orchestrators to track progress through
    multi-step workflows like ReAct loops or Plan-Execute chains.

    Attributes:
        step_number: The step number (1-indexed)
        step_type: Type of step (e.g., "thought", "action", "observation")
        success: Whether the step succeeded
        output: Output from this step
        error: Error if step failed
        metadata: Additional step metadata
        timestamp: When the step completed

    Examples:
        >>> step = StepResult(
        ...     step_number=1,
        ...     step_type="thought",
        ...     success=True,
        ...     output="I need to search for weather data",
        ...     metadata={"agent": "ReAct"}
        ... )
    """

    step_number: int
    step_type: str
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step_number": self.step_number,
            "step_type": self.step_type,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class MultiStepResult(BaseModel):
    """
    Result from a multi-step tool or agent execution.

    Aggregates results from multiple steps and provides
    overall success status and final output.

    Attributes:
        success: Overall success status
        final_output: Final aggregated output
        steps: List of individual step results
        error: Overall error message if failed
        metadata: Execution metadata
        execution_time: Total execution time

    Examples:
        >>> result = MultiStepResult(
        ...     success=True,
        ...     final_output="Task completed successfully",
        ...     steps=[step1, step2, step3],
        ...     metadata={"total_steps": 3},
        ...     execution_time=5.2
        ... )
    """

    success: bool
    final_output: Optional[str] = None
    steps: List[StepResult] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def add_step(self, step: StepResult):
        """
        Add a step result to the collection.

        Args:
            step: The step result to add
        """
        self.steps.append(step)

    def get_successful_steps(self) -> List[StepResult]:
        """
        Get all successful steps.

        Returns:
            List of successful step results
        """
        return [step for step in self.steps if step.success]

    def get_failed_steps(self) -> List[StepResult]:
        """
        Get all failed steps.

        Returns:
            List of failed step results
        """
        return [step for step in self.steps if not step.success]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "final_output": self.final_output,
            "steps": [step.to_dict() for step in self.steps],
            "error": self.error,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }


class CodeExecutionResult(ToolResult):
    """
    Specialized result for code execution.

    Extends ToolResult with code-specific fields like
    stdout, stderr, exit code, and generated code.

    Attributes:
        code: The executed code
        stdout: Standard output
        stderr: Standard error
        exit_code: Process exit code
        variables: Variables from execution context
    """

    code: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    variables: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_execution(
        cls,
        success: bool,
        code: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
        variables: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ) -> "CodeExecutionResult":
        """
        Create result from code execution.

        Args:
            success: Whether execution succeeded
            code: The executed code
            stdout: Standard output
            stderr: Standard error
            exit_code: Exit code
            variables: Captured variables
            execution_time: Execution time

        Returns:
            CodeExecutionResult instance
        """
        # Determine output and error
        output = stdout if success else None
        error = stderr if not success else None

        return cls(
            success=success,
            output=output,
            error=error,
            error_type="CodeExecutionError" if not success else None,
            code=code,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            variables=variables or {},
            execution_time=execution_time,
            metadata={
                "exit_code": exit_code,
                "has_stdout": bool(stdout),
                "has_stderr": bool(stderr),
                "variable_count": len(variables or {})
            }
        )


class FileAnalysisResult(ToolResult):
    """
    Specialized result for file analysis.

    Extends ToolResult with file-specific fields like
    file type, size, structure, and analysis details.

    Attributes:
        file_path: Path to analyzed file
        file_type: Type of file (e.g., "csv", "json")
        file_size: Size in bytes
        structure: File structure information
        preview: Preview of file contents
    """

    file_path: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    structure: Dict[str, Any] = Field(default_factory=dict)
    preview: Optional[str] = None

    @classmethod
    def from_analysis(
        cls,
        success: bool,
        file_path: str,
        file_type: str,
        analysis: str,
        structure: Optional[Dict[str, Any]] = None,
        file_size: Optional[int] = None,
        preview: Optional[str] = None,
        execution_time: Optional[float] = None
    ) -> "FileAnalysisResult":
        """
        Create result from file analysis.

        Args:
            success: Whether analysis succeeded
            file_path: Path to file
            file_type: Type of file
            analysis: Analysis text
            structure: File structure info
            file_size: File size in bytes
            preview: File preview
            execution_time: Execution time

        Returns:
            FileAnalysisResult instance
        """
        return cls(
            success=success,
            output=analysis,
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
            structure=structure or {},
            preview=preview,
            execution_time=execution_time,
            metadata={
                "file_type": file_type,
                "file_size": file_size,
                "has_structure": bool(structure)
            }
        )
