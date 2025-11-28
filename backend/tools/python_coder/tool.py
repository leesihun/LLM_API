"""
Python Coder Tool - BaseTool Implementation
============================================
Wraps PythonCoderTool orchestrator with BaseTool interface.

This provides a standardized interface for the ReAct agent while
maintaining backward compatibility with existing code.

Created: 2025-01-20
Version: 3.0.0 (with BaseTool)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core import BaseTool, ToolResult, CodeExecutionResult
from backend.utils.logging_utils import get_logger
from backend.tools.python_coder.orchestrator import PythonCoderTool as PythonCoderOrchestrator

logger = get_logger(__name__)


class PythonCoderTool(BaseTool):
    """
    Python code generation and execution tool with BaseTool interface.
    
    Features:
    - Generates Python code from natural language
    - Executes code safely in sandboxed environment
    - Iterative verification and fixing
    - File handling and context extraction
    - Returns standardized ToolResult
    
    Usage:
        >>> tool = PythonCoderTool()
        >>> result = await tool.execute(
        ...     query="Calculate the mean of [1,2,3,4,5]",
        ...     file_paths=None
        ... )
        >>> print(result.output)  # "The mean is 3.0"
    """
    
    def __init__(self):
        """Initialize Python Coder Tool"""
        super().__init__()
        # Use existing orchestrator for all the heavy lifting
        self.orchestrator = PythonCoderOrchestrator()
        logger.info("[PythonCoderTool] Initialized with BaseTool interface")
    
    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        is_prestep: bool = False,
        stage_prefix: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
        plan_context: Optional[dict] = None,
        react_context: Optional[dict] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute Python code generation and execution.
        
        Args:
            query: Task description or question to answer with code
            context: Optional additional context
            file_paths: Optional list of input file paths
            session_id: Optional session ID for stateful execution
            is_prestep: Whether called from ReAct pre-step
            stage_prefix: Optional stage prefix for naming
            conversation_history: Past conversation turns
            plan_context: Plan-Execute workflow context
            react_context: ReAct iteration context
            **kwargs: Additional parameters
            
        Returns:
            ToolResult with execution outcome
        """
        self._log_execution_start(
            query=query[:100],
            file_count=len(file_paths) if file_paths else 0,
            session_id=session_id
        )
        
        try:
            # Validate inputs
            if not self.validate_inputs(query=query):
                return self._handle_validation_error(
                    "Query cannot be empty",
                    parameter="query"
                )
            
            # Execute via orchestrator
            result_dict = await self.orchestrator.execute_code_task(
                query=query,
                context=context,
                file_paths=file_paths,
                session_id=session_id,
                is_prestep=is_prestep,
                stage_prefix=stage_prefix,
                conversation_history=conversation_history,
                plan_context=plan_context,
                react_context=react_context
            )
            
            # Convert orchestrator result to ToolResult
            tool_result = self._convert_to_tool_result(result_dict)
            
            self._log_execution_end(tool_result)
            return tool_result
            
        except Exception as e:
            return self._handle_error(e, "execute")
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate tool inputs.
        
        Args:
            **kwargs: Must contain 'query' key
            
        Returns:
            True if inputs are valid
        """
        query = kwargs.get("query", "")
        
        # Query must be non-empty string
        if not isinstance(query, str) or len(query.strip()) == 0:
            return False
        
        # File paths must be list if provided
        file_paths = kwargs.get("file_paths")
        if file_paths is not None and not isinstance(file_paths, list):
            return False
        
        return True
    
    def _convert_to_tool_result(self, result_dict: Dict[str, Any]) -> ToolResult:
        """
        Convert orchestrator result dictionary to ToolResult with enhanced error history.

        Args:
            result_dict: Result from orchestrator.execute_code_task()

        Returns:
            Standardized ToolResult
        """
        success = result_dict.get("success", False)

        # Determine output
        if success:
            output = result_dict.get("output", "")
            # If output is empty but code exists, use a default message
            if not output and result_dict.get("code"):
                output = "Code executed successfully"
        else:
            output = None

        # Enhanced error with attempt history
        error = None
        error_type = None
        if not success:
            error = result_dict.get("error", "Unknown error")
            error_type = result_dict.get("error_type", "CodeExecutionError")

            # Append attempt summary if multiple attempts were made
            attempt_history = result_dict.get("attempt_history", [])
            total_attempts = result_dict.get("total_attempts", len(attempt_history))

            if total_attempts > 1 and attempt_history:
                error_parts = [error, f"\n\n{'='*60}"]
                error_parts.append(f"EXECUTION HISTORY ({total_attempts} attempts made)")
                error_parts.append('='*60)

                for attempt in attempt_history:
                    attempt_num = attempt.get("attempt", "?")
                    exec_success = attempt.get("execution_success", False)
                    status = "✓ SUCCESS" if exec_success else "✗ FAILED"

                    error_parts.append(f"\nAttempt {attempt_num}: {status}")

                    if not exec_success:
                        err_type = attempt.get("error_type", ("Unknown", ""))
                        if isinstance(err_type, tuple):
                            err_type = err_type[0]
                        error_parts.append(f"  Error Type: {err_type}")

                        # Include brief error message (truncated)
                        attempt_error = attempt.get("execution_error", "")
                        if attempt_error:
                            truncated_error = attempt_error[:200] + "..." if len(attempt_error) > 200 else attempt_error
                            error_parts.append(f"  Error: {truncated_error}")
                    else:
                        # Include execution time for successful attempts
                        exec_time = attempt.get("execution_time", 0)
                        error_parts.append(f"  Execution Time: {exec_time:.2f}s")

                error = "\n".join(error_parts)
        
        # Build metadata
        metadata = {
            "code": result_dict.get("code", ""),
            "execution_time": result_dict.get("execution_time"),
            "total_attempts": result_dict.get("total_attempts", 1),
            "verification_attempts": result_dict.get("verification_attempts", 0),
            "files_created": result_dict.get("files_created", []),
            "session_id": result_dict.get("session_id")
        }
        
        # Include verification history if present
        if "verification_history" in result_dict:
            metadata["verification_history"] = result_dict["verification_history"]
        
        # Include execution history if present
        if "execution_attempts_history" in result_dict:
            metadata["execution_attempts_history"] = result_dict["execution_attempts_history"]
        
        return ToolResult(
            success=success,
            output=output,
            error=error,
            error_type=error_type,
            metadata=metadata,
            execution_time=result_dict.get("execution_time")
        )


# Global singleton instance for backward compatibility
python_coder_tool = PythonCoderTool()


# Backward compatibility exports
__all__ = [
    'PythonCoderTool',
    'python_coder_tool',
]
