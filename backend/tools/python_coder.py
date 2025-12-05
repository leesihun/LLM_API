"""
Python Coder Tool
=================
Executes Python code directly in a sandboxed environment.
No LLM generation - code is provided directly by the ReAct agent.

Version: 2.0.0 - Simplified to pure execution (no internal LLM)
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from backend.core import BaseTool, ToolResult
from backend.config.settings import settings
from backend.utils.logging_utils import get_logger
from backend.tools.code_sandbox import CodeSandbox

logger = get_logger(__name__)


class PythonCoderTool(BaseTool):
    """
    Executes Python code in a sandboxed environment.
    Code is provided directly - no internal LLM generation.
    """

    def __init__(self):
        super().__init__()
        self.sandbox = CodeSandbox(
            timeout=settings.python_code_timeout,
            execution_base_dir=settings.python_code_execution_dir
        )
        logger.info("[PythonCoder] Initialized (execution-only mode, no LLM)")

    def validate_inputs(self, **kwargs) -> bool:
        """Validate that code is provided."""
        code = kwargs.get("code", "")
        return bool(code and code.strip())

    async def execute(
        self,
        code: str,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        stage_prefix: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute Python code directly.

        Args:
            code: Python code to execute
            file_paths: Optional list of file paths to make available
            session_id: Optional session ID for execution isolation
            stage_prefix: Optional prefix for script naming

        Returns:
            ToolResult with execution results
        """
        self._log_execution_start(code=code[:100] + "..." if len(code) > 100 else code)

        if not self.validate_inputs(code=code):
            return self._handle_validation_error("Code cannot be empty", parameter="code")

        try:
            result = await self.execute_code(
                code=code,
                file_paths=file_paths,
                session_id=session_id,
                stage_prefix=stage_prefix
            )

            if result.get("success"):
                return ToolResult.success_result(
                    output=result,
                    execution_time=self._elapsed_time()
                )
            else:
                return ToolResult.failure_result(
                    error=result.get("error", "Code execution failed"),
                    error_type="CodeExecutionError",
                    metadata={"output": result.get("output", "")},
                    execution_time=self._elapsed_time()
                )

        except Exception as e:
            return self._handle_error(e, "execute")

    async def execute_code(
        self,
        code: str,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        stage_prefix: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute code in sandbox.

        Args:
            code: Python code string to execute
            file_paths: List of file paths to make available in sandbox
            session_id: Session ID for execution directory
            stage_prefix: Prefix for script file naming

        Returns:
            Dict with success, output, error, execution_time, created_files
        """
        # Prepare input files
        input_files = {}
        if file_paths:
            for fp in file_paths:
                path = Path(fp)
                if path.exists():
                    input_files[str(path)] = path.name
                    logger.debug(f"[PythonCoder] Added file: {path.name}")

        logger.info(f"[PythonCoder] Executing code ({len(code)} chars), files: {list(input_files.values())}")

        # Execute in sandbox
        result = self.sandbox.execute(
            code=code,
            input_files=input_files,
            session_id=session_id,
            stage_name=stage_prefix or "run"
        )

        if result["success"]:
            logger.info(f"[PythonCoder] Execution successful in {result.get('execution_time', 0):.2f}s")
            return {
                "success": True,
                "output": result["output"],
                "code": code,
                "execution_time": result.get("execution_time", 0),
                "created_files": result.get("created_files", [])
            }
        else:
            logger.warning(f"[PythonCoder] Execution failed: {result.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": result.get("error", "Unknown execution error"),
                "output": result.get("output", ""),
                "code": code,
                "execution_time": result.get("execution_time", 0)
            }


# Global instance
python_coder_tool = PythonCoderTool()
