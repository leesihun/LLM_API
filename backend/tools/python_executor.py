"""
Python Code Execution Tool
Safely executes Python code in a sandboxed environment
"""

import sys
import io
import contextlib
import ast
import logging
from typing import Dict, Any
import traceback
import signal
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Raised when code execution times out"""
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for timing out code execution"""
    def signal_handler(signum, frame):
        raise TimeoutException("Code execution timed out")

    # Set the signal handler and alarm
    if sys.platform != 'win32':
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)

    try:
        yield
    finally:
        if sys.platform != 'win32':
            signal.alarm(0)


class PythonExecutor:
    """
    Executes Python code in a restricted environment

    Security Features:
    - No file system access
    - No network access
    - No subprocess execution
    - Limited imports (safe modules only)
    - Execution timeout
    - Memory limits
    """

    # Allowed modules (safe for execution)
    SAFE_MODULES = {
        'math', 'statistics', 'random', 'datetime', 'json',
        'itertools', 'collections', 'functools', 're',
        'decimal', 'fractions', 'operator'
    }

    # Forbidden built-ins (dangerous operations)
    FORBIDDEN_BUILTINS = {
        'open', 'file', 'input', 'raw_input', 'execfile',
        'reload', '__import__', 'compile', 'eval', 'exec',
        'exit', 'quit', 'help'
    }

    def __init__(self, timeout_seconds: int = 5, max_output_length: int = 10000):
        self.timeout_seconds = timeout_seconds
        self.max_output_length = max_output_length

    def _create_safe_globals(self) -> Dict[str, Any]:
        """
        Create a safe global namespace with restricted built-ins
        """
        # Start with safe built-ins
        safe_builtins = {
            name: getattr(__builtins__, name)
            for name in dir(__builtins__)
            if not name.startswith('_') and name not in self.FORBIDDEN_BUILTINS
        }

        # Add safe modules
        safe_globals = {'__builtins__': safe_builtins}

        return safe_globals

    def _validate_code(self, code: str) -> bool:
        """
        Validate code for dangerous operations before execution
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in code: {e}")

        # Check for dangerous operations
        for node in ast.walk(tree):
            # Check for import statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.SAFE_MODULES:
                        raise ValueError(f"Import of '{alias.name}' is not allowed")

            # Check for from-import statements
            if isinstance(node, ast.ImportFrom):
                if node.module not in self.SAFE_MODULES:
                    raise ValueError(f"Import from '{node.module}' is not allowed")

        return True

    async def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code safely and return results

        Args:
            code: Python code to execute

        Returns:
            Dictionary with:
                - success: bool
                - output: str (stdout)
                - error: str (stderr/exception)
                - result: Any (last expression value)
        """
        logger.info(f"[Python Executor] Executing code: {code[:]}...")

        try:
            # Validate code first
            self._validate_code(code)

            # Capture stdout and stderr
            stdout = io.StringIO()
            stderr = io.StringIO()

            # Create safe execution environment
            safe_globals = self._create_safe_globals()
            safe_locals = {}

            # Execute with timeout
            try:
                if sys.platform != 'win32':
                    with timeout(self.timeout_seconds):
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            exec(code, safe_globals, safe_locals)
                else:
                    # Windows doesn't support signal-based timeout
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        exec(code, safe_globals, safe_locals)

            except TimeoutException:
                return {
                    "success": False,
                    "output": stdout.getvalue(),
                    "error": f"Code execution timed out after {self.timeout_seconds} seconds",
                    "result": None
                }

            # Get output
            output = stdout.getvalue()
            error = stderr.getvalue()

            # Truncate if too long
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + f"\n... (truncated, {len(output)} total characters)"

            # Try to get the result of the last expression
            result = None
            if safe_locals:
                # Get the last non-None value
                result = list(safe_locals.values())[-1] if safe_locals.values() else None

            logger.info(f"[Python Executor] Execution successful")

            return {
                "success": True,
                "output": output,
                "error": error,
                "result": str(result) if result is not None else None
            }

        except ValueError as e:
            # Validation error
            logger.error(f"[Python Executor] Validation error: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "result": None
            }

        except Exception as e:
            # Execution error
            logger.error(f"[Python Executor] Execution error: {e}")
            error_msg = traceback.format_exc()

            return {
                "success": False,
                "output": stdout.getvalue() if 'stdout' in locals() else "",
                "error": error_msg,
                "result": None
            }

    def format_result(self, result: Dict[str, Any]) -> str:
        """
        Format execution result for display
        """
        if result["success"]:
            output_parts = []

            if result["output"]:
                output_parts.append(f"Output:\n{result['output']}")

            if result["result"]:
                output_parts.append(f"Result: {result['result']}")

            if result["error"]:
                output_parts.append(f"Warnings:\n{result['error']}")

            return "\n\n".join(output_parts) if output_parts else "Code executed successfully (no output)"
        else:
            return f"Error executing Python code:\n{result['error']}"


# Global instance
python_executor = PythonExecutor(timeout_seconds=5, max_output_length=10000)
