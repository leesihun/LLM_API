"""Code executor for running Python code in isolated subprocess."""

import ast
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Security: Blocked imports
BLOCKED_IMPORTS = [
    "socket", "subprocess", "os.system", "eval", "exec",
    "__import__", "importlib", "shutil.rmtree", "pickle",
]

# Supported file types for input files
SUPPORTED_FILE_TYPES = [
    ".txt", ".md", ".log", ".rtf",  # Text
    ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",  # Data
    ".xlsx", ".xls", ".xlsm", ".docx", ".doc",  # Office
    ".pdf",  # PDF
    ".dat", ".h5", ".hdf5", ".nc", ".parquet", ".feather",  # Scientific
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg",  # Images
    ".zip", ".tar", ".gz", ".bz2", ".7z",  # Compressed
]


class CodeExecutor:
    """
    Executes Python code in isolated subprocess with security restrictions.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 512,
        execution_base_dir: str = "./data/scratch"
    ):
        """
        Initialize code executor.

        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            execution_base_dir: Base directory for temporary execution folders
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.execution_base_dir = Path(execution_base_dir).resolve()
        self.execution_base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[CodeExecutor] Initialized with timeout={timeout}s, max_memory={max_memory_mb}MB")

    def validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate that code only imports safe packages.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module in BLOCKED_IMPORTS:
                        issues.append(f"Blocked import detected: {module}")

            elif isinstance(node, ast.ImportFrom):
                module = node.module.split('.')[0] if node.module else ''
                if module in BLOCKED_IMPORTS:
                    issues.append(f"Blocked import detected: {module}")

            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', '__import__']:
                        issues.append(f"Dangerous function call detected: {node.func.id}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def execute(
        self,
        code: str,
        input_files: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
        stage_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code in isolated subprocess.

        Args:
            code: Python code to execute
            input_files: Optional dict mapping original file paths to their basenames
            session_id: Optional session ID to use as execution directory name
            stage_name: Optional stage name for saving code (e.g., "verify1", "exec2")

        Returns:
            Dict with keys: success, output, error, execution_time, return_code
        """
        # Use session_id if provided, otherwise generate unique ID
        execution_id = session_id if session_id else uuid.uuid4().hex
        execution_dir = self.execution_base_dir / execution_id

        try:
            # Create execution directory
            execution_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[CodeExecutor] Using execution directory: {execution_dir}")

            # Copy input files to execution directory
            if input_files:
                for original_path, basename in input_files.items():
                    target_path = execution_dir / basename
                    shutil.copy2(original_path, target_path)
                    logger.debug(f"[CodeExecutor] Copied {original_path} -> {target_path}")

            # Write code to script.py (main execution file)
            script_path = execution_dir / "script.py"
            script_path.write_text(code, encoding='utf-8')
            logger.debug(f"[CodeExecutor] Wrote code to {script_path}")

            # ALSO save to stage-specific file if stage_name provided
            if stage_name:
                stage_script_path = execution_dir / f"script_{stage_name}.py"
                stage_script_path.write_text(code, encoding='utf-8')
                logger.info(f"[CodeExecutor] 💾 Saved stage code to {stage_script_path.name}")

            # Execute code using current Python interpreter
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                timeout=self.timeout,
                cwd=str(execution_dir),
                text=True
            )
            execution_time = time.time() - start_time

            # Log execution result
            if result.returncode == 0:
                logger.success("Code execution succeeded", f"{execution_time:.2f}s")
            else:
                logger.failure("Code execution failed", f"Return code: {result.returncode}")

            # Log stdout
            if result.stdout:
                logger.multiline(result.stdout, title="STDOUT", max_lines=50)
            else:
                logger.info("STDOUT: (empty)")

            # Log stderr
            if result.stderr:
                if result.returncode != 0:
                    logger.multiline(result.stderr, title="STDERR - ERROR", max_lines=30)
                else:
                    logger.multiline(result.stderr, title="STDERR - WARNING", max_lines=20)

            # Enhanced success detection: check for error patterns in stdout
            has_error_in_output = False
            if result.returncode == 0 and result.stdout:
                error_patterns = [
                    "Error:", "error:", "ERROR:",
                    "Failed:", "failed:", "FAILED:",
                    "Exception:", "exception:",
                    "not found", "Not found", "NOT FOUND",
                    "does not contain", "does not exist",
                    "No valid", "no valid",
                    "Invalid", "invalid"
                ]
                stdout_lower = result.stdout.lower()
                for pattern in error_patterns:
                    if pattern.lower() in stdout_lower:
                        has_error_in_output = True
                        logger.warning(f"[CodeExecutor] ⚠️  Error pattern detected: '{pattern}'")
                        break

            # Determine actual success status
            is_success = result.returncode == 0 and not has_error_in_output

            if has_error_in_output:
                logger.error(f"[CodeExecutor] Code printed error messages despite return code 0")

            return {
                "success": is_success,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else (result.stdout if has_error_in_output else None),
                "execution_time": execution_time,
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            logger.error(f"[CodeExecutor] Execution timeout after {self.timeout}s")
            return {
                "success": False,
                "output": "",
                "error": f"Execution timeout after {self.timeout} seconds",
                "execution_time": self.timeout,
                "return_code": -1
            }

        except Exception as e:
            logger.error(f"[CodeExecutor] Execution failed: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": 0,
                "return_code": -1
            }

        finally:
            # Cleanup execution directory (only if temporary, not session-based)
            if not session_id:
                try:
                    if execution_dir.exists():
                        shutil.rmtree(execution_dir)
                        logger.debug(f"[CodeExecutor] Cleaned up temporary directory")
                except Exception as e:
                    logger.warning(f"[CodeExecutor] Failed to cleanup: {e}")
            else:
                logger.debug(f"[CodeExecutor] Keeping session directory")

    def validate_file_type(self, file_path: str) -> bool:
        """
        Check if file type is supported.

        Args:
            file_path: Path to file

        Returns:
            True if file type is supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in SUPPORTED_FILE_TYPES


# Backward compatibility
PythonExecutor = CodeExecutor
