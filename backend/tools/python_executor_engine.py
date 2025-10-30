"""
Python Code Executor Engine

Low-level subprocess execution engine with security restrictions.
Executes Python code in isolated temporary directories with resource limits.
"""

import ast
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BLOCKED_IMPORTS = [
    "socket", "subprocess", "os.system", "eval", "exec",
    "__import__", "importlib", "shutil.rmtree", "pickle",
]

SUPPORTED_FILE_TYPES = [
    ".txt", ".md", ".log", ".rtf",  # Text
    ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",  # Data
    ".xlsx", ".xls", ".xlsm", ".docx", ".doc",  # Office
    ".pdf",  # PDF
    ".dat", ".h5", ".hdf5", ".nc", ".parquet", ".feather",  # Scientific
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg",  # Images
    ".zip", ".tar", ".gz", ".bz2", ".7z",  # Compressed
]


class PythonExecutor:
    """
    Executes Python code in isolated subprocess with security restrictions.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 512,
        execution_base_dir: str = "./data/code_execution"
    ):
        """
        Initialize Python executor.

        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB (future: cgroups)
            execution_base_dir: Base directory for temporary execution folders
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.execution_base_dir = Path(execution_base_dir).resolve()
        self.execution_base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[PythonExecutor] Initialized with timeout={timeout}s, max_memory={max_memory_mb}MB")

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

    def execute_code(
        self,
        code: str,
        input_files: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code in isolated subprocess.

        Args:
            code: Python code to execute
            input_files: Optional dict mapping original file paths to their basenames

        Returns:
            Dict with keys: success, output, error, execution_time
        """
        import time

        execution_id = uuid.uuid4().hex
        execution_dir = self.execution_base_dir / execution_id

        try:
            # Create execution directory
            execution_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[PythonExecutor] Created execution directory: {execution_dir}")

            # Copy input files to execution directory
            if input_files:
                for original_path, basename in input_files.items():
                    target_path = execution_dir / basename
                    shutil.copy2(original_path, target_path)
                    logger.debug(f"[PythonExecutor] Copied {original_path} -> {target_path}")

            # Write code to script.py
            script_path = execution_dir / "script.py"
            script_path.write_text(code, encoding='utf-8')
            logger.debug(f"[PythonExecutor] Wrote code to {script_path}")

            # Execute code
            start_time = time.time()
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                timeout=self.timeout,
                cwd=str(execution_dir),
                text=True
            )
            execution_time = time.time() - start_time

            logger.info(f"[PythonExecutor] Execution completed in {execution_time:.2f}s")

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "execution_time": execution_time,
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            logger.error(f"[PythonExecutor] Execution timeout after {self.timeout}s")
            return {
                "success": False,
                "output": "",
                "error": f"Execution timeout after {self.timeout} seconds",
                "execution_time": self.timeout,
                "return_code": -1
            }

        except Exception as e:
            logger.error(f"[PythonExecutor] Execution failed: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": 0,
                "return_code": -1
            }

        finally:
            # Cleanup execution directory
            try:
                if execution_dir.exists():
                    shutil.rmtree(execution_dir)
                    logger.debug(f"[PythonExecutor] Cleaned up {execution_dir}")
            except Exception as e:
                logger.warning(f"[PythonExecutor] Failed to cleanup {execution_dir}: {e}")

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

    def get_safe_packages(self) -> List[str]:
        """Get list of safe packages that can be imported."""
        return SAFE_PACKAGES.copy()

    def get_blocked_imports(self) -> List[str]:
        """Get list of blocked imports."""
        return BLOCKED_IMPORTS.copy()

    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file types."""
        return SUPPORTED_FILE_TYPES.copy()
