"""
Python Code Executor Tool
Executes Python code in sandboxed environment within session scratch directory
"""
import os
import sys
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import config


class PythonCoderTool:
    """
    Execute Python code in sandboxed session directory
    Supports reading/editing files from previous executions
    """

    def __init__(self, session_id: str):
        """
        Initialize Python executor for a session

        Args:
            session_id: Session ID for workspace isolation
        """
        self.session_id = session_id
        self.workspace = config.PYTHON_WORKSPACE_DIR / session_id
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.timeout = config.PYTHON_EXECUTOR_TIMEOUT
        self.max_output_size = config.PYTHON_EXECUTOR_MAX_OUTPUT_SIZE
        self.allowed_modules = config.PYTHON_ALLOWED_MODULES

    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code in sandboxed environment

        Args:
            code: Python code to execute
            timeout: Execution timeout (optional)
            context: Additional context (for logging/tracking)

        Returns:
            Execution result dictionary
        """
        exec_timeout = timeout or self.timeout
        start_time = time.time()

        # Log Python code execution
        print(f"\n[PYTHON] Executing code in workspace: {self.workspace}")
        code_preview = code[:200] + "..." if len(code) > 200 else code
        print(f"[PYTHON] Code:\n{code_preview}")
        print(f"[PYTHON] Timeout: {exec_timeout}s")

        # Create execution script with restricted imports
        script_path = self.workspace / f"exec_{int(time.time() * 1000)}.py"

        # Wrap code with safety checks
        wrapped_code = self._wrap_code(code)

        # Write script
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(wrapped_code)

        try:
            # Execute in subprocess with timeout
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=exec_timeout,
                env=self._get_safe_env()
            )

            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode

            # Limit output size
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n... (output truncated)"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n... (output truncated)"

            execution_time = time.time() - start_time

            # Get list of files created/modified
            files = self._get_workspace_files()

            success = returncode == 0

            # Log execution result
            print(f"\n[PYTHON] Execution completed in {execution_time:.2f}s")
            print(f"[PYTHON] Return code: {returncode}")
            if stdout:
                stdout_preview = stdout[:300] + "..." if len(stdout) > 300 else stdout
                print(f"[PYTHON] STDOUT:\n{stdout_preview}")
            if stderr:
                stderr_preview = stderr[:300] + "..." if len(stderr) > 300 else stderr
                print(f"[PYTHON] STDERR:\n{stderr_preview}")
            if files:
                print(f"[PYTHON] Files created: {list(files.keys())}")

            return {
                "success": success,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": returncode,
                "execution_time": execution_time,
                "files": files,
                "workspace": str(self.workspace),
                "error": None if success else stderr
            }

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"\n[PYTHON] ERROR: Execution timeout after {exec_timeout}s")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timeout after {exec_timeout} seconds",
                "returncode": -1,
                "execution_time": execution_time,
                "files": self._get_workspace_files(),
                "workspace": str(self.workspace),
                "error": "Timeout"
            }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"\n[PYTHON] ERROR: {str(e)}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "execution_time": execution_time,
                "files": self._get_workspace_files(),
                "workspace": str(self.workspace),
                "error": str(e)
            }

    def _wrap_code(self, code: str) -> str:
        """
        Wrap code with safety restrictions

        Args:
            code: Original code

        Returns:
            Wrapped code with import restrictions
        """
        # Add import hook to restrict modules
        wrapper = f'''
import sys
import builtins

# Allowed modules
ALLOWED_MODULES = {self.allowed_modules}

# Store original import
_original_import = builtins.__import__

def restricted_import(name, *args, **kwargs):
    """Restrict imports to allowed modules"""
    base_module = name.split('.')[0]
    if base_module not in ALLOWED_MODULES:
        raise ImportError(f"Module '{{name}}' is not allowed in sandboxed execution")
    return _original_import(name, *args, **kwargs)

# Replace built-in import
builtins.__import__ = restricted_import

# User code starts here
try:
{self._indent_code(code, 4)}
except Exception as e:
    import traceback
    print("EXECUTION ERROR:", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
'''
        return wrapper

    def _indent_code(self, code: str, spaces: int) -> str:
        """
        Indent code by specified spaces

        Args:
            code: Code to indent
            spaces: Number of spaces

        Returns:
            Indented code
        """
        indent = ' ' * spaces
        return '\n'.join(indent + line for line in code.split('\n'))

    def _get_safe_env(self) -> Dict[str, str]:
        """
        Get safe environment variables

        Returns:
            Environment dict
        """
        # Start with minimal environment
        safe_env = {
            'PYTHONPATH': '',
            'PATH': os.environ.get('PATH', ''),
            'SYSTEMROOT': os.environ.get('SYSTEMROOT', ''),  # Required on Windows
            'TEMP': str(self.workspace),
            'TMP': str(self.workspace),
        }
        return safe_env

    def _get_workspace_files(self) -> Dict[str, Any]:
        """
        Get list of files in workspace

        Returns:
            Dictionary of files with metadata
        """
        files = {}

        for file_path in self.workspace.iterdir():
            if file_path.is_file():
                rel_path = file_path.name
                files[rel_path] = {
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                    "path": str(file_path)
                }

        return files

    def read_file(self, filename: str) -> Optional[str]:
        """
        Read a file from workspace

        Args:
            filename: File name

        Returns:
            File contents or None
        """
        file_path = self.workspace / filename

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

    def list_files(self) -> List[str]:
        """
        List all files in workspace

        Returns:
            List of file names
        """
        return [f.name for f in self.workspace.iterdir() if f.is_file()]

    def clear_workspace(self):
        """Clear all files in workspace"""
        import shutil
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
