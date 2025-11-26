"""Core code executor for running Python code in isolated environments."""

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from backend.utils.logging_utils import get_logger
from .sandbox import SandboxConfig
from .import_validator import ImportValidator
from .repl_manager import REPLManager
from . import utils

logger = get_logger(__name__)


class CodeExecutor:
    """
    Executes Python code in isolated subprocess with security restrictions.
    Supports both traditional subprocess and persistent REPL modes.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 512,
        execution_base_dir: str = "./data/scratch",
        use_persistent_repl: bool = True
    ):
        """
        Initialize code executor.

        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            execution_base_dir: Base directory for temporary execution folders
            use_persistent_repl: Use persistent REPL for faster retries (default: True)
        """
        # Initialize sandbox configuration
        self.sandbox = SandboxConfig(
            timeout=timeout,
            max_memory_mb=max_memory_mb,
            execution_base_dir=execution_base_dir,
            use_persistent_repl=use_persistent_repl
        )

        # Initialize components
        self.import_validator = ImportValidator()
        self.repl_manager = REPLManager(timeout=timeout) if use_persistent_repl else None

        # File caching (session-based)
        self._file_cache: Dict[str, Dict[str, str]] = {}

        # Legacy properties for backward compatibility
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.execution_base_dir = self.sandbox.execution_base_dir
        self.use_persistent_repl = use_persistent_repl

        logger.info(
            f"[CodeExecutor] Initialized with timeout={timeout}s, "
            f"max_memory={max_memory_mb}MB, repl_mode={use_persistent_repl}"
        )

    def validate_imports(self, code: str) -> tuple:
        """
        Validate that code only imports safe packages.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        return self.import_validator.validate(code)

    def validate_file_type(self, file_path: str) -> bool:
        """
        Check if file type is supported.

        Args:
            file_path: Path to file

        Returns:
            True if file type is supported
        """
        return self.sandbox.is_file_type_supported(file_path)

    def execute(
        self,
        code: str,
        input_files: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
        stage_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code in isolated subprocess or persistent REPL.

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
        execution_dir = self.sandbox.get_execution_dir(execution_id)

        try:
            logger.info(f"[CodeExecutor] Using execution directory: {execution_dir}")

            # Snapshot files BEFORE execution
            files_before = self._get_user_files(execution_dir)

            # Copy input files to execution directory (with caching)
            if input_files:
                utils.prepare_input_files(
                    input_files,
                    execution_dir,
                    self._file_cache,
                    session_id
                )

            # Save code to files
            utils.save_code_to_file(code, execution_dir, stage_name)

            # Choose execution mode based on configuration and session
            if self.use_persistent_repl and session_id:
                result = self._execute_with_repl(code, execution_dir, session_id)
            else:
                result = self._execute_with_subprocess(code, execution_dir, session_id)

            # Snapshot files AFTER execution and find created files
            files_after = self._get_user_files(execution_dir)
            created_files = list(files_after - files_before)
            result["created_files"] = created_files

            if created_files:
                logger.info(f"[CodeExecutor] Code created {len(created_files)} new file(s): {', '.join(created_files)}")

            return result

        except Exception as e:
            logger.error(f"[CodeExecutor] Execution setup failed: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": 0,
                "return_code": -1
            }

    def _execute_with_repl(
        self,
        code: str,
        execution_dir: Path,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Execute code using persistent REPL.

        Args:
            code: Python code to execute
            execution_dir: Execution directory
            session_id: Session ID

        Returns:
            Execution result dict
        """
        try:
            # Get or create REPL for this session
            logger.info(f"[CodeExecutor] [FAST] Executing in persistent REPL")
            repl = self.repl_manager.get_or_create(session_id, execution_dir)

            # Execute in REPL
            result = repl.execute(code)

            # Log execution result
            utils.log_execution_result(result)

            # Enhanced error detection
            result = utils.enhance_error_detection(result)

            return result

        except Exception as e:
            logger.error(f"[CodeExecutor] REPL execution failed: {e}")
            # Fallback to subprocess
            logger.info("[CodeExecutor] Falling back to subprocess mode")
            return self._execute_with_subprocess(code, execution_dir, session_id)

    def _execute_with_subprocess(
        self,
        code: str,
        execution_dir: Path,
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Execute code using traditional subprocess.

        Args:
            code: Python code to execute
            execution_dir: Execution directory
            session_id: Optional session ID

        Returns:
            Execution result dict
        """
        script_path = execution_dir / "script.py"

        try:
            # Get execution environment
            env = utils.get_execution_env()

            logger.info("[CodeExecutor] [SLOW] Executing in subprocess mode")
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                timeout=self.timeout,
                cwd=str(execution_dir),
                encoding='utf-8',
                errors='backslashreplace',
                env=env
            )
            execution_time = time.time() - start_time

            result_dict = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "namespace": {}  # Subprocess mode doesn't capture namespace
            }

            # Log execution result
            utils.log_execution_result(result_dict)

            # Enhanced error detection
            result_dict = utils.enhance_error_detection(result_dict)

            return result_dict

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
            logger.error(f"[CodeExecutor] Subprocess execution failed: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": 0,
                "return_code": -1
            }

        finally:
            # Cleanup execution directory (only if temporary)
            utils.cleanup_execution_dir(execution_dir, session_id)

    def _get_user_files(self, execution_dir: Path) -> set:
        """
        Get set of user-created files in execution directory.
        Excludes system files, scripts, and infrastructure files.

        Args:
            execution_dir: Directory to scan

        Returns:
            Set of relative file paths (basenames only)
        """
        if not execution_dir.exists():
            return set()

        exclude_patterns = {
            'script.py',          # Main execution script
            'notepad.json',       # Session notepad
            '__pycache__',        # Python cache
            'variables',          # Variable storage directory
            'prompts',            # Saved prompts directory
        }

        user_files = set()
        for item in execution_dir.iterdir():
            # Skip directories (except we track them by name)
            if item.is_dir():
                if item.name not in exclude_patterns:
                    # Note directory existence but don't recurse
                    continue
            else:
                # Skip excluded files and script_*.py files
                if item.name in exclude_patterns or item.name.startswith('script_'):
                    continue

                user_files.add(item.name)

        return user_files

    def cleanup_session(self, session_id: str):
        """
        Cleanup REPL and cached data for a session.

        Args:
            session_id: Session ID to cleanup
        """
        # Stop REPL
        if self.repl_manager:
            self.repl_manager.cleanup_session(session_id)

        # Clear file cache
        if session_id in self._file_cache:
            del self._file_cache[session_id]
            logger.debug(f"[CodeExecutor] Cleared file cache for session {session_id[:8]}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get executor statistics.

        Returns:
            Dictionary with executor statistics
        """
        stats = {
            "config": self.sandbox.to_dict(),
            "file_cache_sessions": len(self._file_cache)
        }

        if self.repl_manager:
            stats["repl_stats"] = self.repl_manager.get_stats()

        return stats


# Backward compatibility alias
PythonExecutor = CodeExecutor
