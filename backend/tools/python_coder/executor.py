"""Code executor for running Python code in isolated subprocess."""

import ast
import os
import shutil
import subprocess
import sys
import time
import uuid
import threading
import queue
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


class PersistentREPL:
    """
    Persistent Python REPL for fast repeated code execution.
    Reuses a single Python subprocess to avoid spawning overhead.
    """

    def __init__(self, execution_dir: Path, timeout: int = 30):
        """
        Initialize persistent REPL.

        Args:
            execution_dir: Working directory for code execution
            timeout: Maximum execution time per code block
        """
        self.execution_dir = execution_dir
        self.timeout = timeout
        self.process: Optional[subprocess.Popen] = None
        self.stdout_queue: queue.Queue = queue.Queue()
        self.stderr_queue: queue.Queue = queue.Queue()
        self._reader_threads: List[threading.Thread] = []
        self._is_healthy = False

        logger.debug(f"[PersistentREPL] Initialized for {execution_dir}")

    def start(self) -> bool:
        """
        Start the persistent Python REPL process.

        Returns:
            True if started successfully, False otherwise
        """
        try:
            # Create environment with UTF-8 encoding forced (fixes Windows cp949 codec errors)
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            # Start Python in unbuffered mode (-u) with working directory
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-c", self._get_repl_bootstrap()],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.execution_dir),
                encoding='utf-8',
                errors='backslashreplace',
                bufsize=0,  # Unbuffered
                env=env  # Pass environment with UTF-8 encoding
            )

            # Start reader threads for stdout/stderr
            self._start_reader_threads()

            # Wait for ready signal
            ready = self._wait_for_ready(timeout=5)

            if ready:
                self._is_healthy = True
                logger.success("[PersistentREPL] Started successfully")
                return True
            else:
                logger.error("[PersistentREPL] Failed to receive ready signal")
                self.stop()
                return False

        except Exception as e:
            logger.error(f"[PersistentREPL] Failed to start: {e}")
            self.stop()
            return False

    def _get_repl_bootstrap(self) -> str:
        """Get bootstrap code that runs in the REPL."""
        return '''
import sys
import traceback
import io

# Force UTF-8 encoding for stdout/stderr (fixes Windows cp949 codec errors)
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'backslashreplace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'backslashreplace')

# Signal ready
print("<<<REPL_READY>>>", flush=True)

# Main REPL loop
while True:
    try:
        # Read delimiter
        delimiter = input()
        if delimiter != "<<<CODE_START>>>":
            continue

        # Read code lines until end delimiter
        code_lines = []
        while True:
            line = input()
            if line == "<<<CODE_END>>>":
                break
            code_lines.append(line)

        code = "\\n".join(code_lines)

        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            # Execute code in clean namespace
            namespace = {"__name__": "__main__"}
            exec(code, namespace)

            # Get captured output
            stdout_val = sys.stdout.getvalue()
            stderr_val = sys.stderr.getvalue()

            # Signal success
            print("<<<EXEC_SUCCESS>>>", file=old_stdout, flush=True)
            print(stdout_val, file=old_stdout, end="", flush=True)
            print("<<<EXEC_END>>>", file=old_stdout, flush=True)

            if stderr_val:
                print(stderr_val, file=old_stderr, end="", flush=True)

        except Exception as e:
            # Signal error
            print("<<<EXEC_ERROR>>>", file=old_stdout, flush=True)
            traceback.print_exc(file=old_stdout)
            print("<<<EXEC_END>>>", file=old_stdout, flush=True)

        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    except EOFError:
        break
    except Exception as e:
        print(f"<<<REPL_ERROR>>> {e}", file=sys.stdout, flush=True)
        break
'''

    def _start_reader_threads(self):
        """Start background threads to read stdout/stderr."""
        def read_stream(stream, output_queue):
            try:
                for line in iter(stream.readline, ''):
                    if line:
                        output_queue.put(line)
            except Exception as e:
                logger.warning(f"[PersistentREPL] Reader thread error: {e}")

        stdout_thread = threading.Thread(
            target=read_stream,
            args=(self.process.stdout, self.stdout_queue),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=read_stream,
            args=(self.process.stderr, self.stderr_queue),
            daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()
        self._reader_threads = [stdout_thread, stderr_thread]

    def _wait_for_ready(self, timeout: float) -> bool:
        """Wait for REPL ready signal."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                line = self.stdout_queue.get(timeout=0.1)
                if "<<<REPL_READY>>>" in line:
                    return True
            except queue.Empty:
                continue
        return False

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute code in the persistent REPL.

        Args:
            code: Python code to execute

        Returns:
            Dict with keys: success, output, error, execution_time, return_code
        """
        if not self._is_healthy or not self.process or self.process.poll() is not None:
            logger.warning("[PersistentREPL] REPL not healthy, attempting restart")
            if not self.start():
                return {
                    "success": False,
                    "output": "",
                    "error": "REPL failed to start",
                    "execution_time": 0,
                    "return_code": -1
                }

        try:
            # Clear queues
            self._clear_queues()

            # Send code with delimiters
            start_time = time.time()
            self.process.stdin.write("<<<CODE_START>>>\n")
            self.process.stdin.write(code + "\n")
            self.process.stdin.write("<<<CODE_END>>>\n")
            self.process.stdin.flush()

            # Wait for execution result
            result = self._wait_for_result(timeout=self.timeout)
            execution_time = time.time() - start_time

            return {
                "success": result["success"],
                "output": result["output"],
                "error": result["error"],
                "execution_time": execution_time,
                "return_code": 0 if result["success"] else 1
            }

        except Exception as e:
            logger.error(f"[PersistentREPL] Execution failed: {e}")
            self._is_healthy = False
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": 0,
                "return_code": -1
            }

    def _wait_for_result(self, timeout: float) -> Dict[str, Any]:
        """Wait for execution result from REPL."""
        start_time = time.time()
        output_lines = []
        error_lines = []
        success = None

        while time.time() - start_time < timeout:
            try:
                # Check stdout
                try:
                    line = self.stdout_queue.get(timeout=0.1)

                    if "<<<EXEC_SUCCESS>>>" in line:
                        success = True
                    elif "<<<EXEC_ERROR>>>" in line:
                        success = False
                    elif "<<<EXEC_END>>>" in line:
                        # Execution finished
                        return {
                            "success": success if success is not None else False,
                            "output": "".join(output_lines),
                            "error": "".join(error_lines) if not success else None
                        }
                    elif "<<<REPL_ERROR>>>" in line:
                        # REPL itself crashed
                        self._is_healthy = False
                        return {
                            "success": False,
                            "output": "",
                            "error": f"REPL crashed: {line}"
                        }
                    else:
                        output_lines.append(line)
                except queue.Empty:
                    pass

                # Check stderr
                try:
                    error_line = self.stderr_queue.get_nowait()
                    error_lines.append(error_line)
                except queue.Empty:
                    pass

            except Exception as e:
                logger.error(f"[PersistentREPL] Error reading result: {e}")
                break

        # Timeout
        self._is_healthy = False
        return {
            "success": False,
            "output": "".join(output_lines),
            "error": f"Execution timeout after {timeout} seconds"
        }

    def _clear_queues(self):
        """Clear stdout/stderr queues."""
        while not self.stdout_queue.empty():
            try:
                self.stdout_queue.get_nowait()
            except queue.Empty:
                break
        while not self.stderr_queue.empty():
            try:
                self.stderr_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        """Stop the REPL process and cleanup."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None
                self._is_healthy = False
        logger.debug("[PersistentREPL] Stopped")

    def is_alive(self) -> bool:
        """Check if REPL is alive and healthy."""
        return (
            self._is_healthy
            and self.process is not None
            and self.process.poll() is None
        )


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
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.execution_base_dir = Path(execution_base_dir).resolve()
        self.execution_base_dir.mkdir(parents=True, exist_ok=True)
        self.use_persistent_repl = use_persistent_repl

        # REPL management (session-based)
        self._repls: Dict[str, PersistentREPL] = {}

        # File caching (session-based)
        self._file_cache: Dict[str, Dict[str, str]] = {}

        logger.info(
            f"[CodeExecutor] Initialized with timeout={timeout}s, "
            f"max_memory={max_memory_mb}MB, repl_mode={use_persistent_repl}"
        )

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
        execution_dir = self.execution_base_dir / execution_id

        try:
            # Create execution directory
            execution_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[CodeExecutor] Using execution directory: {execution_dir}")

            # Copy input files to execution directory (with caching)
            if input_files:
                self._prepare_input_files(input_files, execution_dir, session_id)

            # Write code to script.py (main execution file)
            script_path = execution_dir / "script.py"
            script_path.write_text(code, encoding='utf-8')
            logger.debug(f"[CodeExecutor] Wrote code to {script_path}")

            # ALSO save to stage-specific file if stage_name provided
            if stage_name:
                stage_script_path = execution_dir / f"script_{stage_name}.py"
                stage_script_path.write_text(code, encoding='utf-8')
                logger.info(f"[CodeExecutor] [SAVED] Saved stage code to {stage_script_path.name}")

            # Choose execution mode based on configuration and session
            if self.use_persistent_repl and session_id:
                return self._execute_with_repl(code, execution_dir, session_id)
            else:
                return self._execute_with_subprocess(code, execution_dir, session_id)

        except Exception as e:
            logger.error(f"[CodeExecutor] Execution setup failed: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": 0,
                "return_code": -1
            }

    def _prepare_input_files(
        self,
        input_files: Dict[str, str],
        execution_dir: Path,
        session_id: Optional[str]
    ):
        """
        Copy input files to execution directory with caching.

        Args:
            input_files: Dict mapping original file paths to basenames
            execution_dir: Target directory for files
            session_id: Optional session ID for caching
        """
        # Check if files already cached for this session
        if session_id and session_id in self._file_cache:
            logger.info(f"[CodeExecutor] 📁 Using cached input files ({len(input_files)} files)")
            return

        # Copy files
        for original_path, basename in input_files.items():
            target_path = execution_dir / basename
            if not target_path.exists():
                shutil.copy2(original_path, target_path)
                logger.debug(f"[CodeExecutor] Copied {original_path} -> {target_path}")

        # Cache file list for this session
        if session_id:
            self._file_cache[session_id] = input_files
            logger.info(f"[CodeExecutor] 📁 Copied and cached {len(input_files)} input files")
        else:
            logger.debug(f"[CodeExecutor] 📁 Copied {len(input_files)} input files")

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
            if session_id not in self._repls:
                logger.info(f"[CodeExecutor] [STARTING] Starting persistent REPL for session {session_id[:8]}")
                repl = PersistentREPL(execution_dir, timeout=self.timeout)
                if repl.start():
                    self._repls[session_id] = repl
                else:
                    logger.warning("[CodeExecutor] REPL start failed, falling back to subprocess")
                    return self._execute_with_subprocess(code, execution_dir, session_id)

            repl = self._repls[session_id]

            # Execute in REPL
            logger.info("[CodeExecutor] [FAST] Executing in persistent REPL (fast mode)")
            result = repl.execute(code)

            # Log execution result (same as subprocess)
            self._log_execution_result(result)

            # Enhanced success detection
            result = self._enhance_error_detection(result)

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
            # Create environment with UTF-8 encoding forced (fixes Windows cp949 codec errors)
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            logger.info("[CodeExecutor] [SLOW] Executing in subprocess mode")
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                timeout=self.timeout,
                cwd=str(execution_dir),
                encoding='utf-8',
                errors='backslashreplace',
                env=env  # Pass environment with UTF-8 encoding
            )
            execution_time = time.time() - start_time

            result_dict = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "execution_time": execution_time,
                "return_code": result.returncode
            }

            # Log execution result
            self._log_execution_result(result_dict)

            # Enhanced success detection
            result_dict = self._enhance_error_detection(result_dict)

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

    def _log_execution_result(self, result: Dict[str, Any]):
        """Log execution result with appropriate level."""
        execution_time = result.get("execution_time", 0)

        if result["success"]:
            logger.success("Code execution succeeded", f"{execution_time:.2f}s")
        else:
            logger.failure("Code execution failed", f"Return code: {result.get('return_code', -1)}")

        # Log stdout
        output = result.get("output", "")
        if output:
            logger.multiline(output, title="STDOUT", max_lines=50)
        else:
            logger.info("STDOUT: (empty)")

        # Log stderr
        error = result.get("error", "")
        if error:
            if not result["success"]:
                logger.multiline(error, title="STDERR - ERROR", max_lines=30)
            else:
                logger.multiline(error, title="STDERR - WARNING", max_lines=20)

    def _enhance_error_detection(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced error detection: check for error patterns in stdout.

        Args:
            result: Execution result dict

        Returns:
            Updated result dict with enhanced error detection
        """
        if result["return_code"] == 0 and result.get("output"):
            error_patterns = [
                "Error:", "error:", "ERROR:",
                "Failed:", "failed:", "FAILED:",
                "Exception:", "exception:",
                "not found", "Not found", "NOT FOUND",
                "does not contain", "does not exist",
                "No valid", "no valid",
                "Invalid", "invalid"
            ]

            stdout_lower = result["output"].lower()
            for pattern in error_patterns:
                if pattern.lower() in stdout_lower:
                    logger.warning(f"[CodeExecutor] [WARNING] Error pattern detected: '{pattern}'")
                    result["success"] = False
                    result["error"] = result["output"]
                    logger.error(f"[CodeExecutor] Code printed error messages despite return code 0")
                    break

        return result

    def cleanup_session(self, session_id: str):
        """
        Cleanup REPL and cached data for a session.

        Args:
            session_id: Session ID to cleanup
        """
        # Stop REPL
        if session_id in self._repls:
            self._repls[session_id].stop()
            del self._repls[session_id]
            logger.info(f"[CodeExecutor] Stopped REPL for session {session_id[:8]}")

        # Clear file cache
        if session_id in self._file_cache:
            del self._file_cache[session_id]
            logger.debug(f"[CodeExecutor] Cleared file cache for session {session_id[:8]}")

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
