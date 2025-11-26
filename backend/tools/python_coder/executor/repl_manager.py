"""Persistent REPL manager for fast code execution."""

import os
import subprocess
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


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

            # Wait for ready signal (longer timeout for library pre-loading)
            ready = self._wait_for_ready(timeout=60)

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
import json

# Force UTF-8 encoding for stdout/stderr (fixes Windows cp949 codec errors)
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'backslashreplace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'backslashreplace')

# Pre-import common heavy libraries to cache them in memory
# This reduces first-time import overhead from 20-30s to <1s
import time as _warmup_timer
_warmup_start = _warmup_timer.time()
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # Configure matplotlib for non-interactive backend (faster)
    matplotlib.use('Agg')
    _warmup_time = _warmup_timer.time() - _warmup_start
    print(f"<<<WARMUP_COMPLETE>>> Preloaded libraries in {_warmup_time:.2f}s", flush=True)
except ImportError as e:
    print(f"<<<WARMUP_WARNING>>> Some libraries not available: {e}", flush=True)

# Signal ready
print("<<<REPL_READY>>>", flush=True)

# Create persistent namespace with pre-imported libraries
# This allows code to reuse imports without re-loading
_persistent_namespace = {
    "__name__": "__main__",
    "pd": pd if 'pd' in dir() else None,
    "np": np if 'np' in dir() else None,
    "plt": plt if 'plt' in dir() else None,
}
# Remove None values
_persistent_namespace = {k: v for k, v in _persistent_namespace.items() if v is not None}

def _extract_namespace_info(namespace):
    """Extract variable information from namespace."""
    info = {}
    for name, value in namespace.items():
        # Skip private and built-in variables
        if name.startswith('_'):
            continue

        # Get type information
        var_type = type(value).__name__
        var_module = type(value).__module__

        # Skip modules and callables
        if var_module == 'builtins' and var_type == 'module':
            continue
        if callable(value) and not isinstance(value, type):
            continue

        # Build type info
        full_type = f"{var_module}.{var_type}" if var_module != "builtins" else var_type

        # Get additional metadata based on type
        meta = {"type": full_type}

        try:
            # Check for pandas DataFrame
            if var_type == 'DataFrame' and 'pandas' in var_module:
                meta["shape"] = list(value.shape)
                meta["columns"] = list(value.columns)
            # Check for numpy array
            elif var_type == 'ndarray' and 'numpy' in var_module:
                meta["shape"] = list(value.shape)
                meta["dtype"] = str(value.dtype)
            # Check for dict
            elif var_type == 'dict':
                meta["keys"] = list(value.keys())[:10]
            # Check for list
            elif var_type == 'list':
                meta["length"] = len(value)
            # Simple types
            elif var_type in ['int', 'float', 'str', 'bool']:
                meta["value"] = str(value)[:100]
        except Exception:
            pass

        info[name] = meta

    return info

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
            # Execute code in persistent namespace (preserves imports and variables)
            # Make a copy to avoid polluting the global persistent namespace
            namespace = _persistent_namespace.copy()
            exec(code, namespace)

            # Get captured output
            stdout_val = sys.stdout.getvalue()
            stderr_val = sys.stderr.getvalue()

            # Extract namespace information
            namespace_info = _extract_namespace_info(namespace)

            # Signal success
            print("<<<EXEC_SUCCESS>>>", file=old_stdout, flush=True)
            print(stdout_val, file=old_stdout, end="", flush=True)

            # Send namespace info
            if namespace_info:
                print("<<<NAMESPACE_START>>>", file=old_stdout, flush=True)
                print(json.dumps(namespace_info), file=old_stdout, flush=True)
                print("<<<NAMESPACE_END>>>", file=old_stdout, flush=True)

            print("<<<EXEC_END>>>", file=old_stdout, flush=True)

            if stderr_val:
                print(stderr_val, file=old_stderr, end="", flush=True)

        except Exception as e:
            # Capture namespace BEFORE signaling error (for debugging)
            error_namespace = _extract_namespace_info(namespace)
            
            # Signal error
            print("<<<EXEC_ERROR>>>", file=old_stdout, flush=True)
            traceback.print_exc(file=old_stdout)
            
            # Send namespace info even on error (helps LLM understand what went wrong)
            if error_namespace:
                print("<<<ERROR_NAMESPACE_START>>>", file=old_stdout, flush=True)
                print(json.dumps(error_namespace), file=old_stdout, flush=True)
                print("<<<ERROR_NAMESPACE_END>>>", file=old_stdout, flush=True)
            
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
        warmup_logged = False
        while time.time() - start_time < timeout:
            try:
                line = self.stdout_queue.get(timeout=0.1)
                if "<<<WARMUP_COMPLETE>>>" in line and not warmup_logged:
                    logger.success(f"[PersistentREPL] {line.strip()}")
                    warmup_logged = True
                elif "<<<WARMUP_WARNING>>>" in line:
                    logger.warning(f"[PersistentREPL] {line.strip()}")
                elif "<<<REPL_READY>>>" in line:
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
        import json

        start_time = time.time()
        output_lines = []
        error_lines = []
        success = None
        namespace_info = {}
        in_namespace = False
        in_error_namespace = False
        namespace_lines = []

        while time.time() - start_time < timeout:
            try:
                # Check stdout
                try:
                    line = self.stdout_queue.get(timeout=0.1)

                    if "<<<EXEC_SUCCESS>>>" in line:
                        success = True
                    elif "<<<EXEC_ERROR>>>" in line:
                        success = False
                    elif "<<<NAMESPACE_START>>>" in line:
                        in_namespace = True
                        in_error_namespace = False
                    elif "<<<NAMESPACE_END>>>" in line:
                        in_namespace = False
                        # Parse namespace JSON (success case)
                        try:
                            namespace_json = "".join(namespace_lines)
                            namespace_info = json.loads(namespace_json)
                        except Exception as e:
                            logger.warning(f"[PersistentREPL] Failed to parse namespace: {e}")
                        namespace_lines = []
                    elif "<<<ERROR_NAMESPACE_START>>>" in line:
                        # Namespace captured on error (for debugging)
                        in_error_namespace = True
                        in_namespace = False
                    elif "<<<ERROR_NAMESPACE_END>>>" in line:
                        in_error_namespace = False
                        # Parse error namespace JSON
                        try:
                            namespace_json = "".join(namespace_lines)
                            namespace_info = json.loads(namespace_json)
                            logger.debug(f"[PersistentREPL] Captured error namespace: {len(namespace_info)} vars")
                        except Exception as e:
                            logger.warning(f"[PersistentREPL] Failed to parse error namespace: {e}")
                        namespace_lines = []
                    elif "<<<EXEC_END>>>" in line:
                        # Execution finished
                        return {
                            "success": success if success is not None else False,
                            "output": "".join(output_lines),
                            "error": "".join(error_lines) if not success else None,
                            "namespace": namespace_info
                        }
                    elif "<<<REPL_ERROR>>>" in line:
                        # REPL itself crashed
                        self._is_healthy = False
                        return {
                            "success": False,
                            "output": "",
                            "error": f"REPL crashed: {line}",
                            "namespace": {}
                        }
                    else:
                        # Collect lines
                        if in_namespace or in_error_namespace:
                            namespace_lines.append(line)
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
            "error": f"Execution timeout after {timeout} seconds",
            "namespace": {}
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


class REPLManager:
    """
    Manages multiple persistent REPL instances (one per session).
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize REPL manager.

        Args:
            timeout: Default timeout for REPL execution
        """
        self.timeout = timeout
        self._repls: Dict[str, PersistentREPL] = {}
        logger.debug(f"[REPLManager] Initialized with timeout={timeout}s")

    def get_or_create(self, session_id: str, execution_dir: Path) -> PersistentREPL:
        """
        Get existing REPL or create a new one for a session.

        Args:
            session_id: Session identifier
            execution_dir: Execution directory for REPL

        Returns:
            PersistentREPL instance
        """
        if session_id not in self._repls:
            logger.info(f"[REPLManager] Creating REPL for session {session_id[:8]}")
            repl = PersistentREPL(execution_dir, timeout=self.timeout)
            if repl.start():
                self._repls[session_id] = repl
            else:
                raise RuntimeError(f"Failed to start REPL for session {session_id}")

        return self._repls[session_id]

    def cleanup_session(self, session_id: str):
        """
        Stop and cleanup REPL for a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self._repls:
            self._repls[session_id].stop()
            del self._repls[session_id]
            logger.info(f"[REPLManager] Cleaned up REPL for session {session_id[:8]}")

    def cleanup_all(self):
        """Stop and cleanup all REPLs."""
        for session_id in list(self._repls.keys()):
            self.cleanup_session(session_id)
        logger.info("[REPLManager] Cleaned up all REPLs")

    def get_active_sessions(self) -> List[str]:
        """Get list of sessions with active REPLs."""
        return [sid for sid, repl in self._repls.items() if repl.is_alive()]

    def get_stats(self) -> Dict[str, Any]:
        """Get REPL manager statistics."""
        return {
            "total_repls": len(self._repls),
            "active_repls": len(self.get_active_sessions()),
            "timeout": self.timeout
        }
