"""
Nanocoder Python Code Executor
Executes natural language instructions by using nanocoder CLI tool
"""
import os
import sys
import time
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import config
from tools.python_coder.base import BasePythonExecutor
from tools.python_coder.nanocoder_config import ensure_nanocoder_config


def log_to_prompts_file(message: str):
    """Write message to prompts.log"""
    try:
        with open(config.PROMPTS_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"[WARNING] Failed to write to prompts.log: {e}")


class NanocoderExecutor(BasePythonExecutor):
    """
    Nanocoder-based executor for natural language code generation and execution

    Receives natural language instructions, uses nanocoder to generate and execute code,
    and returns standardized results.
    """

    def __init__(self, session_id: str):
        """
        Initialize Nanocoder executor

        Args:
            session_id: Session ID for workspace isolation
        """
        super().__init__(session_id)
        self.workspace = config.PYTHON_WORKSPACE_DIR / session_id
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.timeout = config.NANOCODER_TIMEOUT
        self.max_output_size = config.PYTHON_EXECUTOR_MAX_OUTPUT_SIZE
        self.nanocoder_path = config.NANOCODER_PATH

        # Ensure nanocoder config exists
        ensure_nanocoder_config()

        # Check if nanocoder is available
        self._check_nanocoder_available()

    def _check_nanocoder_available(self):
        """
        Check if nanocoder is available in the system

        Raises:
            RuntimeError: If nanocoder is not available
        """
        try:
            # Try to run nanocoder --version
            result = subprocess.run(
                [self.nanocoder_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=True,  # Use shell for Windows compatibility
                encoding='utf-8',  # Force UTF-8 encoding
                errors='ignore'  # Ignore decode errors
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Nanocoder check failed. Make sure nanocoder is installed.\n"
                    f"Install with: npm install -g @nanocollective/nanocoder\n"
                    f"Current path: {self.nanocoder_path}"
                )

            # Skip printing version to avoid Unicode issues on Windows
            print(f"[NANOCODER] Nanocoder is available")

        except FileNotFoundError:
            raise RuntimeError(
                f"Nanocoder not found at: {self.nanocoder_path}\n"
                f"Install with: npm install -g @nanocollective/nanocoder\n"
                f"Verify installation with: nanocoder --version"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Nanocoder check timed out. Path: {self.nanocoder_path}"
            )

    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute natural language instruction using Nanocoder

        Args:
            code: Natural language instruction (NOT Python code)
            timeout: Execution timeout (optional)
            context: Additional context (unused)

        Returns:
            Standardized execution result dictionary
        """
        exec_timeout = timeout or self.timeout
        start_time = time.time()

        # Log execution start
        log_to_prompts_file("\n\n")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"TOOL EXECUTION: python_coder (Nanocoder)")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_to_prompts_file(f"Session ID: {self.session_id}")
        log_to_prompts_file(f"Workspace: {self.workspace}")
        log_to_prompts_file(f"Instruction Length: {len(code)} chars")
        log_to_prompts_file(f"Timeout: {exec_timeout}s")
        log_to_prompts_file(f"")
        log_to_prompts_file(f"NATURAL LANGUAGE INSTRUCTION:")
        for line in code.split('\n'):
            log_to_prompts_file(f"  {line}")

        print("\n" + "=" * 80)
        print("[NANOCODER] execute() called")
        print("=" * 80)
        print(f"Session ID: {self.session_id}")
        print(f"Workspace: {self.workspace}")
        print(f"Instruction: {code[:100]}...")
        print(f"Timeout: {exec_timeout}s")

        try:
            # Change to workspace directory
            original_cwd = os.getcwd()
            os.chdir(self.workspace)
            print(f"[NANOCODER] Changed to workspace: {self.workspace}")

            try:
                # Build nanocoder command
                # Use "run" subcommand for non-interactive execution
                nanocoder_cmd = f'{self.nanocoder_path} run "{code}"'

                # Configure environment for nanocoder
                env = os.environ.copy()

                # Set nanocoder config directory to absolute path from project root
                project_root = Path(original_cwd)
                nanocoder_config_path = project_root / config.NANOCODER_CONFIG_DIR
                env["NANOCODER_CONFIG_DIR"] = str(nanocoder_config_path.absolute())

                print(f"\n[NANOCODER] Executing command...")
                print(f"[NANOCODER] Command: {nanocoder_cmd}")
                print(f"[NANOCODER] Config dir: {env['NANOCODER_CONFIG_DIR']}")

                log_to_prompts_file(f"")
                log_to_prompts_file(f"EXECUTING...")
                log_to_prompts_file(f"Command: {nanocoder_cmd}")

                # Execute nanocoder
                result = subprocess.run(
                    nanocoder_cmd,
                    capture_output=True,
                    text=True,
                    timeout=exec_timeout,
                    env=env,
                    shell=True,  # Use shell to handle command string
                    encoding='utf-8',  # Force UTF-8 encoding
                    errors='ignore'  # Ignore decode errors
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
                files = self._get_workspace_files()
                success = returncode == 0

                # Log result
                log_to_prompts_file(f"")
                log_to_prompts_file(f"OUTPUT:")
                log_to_prompts_file(f"  Status: {'SUCCESS' if success else 'FAILED'}")
                log_to_prompts_file(f"  Return Code: {returncode}")
                log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")

                if stdout:
                    log_to_prompts_file(f"")
                    log_to_prompts_file(f"STDOUT:")
                    for line in stdout.split('\n'):
                        log_to_prompts_file(f"  {line}")

                if stderr:
                    log_to_prompts_file(f"")
                    log_to_prompts_file(f"STDERR:")
                    for line in stderr.split('\n'):
                        log_to_prompts_file(f"  {line}")

                if files:
                    log_to_prompts_file(f"")
                    log_to_prompts_file(f"FILES:")
                    for filename, meta in files.items():
                        log_to_prompts_file(f"  {filename} ({meta['size']} bytes)")

                log_to_prompts_file(f"")
                log_to_prompts_file("=" * 80)

                # Console output
                print(f"\n[NANOCODER] Execution completed in {execution_time:.2f}s")
                print(f"[NANOCODER] Success: {success}")
                if stdout:
                    preview = stdout[:300]
                    if len(stdout) > 300:
                        preview += "..."
                    print(f"[NANOCODER] STDOUT:\n{preview}")
                if stderr:
                    preview = stderr[:300]
                    if len(stderr) > 300:
                        preview += "..."
                    print(f"[NANOCODER] STDERR:\n{preview}")
                if files:
                    print(f"[NANOCODER] Files: {list(files.keys())}")

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

            finally:
                # Restore original directory
                os.chdir(original_cwd)
                print(f"[NANOCODER] Restored original directory")

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            error_msg = f"Nanocoder execution timeout after {exec_timeout} seconds"

            print(f"\n[NANOCODER] TIMEOUT: {error_msg}")

            log_to_prompts_file(f"")
            log_to_prompts_file(f"ERROR: TimeoutExpired")
            log_to_prompts_file(f"  {error_msg}")
            log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")
            log_to_prompts_file(f"")
            log_to_prompts_file("=" * 80)

            return {
                "success": False,
                "stdout": "",
                "stderr": error_msg,
                "returncode": -1,
                "execution_time": execution_time,
                "files": self._get_workspace_files(),
                "workspace": str(self.workspace),
                "error": error_msg
            }

        except FileNotFoundError as e:
            execution_time = time.time() - start_time
            error_msg = f"Nanocoder not found: {str(e)}"

            print(f"\n[NANOCODER] ERROR: {error_msg}")

            log_to_prompts_file(f"")
            log_to_prompts_file(f"ERROR: FileNotFoundError")
            log_to_prompts_file(f"  {error_msg}")
            log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")
            log_to_prompts_file(f"")
            log_to_prompts_file("=" * 80)

            raise RuntimeError(
                f"Nanocoder not found. Install with: npm install -g @nanocollective/nanocoder\n"
                f"Verify with: nanocoder --version"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            print(f"\n[NANOCODER] ERROR: {error_msg}")

            # Print full traceback for debugging
            import traceback
            print(f"[NANOCODER] Full traceback:")
            traceback.print_exc()

            log_to_prompts_file(f"")
            log_to_prompts_file(f"ERROR: {type(e).__name__}")
            log_to_prompts_file(f"  {error_msg}")
            log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")
            log_to_prompts_file(f"")
            log_to_prompts_file("=" * 80)

            raise

    def _get_workspace_files(self) -> Dict[str, Any]:
        """
        Get list of files in workspace

        Returns:
            Dictionary of files with metadata
        """
        files = {}

        try:
            for file_path in self.workspace.iterdir():
                if file_path.is_file():
                    files[file_path.name] = {
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                        "path": str(file_path)
                    }
        except Exception as e:
            print(f"[NANOCODER] Warning: Failed to list files: {e}")

        return files

    def read_file(self, filename: str) -> Optional[str]:
        """
        Read a file from workspace

        Args:
            filename: File name

        Returns:
            File contents or None if not found
        """
        file_path = self.workspace / filename

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[NANOCODER] Warning: Failed to read '{filename}': {e}")
            return None

    def list_files(self) -> List[str]:
        """
        List all files in workspace

        Returns:
            List of file names
        """
        try:
            return [f.name for f in self.workspace.iterdir() if f.is_file()]
        except Exception as e:
            print(f"[NANOCODER] Warning: Failed to list files: {e}")
            return []

    def clear_workspace(self):
        """Clear all files in workspace"""
        import shutil

        try:
            if self.workspace.exists():
                shutil.rmtree(self.workspace)
            self.workspace.mkdir(parents=True, exist_ok=True)
            print(f"[NANOCODER] Workspace cleared: {self.workspace}")
        except Exception as e:
            print(f"[NANOCODER] Warning: Failed to clear workspace: {e}")
