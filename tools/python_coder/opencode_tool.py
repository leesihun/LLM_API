"""
OpenCode Python Code Executor
Executes natural language instructions using OpenCode AI coding agent
"""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from tools.python_coder.base import BasePythonExecutor
from tools.python_coder.opencode_server import get_server_manager


def log_to_prompts_file(message: str) -> None:
    """Write message to prompts.log"""
    try:
        with open(config.PROMPTS_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"[WARNING] Failed to write to prompts.log: {e}")


class OpenCodeExecutor(BasePythonExecutor):
    """
    OpenCode-based executor for natural language code generation and execution

    Uses persistent opencode server for reduced latency.
    Maintains session across tool calls within same LLM API session.
    """

    # Class-level session mapping: {llm_session_id: opencode_session_id}
    _session_map: Dict[str, str] = {}

    def __init__(self, session_id: str):
        """
        Initialize OpenCode executor

        Args:
            session_id: LLM API session ID for workspace isolation
        """
        super().__init__(session_id)

        self.workspace = config.PYTHON_WORKSPACE_DIR / session_id
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.timeout = config.OPENCODE_TIMEOUT
        self.max_output_size = config.PYTHON_EXECUTOR_MAX_OUTPUT_SIZE

        # Get server manager (ensures server is running)
        self._server = get_server_manager()

    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute natural language instruction using OpenCode

        Args:
            code: Natural language instruction (passed as 'code' for API compatibility)
            timeout: Execution timeout in seconds
            context: Additional context (unused)

        Returns:
            Standardized execution result dictionary

        Raises:
            RuntimeError: If opencode server is unavailable
        """
        instruction = code  # Alias for clarity in OpenCode context
        exec_timeout = timeout or self.timeout
        start_time = time.time()

        # Log execution start
        self._log_start(instruction, exec_timeout)

        # Ensure server is running
        self._server.ensure_running()

        # #region agent log
        import json as _json
        _log_data = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "opencode_tool.py:execute:server_check", "message": "Server status after ensure_running", "data": {"server_url": self._server.server_url, "is_running": self._server.is_running()}, "timestamp": int(time.time() * 1000)}
        with open(r"c:\Users\Lee\Desktop\Huni\LLM_API\.cursor\debug.log", "a", encoding="utf-8") as _f: _f.write(_json.dumps(_log_data) + "\n")
        # #endregion

        # Build command
        cmd = self._build_command(instruction)

        print(f"\n[OPENCODE] Executing: {' '.join(cmd[:6])}...")

        try:
            # #region agent log
            import json as _json
            _log_data = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B,C", "location": "opencode_tool.py:execute:pre_run", "message": "About to execute subprocess", "data": {"cmd_count": len(cmd), "cwd": str(self.workspace), "timeout": exec_timeout}, "timestamp": int(time.time() * 1000)}
            with open(r"c:\Users\Lee\Desktop\Huni\LLM_API\.cursor\debug.log", "a", encoding="utf-8") as _f: _f.write(_json.dumps(_log_data) + "\n")
            # #endregion

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=exec_timeout,
                cwd=str(self.workspace),
                encoding='utf-8',
                errors='replace'
            )

            # #region agent log
            _log_data = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A,B,C,D,E", "location": "opencode_tool.py:execute:post_run", "message": "Subprocess completed", "data": {"returncode": result.returncode, "stdout_len": len(result.stdout), "stderr_len": len(result.stderr), "stdout_preview": result.stdout[:500] if result.stdout else "", "stderr_preview": result.stderr[:500] if result.stderr else ""}, "timestamp": int(time.time() * 1000)}
            with open(r"c:\Users\Lee\Desktop\Huni\LLM_API\.cursor\debug.log", "a", encoding="utf-8") as _f: _f.write(_json.dumps(_log_data) + "\n")
            # #endregion

            # Parse JSON output
            output_text, opencode_session_id, error_msg = self._parse_output(result.stdout)

            # Store session mapping for continuation
            if opencode_session_id:
                OpenCodeExecutor._session_map[self.session_id] = opencode_session_id

            execution_time = time.time() - start_time
            files = self._get_workspace_files()
            success = result.returncode == 0 and error_msg is None

            # Log result
            self._log_result(success, result.returncode, execution_time, output_text, result.stderr, files, error_msg)

            return {
                "success": success,
                "stdout": output_text,
                "stderr": result.stderr if result.stderr else "",
                "returncode": result.returncode,
                "execution_time": execution_time,
                "files": files,
                "workspace": str(self.workspace),
                "error": error_msg
            }

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            error_msg = f"OpenCode execution timeout after {exec_timeout} seconds"

            self._log_error("TimeoutExpired", error_msg, execution_time)

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

    def _build_command(self, instruction: str) -> List[str]:
        """Build opencode run command"""
        # On Windows, use .cmd extension for npm global binaries
        import sys
        opencode_cmd = config.OPENCODE_PATH
        if sys.platform == "win32" and not opencode_cmd.endswith(".cmd"):
            opencode_cmd = f"{opencode_cmd}.cmd"

        cmd = [
            opencode_cmd,
            "run",
            instruction,
            "--format", "json",
            "--attach", self._server.server_url,
            "--model", f"{config.OPENCODE_PROVIDER}/{config.OPENCODE_MODEL}",
            # Note: Working directory is set via subprocess cwd parameter, not CLI flag
        ]

        # Continue existing session if available
        if self.session_id in OpenCodeExecutor._session_map:
            opencode_session = OpenCodeExecutor._session_map[self.session_id]
            cmd.extend(["--session", opencode_session])

        # #region agent log
        import json as _json
        _log_data = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A,B,D", "location": "opencode_tool.py:_build_command", "message": "Command built", "data": {"opencode_cmd": opencode_cmd, "instruction_len": len(instruction), "instruction_preview": instruction[:100], "server_url": self._server.server_url, "model": f"{config.OPENCODE_PROVIDER}/{config.OPENCODE_MODEL}", "workspace": str(self.workspace), "full_cmd": cmd}, "timestamp": int(time.time() * 1000)}
        with open(r"c:\Users\Lee\Desktop\Huni\LLM_API\.cursor\debug.log", "a", encoding="utf-8") as _f: _f.write(_json.dumps(_log_data) + "\n")
        # #endregion

        return cmd

    def _parse_output(self, stdout: str) -> tuple:
        """
        Parse OpenCode output - extract session ID and return full output

        Returns:
            Tuple of (text_output, opencode_session_id, error_message)
        """
        session_id = None
        error_msg = None
        output_parts = []

        for line in stdout.strip().split('\n'):
            if not line:
                continue

            # Try to extract session ID and error from JSON events
            try:
                event = json.loads(line)
                event_type = event.get("type")
                part = event.get("part", {})

                # Extract session ID from any event
                if not session_id:
                    session_id = event.get("sessionID") or part.get("sessionID")

                # Check for errors
                if event_type == "error":
                    error_msg = part.get("message") or part.get("error") or str(part)

                # Extract any text content from events
                for key in ["text", "content", "output", "stdout", "result"]:
                    value = part.get(key)
                    if value and isinstance(value, str) and value.strip():
                        output_parts.append(value)
                        break

            except json.JSONDecodeError:
                # Non-JSON line - include if not a log prefix
                if not line.startswith("INFO ") and not line.startswith("DEBUG "):
                    output_parts.append(line)

        # Return combined output
        return "\n".join(output_parts), session_id, error_msg

    def _get_workspace_files(self) -> Dict[str, Any]:
        """Get list of files in workspace with metadata"""
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
            print(f"[OPENCODE] Warning: Failed to list files: {e}")
        return files

    def read_file(self, filename: str) -> Optional[str]:
        """Read a file from workspace"""
        file_path = self.workspace / filename
        if not file_path.exists():
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

    def list_files(self) -> List[str]:
        """List all files in workspace"""
        try:
            return [f.name for f in self.workspace.iterdir() if f.is_file()]
        except Exception:
            return []

    def clear_workspace(self) -> None:
        """Clear all files in workspace"""
        import shutil
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Logging Methods
    # =========================================================================

    def _log_start(self, instruction: str, timeout: int) -> None:
        """Log execution start"""
        log_to_prompts_file("\n\n")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file("TOOL EXECUTION: python_coder (OpenCode)")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_to_prompts_file(f"Session ID: {self.session_id}")
        log_to_prompts_file(f"Workspace: {self.workspace}")
        log_to_prompts_file(f"Timeout: {timeout}s")
        log_to_prompts_file("")
        log_to_prompts_file("INSTRUCTION:")
        for line in instruction.split('\n'):
            log_to_prompts_file(f"  {line}")

        print("\n" + "=" * 80)
        print("[OPENCODE] execute() called")
        print("=" * 80)
        print(f"Session ID: {self.session_id}")
        print(f"Workspace: {self.workspace}")
        print(f"Instruction: {instruction[:100]}...")

    def _log_result(
        self,
        success: bool,
        returncode: int,
        execution_time: float,
        stdout: str,
        stderr: str,
        files: Dict[str, Any],
        error: Optional[str]
    ) -> None:
        """Log execution result"""
        log_to_prompts_file("")
        log_to_prompts_file("OUTPUT:")
        log_to_prompts_file(f"  Status: {'SUCCESS' if success else 'FAILED'}")
        log_to_prompts_file(f"  Return Code: {returncode}")
        log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")

        if stdout:
            log_to_prompts_file("")
            log_to_prompts_file("STDOUT:")
            for line in stdout.split('\n')[:50]:  # Limit log lines
                log_to_prompts_file(f"  {line}")

        if stderr:
            log_to_prompts_file("")
            log_to_prompts_file("STDERR:")
            for line in stderr.split('\n')[:20]:
                log_to_prompts_file(f"  {line}")

        if error:
            log_to_prompts_file("")
            log_to_prompts_file(f"ERROR: {error}")

        if files:
            log_to_prompts_file("")
            log_to_prompts_file("FILES:")
            for filename, meta in files.items():
                log_to_prompts_file(f"  {filename} ({meta['size']} bytes)")

        log_to_prompts_file("")
        log_to_prompts_file("=" * 80)

        # Console output
        print(f"\n[OPENCODE] Completed in {execution_time:.2f}s")
        print(f"[OPENCODE] Success: {success}")
        if stdout:
            preview = stdout[:300] + "..." if len(stdout) > 300 else stdout
            print(f"[OPENCODE] Output:\n{preview}")
        if files:
            print(f"[OPENCODE] Files: {list(files.keys())}")

    def _log_error(self, error_type: str, error_msg: str, execution_time: float) -> None:
        """Log execution error"""
        log_to_prompts_file("")
        log_to_prompts_file(f"ERROR: {error_type}")
        log_to_prompts_file(f"  {error_msg}")
        log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")
        log_to_prompts_file("")
        log_to_prompts_file("=" * 80)

        print(f"\n[OPENCODE] ERROR: {error_msg}")
