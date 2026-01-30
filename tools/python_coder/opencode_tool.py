"""
OpenCode Python Code Executor
Two-stage execution: OpenCode generates code, then Python executor runs it
"""
import json
import queue
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
from tools.python_coder.base import BasePythonExecutor


def log_to_prompts_file(message: str) -> None:
    """Write message to prompts.log"""
    try:
        with open(config.PROMPTS_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"[WARNING] Failed to write to prompts.log: {e}")


class OpenCodeExecutor(BasePythonExecutor):
    """
    OpenCode-based executor with two-stage execution:

    Stage 1: OpenCode generates Python code and saves to file
    Stage 2: Python subprocess runs the generated code

    This separation ensures clean code generation and reliable execution.
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
        self.python_timeout = getattr(config, 'PYTHON_EXECUTOR_TIMEOUT', 60)
        self.max_output_size = config.PYTHON_EXECUTOR_MAX_OUTPUT_SIZE

    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute natural language instruction using two-stage approach:
        1. OpenCode generates Python code
        2. Python subprocess runs the generated code

        Args:
            code: Natural language instruction
            timeout: Execution timeout in seconds
            context: Additional context (unused)

        Returns:
            Standardized execution result dictionary with combined outputs
        """
        instruction = code
        exec_timeout = timeout or self.timeout
        start_time = time.time()

        prefix = (
            "ULTRAWORK MODE\n"
            "Your job is to write a Python script that performs the given task.\n"
            "Before writing the code:\n"
            "  1. Think if you have all necessary pre-steps to perform the task\n"
            "  2. Write the code and save it to a .py file in the current directory\n"
            "  3. Include print statements explaining what the task is and what the code does\n"
            "  4. Add print statements to display key information\n"
            "Before running the code, verify the implementation is correct.\n"
            "Then run the code, check the output, and fully explain all results without missing any details:\n"
            "THINK HARD AND MAKE NO MISTAKE!\n"
        )

        instruction = prefix + instruction

        # Log execution start
        self._log_start(instruction, exec_timeout)

        # Track files before OpenCode runs
        files_before = set(self._get_python_files())

        # =====================================================================
        # STAGE 1: OpenCode generates Python code
        # =====================================================================
        print(f"\n[OPENCODE] Stage 1: Generating Python code...")
        self._log_stage("STAGE 1: CODE GENERATION (OpenCode)")

        opencode_result = self._run_opencode(instruction, exec_timeout)

        if not opencode_result["success"]:
            # OpenCode failed - return error
            execution_time = time.time() - start_time
            self._log_result(False, -1, execution_time, opencode_result["output"], "", {}, opencode_result["error"])
            return {
                "success": False,
                "stdout": opencode_result["output"],
                "stderr": opencode_result["error"] or "",
                "returncode": -1,
                "execution_time": execution_time,
                "files": self._get_workspace_files(),
                "workspace": str(self.workspace),
                "error": opencode_result["error"]
            }

        # =====================================================================
        # STAGE 2: Find and run generated Python file
        # =====================================================================
        print(f"\n[OPENCODE] Stage 2: Running generated Python code...")
        self._log_stage("STAGE 2: CODE EXECUTION (Python)")

        # Find newly created Python file
        files_after = set(self._get_python_files())
        new_files = files_after - files_before

        # Try to find the Python file to run
        python_file = self._find_python_file_to_run(new_files, opencode_result["output"])

        if not python_file:
            # No Python file found - return OpenCode output with warning
            execution_time = time.time() - start_time
            warning = "No Python file was generated by OpenCode"
            print(f"[OPENCODE] Warning: {warning}")
            self._log_result(True, 0, execution_time, opencode_result["output"], warning, self._get_workspace_files(), None)
            return {
                "success": True,
                "stdout": opencode_result["output"],
                "stderr": warning,
                "returncode": 0,
                "execution_time": execution_time,
                "files": self._get_workspace_files(),
                "workspace": str(self.workspace),
                "error": None
            }

        # Run the Python file
        remaining_timeout = max(30, exec_timeout - (time.time() - start_time))
        python_result = self._run_python_file(python_file, int(remaining_timeout))

        # =====================================================================
        # Combine outputs from both stages
        # =====================================================================
        execution_time = time.time() - start_time
        files = self._get_workspace_files()

        # Build combined output
        combined_output = self._combine_outputs(
            opencode_output=opencode_result["output"],
            python_file=python_file,
            python_stdout=python_result["stdout"],
            python_stderr=python_result["stderr"],
            python_returncode=python_result["returncode"]
        )

        success = python_result["returncode"] == 0
        error_msg = None if success else f"Python execution failed with code {python_result['returncode']}"

        self._log_result(success, python_result["returncode"], execution_time, combined_output, python_result["stderr"], files, error_msg)

        return {
            "success": success,
            "stdout": combined_output,
            "stderr": python_result["stderr"],
            "returncode": python_result["returncode"],
            "execution_time": execution_time,
            "files": files,
            "workspace": str(self.workspace),
            "error": error_msg
        }

    def _run_opencode(self, instruction: str, timeout: int) -> Dict[str, Any]:
        """
        Run OpenCode to generate Python code

        Returns:
            Dict with keys: success, output, session_id, error
        """
        cmd = self._build_command(instruction)
        print(f"[OPENCODE] Command: {' '.join(cmd[:6])}...")

        stdout_lines = []
        stderr_lines = []

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.workspace),
                encoding='utf-8',
                errors='replace'
            )

            # Read stderr in background
            stderr_queue = queue.Queue()
            def read_stderr():
                for line in process.stderr:
                    stderr_queue.put(line)

            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            # Stream stdout
            deadline = time.time() + timeout
            for line in process.stdout:
                if time.time() > deadline:
                    process.kill()
                    raise subprocess.TimeoutExpired(cmd, timeout)

                stdout_lines.append(line)
                self._log_stream_line(line.rstrip('\n\r'))

            # Wait for completion
            remaining = max(1, deadline - time.time())
            returncode = process.wait(timeout=remaining)

            # Collect stderr
            stderr_thread.join(timeout=1)
            while not stderr_queue.empty():
                stderr_lines.append(stderr_queue.get_nowait())

            # Parse output
            full_stdout = ''.join(stdout_lines)
            output_text, session_id, error_msg = self._parse_output(full_stdout)

            # Store session for continuation
            if session_id:
                OpenCodeExecutor._session_map[self.session_id] = session_id

            return {
                "success": returncode == 0 and error_msg is None,
                "output": output_text,
                "session_id": session_id,
                "error": error_msg
            }

        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except:
                pass
            return {
                "success": False,
                "output": ''.join(stdout_lines),
                "session_id": None,
                "error": f"OpenCode timeout after {timeout}s"
            }
        except Exception as e:
            return {
                "success": False,
                "output": ''.join(stdout_lines),
                "session_id": None,
                "error": f"OpenCode error: {str(e)}"
            }

    def _run_python_file(self, python_file: Path, timeout: int) -> Dict[str, Any]:
        """
        Run a Python file and capture output

        Returns:
            Dict with keys: stdout, stderr, returncode
        """
        print(f"[OPENCODE] Running: python {python_file.name}")
        log_to_prompts_file(f"  Executing: python {python_file.name}")

        try:
            result = subprocess.run(
                ["python", python_file.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
                encoding='utf-8',
                errors='replace'
            )

            log_to_prompts_file(f"  Return code: {result.returncode}")
            if result.stdout:
                log_to_prompts_file(f"  Stdout: {result.stdout[:500]}")
            if result.stderr:
                log_to_prompts_file(f"  Stderr: {result.stderr[:500]}")

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Python execution timeout after {timeout}s",
                "returncode": -1
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Python execution error: {str(e)}",
                "returncode": -1
            }

    def _build_command(self, instruction: str) -> List[str]:
        """Build opencode run command for code generation only"""
        import sys
        opencode_cmd = config.OPENCODE_PATH
        if sys.platform == "win32" and not opencode_cmd.endswith(".cmd"):
            opencode_cmd = f"{opencode_cmd}.cmd"

        # Instruction tells OpenCode to ONLY generate code, not run it
        prefixed_instruction = (
            f"ultrawork, Your job is to write a Python script that performs the given task. "
            f"Save the code to a .py file in the current directory. "
            f"DO NOT run the code yourself - just write and save it. "
            f"After saving, output the filename you created. "
            f"Task: {instruction} "
        )

        cmd = [
            opencode_cmd,
            "run",
            prefixed_instruction,
            "--format", "json",
            "--model", f"{config.OPENCODE_PROVIDER}/{config.OPENCODE_MODEL}",
        ]

        # Continue existing session if available
        if self.session_id in OpenCodeExecutor._session_map:
            opencode_session = OpenCodeExecutor._session_map[self.session_id]
            cmd.extend(["--session", opencode_session])

        return cmd

    def _parse_output(self, stdout: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Parse OpenCode output - extract session ID and text content

        Returns:
            Tuple of (text_output, opencode_session_id, error_message)
        """
        session_id = None
        error_msg = None
        output_parts = []

        for line in stdout.strip().split('\n'):
            if not line:
                continue

            try:
                event = json.loads(line)
                event_type = event.get("type")
                part = event.get("part", {})

                # Extract session ID
                if not session_id:
                    session_id = event.get("sessionID") or part.get("sessionID")

                # Check for errors
                if event_type == "error":
                    error_msg = part.get("message") or part.get("error") or str(part)

                # Extract text content
                for key in ["text", "content", "output", "stdout", "result"]:
                    value = part.get(key)
                    if value and isinstance(value, str) and value.strip():
                        output_parts.append(value)
                        break

            except json.JSONDecodeError:
                if not line.startswith("INFO ") and not line.startswith("DEBUG "):
                    output_parts.append(line)

        combined_output = "\n".join(output_parts)

        # Fallback if no parsed output
        if not combined_output and stdout.strip():
            raw_lines = [l for l in stdout.strip().split('\n')
                        if l and not l.startswith("INFO ") and not l.startswith("DEBUG ")]
            combined_output = "\n".join(raw_lines)

        return combined_output, session_id, error_msg

    def _get_python_files(self) -> List[Path]:
        """Get list of Python files in workspace"""
        try:
            return list(self.workspace.glob("*.py"))
        except Exception:
            return []

    def _find_python_file_to_run(self, new_files: set, opencode_output: str) -> Optional[Path]:
        """
        Find the Python file to run

        Priority:
        1. Newly created .py file
        2. File mentioned in OpenCode output
        3. Most recently modified .py file
        """
        # Priority 1: New file created during this run
        if new_files:
            # If multiple, prefer the one mentioned in output
            for f in new_files:
                if f.name in opencode_output:
                    print(f"[OPENCODE] Found new file mentioned in output: {f.name}")
                    return f
            # Otherwise return the first new file
            file = list(new_files)[0]
            print(f"[OPENCODE] Found new file: {file.name}")
            return file

        # Priority 2: Extract filename from OpenCode output
        # Look for patterns like "saved to solution.py" or "created file.py"
        patterns = [
            r'saved (?:to |as )?["\']?(\w+\.py)["\']?',
            r'created ["\']?(\w+\.py)["\']?',
            r'wrote ["\']?(\w+\.py)["\']?',
            r'file[:\s]+["\']?(\w+\.py)["\']?',
            r'(\w+\.py)'  # Last resort: any .py filename
        ]

        for pattern in patterns:
            matches = re.findall(pattern, opencode_output, re.IGNORECASE)
            for match in matches:
                file_path = self.workspace / match
                if file_path.exists():
                    print(f"[OPENCODE] Found file from output pattern: {match}")
                    return file_path

        # Priority 3: Most recently modified .py file
        py_files = self._get_python_files()
        if py_files:
            # Sort by modification time, most recent first
            py_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            print(f"[OPENCODE] Using most recent .py file: {py_files[0].name}")
            return py_files[0]

        return None

    def _combine_outputs(
        self,
        opencode_output: str,
        python_file: Path,
        python_stdout: str,
        python_stderr: str,
        python_returncode: int
    ) -> str:
        """Combine outputs from OpenCode and Python execution"""
        parts = []

        # OpenCode generation summary (abbreviated)
        if opencode_output:
            # Trim to avoid huge outputs
            if len(opencode_output) > 500:
                parts.append(f"[Code Generation]\n{opencode_output[:500]}...")
            else:
                parts.append(f"[Code Generation]\n{opencode_output}")

        # Python execution output
        parts.append(f"\n[Execution: {python_file.name}]")

        if python_stdout:
            parts.append(python_stdout)

        if python_stderr and python_returncode != 0:
            parts.append(f"\n[Stderr]\n{python_stderr}")

        if python_returncode == 0:
            parts.append("\n[Execution completed successfully]")
        else:
            parts.append(f"\n[Execution failed with code {python_returncode}]")

        return "\n".join(parts)

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
        log_to_prompts_file("TOOL EXECUTION: python_coder (OpenCode Two-Stage)")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_to_prompts_file(f"Session ID: {self.session_id}")
        log_to_prompts_file(f"Workspace: {self.workspace}")
        log_to_prompts_file(f"Timeout: {timeout}s")
        log_to_prompts_file("")
        log_to_prompts_file("INSTRUCTION:")
        for line in instruction.split('\n'):
            log_to_prompts_file(f"  {line}")

        print(f"\n{'=' * 60}")
        print(f"[OPENCODE] Two-Stage Execution")
        print(f"{'=' * 60}")
        print(f"Session: {self.session_id}")
        print(f"Workspace: {self.workspace}")
        print(f"Instruction: {instruction[:100]}...")

    def _log_stage(self, stage_name: str) -> None:
        """Log stage separator"""
        log_to_prompts_file("")
        log_to_prompts_file("-" * 80)
        log_to_prompts_file(stage_name)
        log_to_prompts_file("-" * 80)

    def _log_stream_line(self, line: str) -> None:
        """Log a single line of streaming output"""
        try:
            event = json.loads(line)
            event_type = event.get("type", "unknown")
            part = event.get("part", {})

            if event_type == "text":
                text = part.get("text", "")
                if text:
                    log_to_prompts_file(f"  [TEXT] {text}")
            elif event_type == "tool_call":
                tool_name = part.get("name", "unknown")
                log_to_prompts_file(f"  [TOOL_CALL] {tool_name}")
            elif event_type == "tool_result":
                success = part.get("success", False)
                log_to_prompts_file(f"  [TOOL_RESULT] success={success}")
            elif event_type == "error":
                error = part.get("message", part.get("error", str(part)))
                log_to_prompts_file(f"  [ERROR] {error}")
            else:
                log_to_prompts_file(f"  [{event_type.upper()}] {json.dumps(part)[:200]}")
        except json.JSONDecodeError:
            if line and not line.startswith("INFO ") and not line.startswith("DEBUG "):
                log_to_prompts_file(f"  {line}")

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
        log_to_prompts_file("-" * 80)
        log_to_prompts_file("FINAL RESULT:")
        log_to_prompts_file(f"  Status: {'SUCCESS' if success else 'FAILED'}")
        log_to_prompts_file(f"  Return Code: {returncode}")
        log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")

        if stdout:
            log_to_prompts_file("")
            log_to_prompts_file("COMBINED OUTPUT:")
            for line in stdout.split('\n')[:50]:
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
        print(f"\n{'=' * 60}")
        print(f"[OPENCODE] Completed in {execution_time:.2f}s")
        print(f"[OPENCODE] Success: {success}")
        if stdout:
            preview = stdout[:400] + "..." if len(stdout) > 400 else stdout
            print(f"[OPENCODE] Output:\n{preview}")
        if files:
            print(f"[OPENCODE] Files: {list(files.keys())}")
        print(f"{'=' * 60}")

    def _log_error(self, error_type: str, error_msg: str, execution_time: float) -> None:
        """Log execution error"""
        log_to_prompts_file("")
        log_to_prompts_file(f"ERROR: {error_type}")
        log_to_prompts_file(f"  {error_msg}")
        log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")
        log_to_prompts_file("")
        log_to_prompts_file("=" * 80)

        print(f"\n[OPENCODE] ERROR: {error_msg}")
