"""
OpenInterpreter Python Code Executor
Executes natural language instructions by generating and running Python code
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import config
from tools.python_coder.base import BasePythonExecutor


def log_to_prompts_file(message: str):
    """Write message to prompts.log"""
    try:
        with open(config.PROMPTS_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"[WARNING] Failed to write to prompts.log: {e}")


class OpenInterpreterExecutor(BasePythonExecutor):
    """
    OpenInterpreter-based executor for natural language code generation and execution

    Receives natural language instructions, generates Python code, executes it,
    and automatically retries on errors.
    """

    def __init__(self, session_id: str):
        """
        Initialize OpenInterpreter executor

        Args:
            session_id: Session ID for workspace isolation
        """
        super().__init__(session_id)
        self.workspace = config.PYTHON_WORKSPACE_DIR / session_id
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.timeout = config.PYTHON_EXECUTOR_TIMEOUT
        self.max_output_size = config.PYTHON_EXECUTOR_MAX_OUTPUT_SIZE
        self.max_retries = config.PYTHON_CODER_MAX_RETRIES

        self._interpreter = None

    def _get_interpreter(self):
        """
        Get or create OpenInterpreter instance with configuration

        Returns:
            Configured interpreter instance

        Raises:
            ImportError: If open-interpreter is not installed
        """
        if self._interpreter is not None:
            return self._interpreter

        print(f"[OPENINTERPRETER] Initializing interpreter...")

        try:
            from interpreter import interpreter
        except ImportError:
            raise ImportError(
                "OpenInterpreter not installed. "
                "Install with: pip install open-interpreter"
            )

        # Load system message
        system_prompt_path = config.PROMPTS_DIR / "tools" / "python_coder_openinterpreter.txt"
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_message = f.read()

        # Configure interpreter for Ollama
        # Enable litellm verbose mode for debugging
        import litellm
        litellm.set_verbose = True

        interpreter.llm.model = f"ollama/{config.OLLAMA_MODEL}"
        interpreter.llm.api_base = config.OLLAMA_HOST
        interpreter.llm.temperature = config.TOOL_PARAMETERS.get("python_coder", {}).get("temperature", 0.2)
        interpreter.auto_run = config.PYTHON_CODER_OPENINTERPRETER_AUTO_RUN
        interpreter.offline = config.PYTHON_CODER_OPENINTERPRETER_OFFLINE
        interpreter.safe_mode = config.PYTHON_CODER_OPENINTERPRETER_SAFE_MODE
        interpreter.system_message = system_message

        print(f"[OPENINTERPRETER] Configuration:")
        print(f"  Model: {interpreter.llm.model}")
        print(f"  API Base: {interpreter.llm.api_base}")
        print(f"  Temperature: {interpreter.llm.temperature}")
        print(f"  Auto-run: {interpreter.auto_run}")
        print(f"  Offline: {interpreter.offline}")
        print(f"  Safe mode: {interpreter.safe_mode}")
        print(f"  Workspace: {self.workspace}")

        self._interpreter = interpreter
        return self._interpreter

    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute natural language instruction using OpenInterpreter

        Args:
            code: Natural language instruction (NOT Python code)
            timeout: Execution timeout (note: not enforced by OpenInterpreter)
            context: Additional context (unused)

        Returns:
            Standardized execution result dictionary
        """
        exec_timeout = timeout or self.timeout
        start_time = time.time()

        # Log execution start
        log_to_prompts_file("\n\n")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"TOOL EXECUTION: python_coder (OpenInterpreter)")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_to_prompts_file(f"Session ID: {self.session_id}")
        log_to_prompts_file(f"Workspace: {self.workspace}")
        log_to_prompts_file(f"Instruction Length: {len(code)} chars")
        log_to_prompts_file(f"Timeout: {exec_timeout}s")
        log_to_prompts_file(f"Max Retries: {self.max_retries}")
        log_to_prompts_file(f"")
        log_to_prompts_file(f"NATURAL LANGUAGE INSTRUCTION:")
        for line in code.split('\n'):
            log_to_prompts_file(f"  {line}")

        print("\n" + "=" * 80)
        print("[OPENINTERPRETER] execute() called")
        print("=" * 80)
        print(f"Session ID: {self.session_id}")
        print(f"Workspace: {self.workspace}")
        print(f"Instruction: {code[:100]}...")
        print(f"Timeout: {exec_timeout}s")
        print(f"Max retries: {self.max_retries}")

        try:
            # Get interpreter instance
            interpreter = self._get_interpreter()

            # Change to workspace directory
            original_cwd = os.getcwd()
            os.chdir(self.workspace)
            print(f"[OPENINTERPRETER] Changed to workspace: {self.workspace}")

            try:
                # Execute instruction
                print(f"\n[OPENINTERPRETER] Sending instruction to interpreter...")

                log_to_prompts_file(f"")
                log_to_prompts_file(f"EXECUTING...")

                response_chunks = interpreter.chat(
                    code,
                    display=False,
                    stream=False,
                    blocking=True
                )

                print(f"[OPENINTERPRETER] Received {len(response_chunks)} response chunks")
                log_to_prompts_file(f"Response chunks: {len(response_chunks)}")

                # Parse response into standardized format
                result = self._parse_response(response_chunks, start_time)

                # Add workspace files
                result["files"] = self._get_workspace_files()
                result["workspace"] = str(self.workspace)

                # Log result
                execution_time = time.time() - start_time

                log_to_prompts_file(f"")
                log_to_prompts_file(f"OUTPUT:")
                log_to_prompts_file(f"  Status: {'SUCCESS' if result['success'] else 'FAILED'}")
                log_to_prompts_file(f"  Return Code: {result['returncode']}")
                log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")

                if result["stdout"]:
                    log_to_prompts_file(f"")
                    log_to_prompts_file(f"STDOUT:")
                    for line in result["stdout"].split('\n'):
                        log_to_prompts_file(f"  {line}")

                if result["stderr"]:
                    log_to_prompts_file(f"")
                    log_to_prompts_file(f"STDERR:")
                    for line in result["stderr"].split('\n'):
                        log_to_prompts_file(f"  {line}")

                if result["files"]:
                    log_to_prompts_file(f"")
                    log_to_prompts_file(f"FILES:")
                    for filename, meta in result["files"].items():
                        log_to_prompts_file(f"  {filename} ({meta['size']} bytes)")

                log_to_prompts_file(f"")
                log_to_prompts_file("=" * 80)

                # Console output
                print(f"\n[OPENINTERPRETER] Execution completed in {execution_time:.2f}s")
                print(f"[OPENINTERPRETER] Success: {result['success']}")
                if result["stdout"]:
                    preview = result["stdout"][:300]
                    if len(result["stdout"]) > 300:
                        preview += "..."
                    print(f"[OPENINTERPRETER] STDOUT:\n{preview}")
                if result["stderr"]:
                    preview = result["stderr"][:300]
                    if len(result["stderr"]) > 300:
                        preview += "..."
                    print(f"[OPENINTERPRETER] STDERR:\n{preview}")
                if result["files"]:
                    print(f"[OPENINTERPRETER] Files: {list(result['files'].keys())}")

                return result

            finally:
                # Restore original directory
                os.chdir(original_cwd)
                print(f"[OPENINTERPRETER] Restored original directory")

        except ImportError as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            print(f"\n[OPENINTERPRETER] IMPORT ERROR: {error_msg}")

            log_to_prompts_file(f"")
            log_to_prompts_file(f"ERROR: ImportError")
            log_to_prompts_file(f"  {error_msg}")
            log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")
            log_to_prompts_file(f"")
            log_to_prompts_file("=" * 80)

            raise

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            print(f"\n[OPENINTERPRETER] ERROR: {error_msg}")

            # Print full traceback for debugging
            import traceback
            print(f"[OPENINTERPRETER] Full traceback:")
            traceback.print_exc()

            log_to_prompts_file(f"")
            log_to_prompts_file(f"ERROR: {type(e).__name__}")
            log_to_prompts_file(f"  {error_msg}")
            log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")
            log_to_prompts_file(f"")
            log_to_prompts_file("=" * 80)

            # Check if connection error
            if "connect" in error_msg.lower() or "connection" in error_msg.lower():
                raise ConnectionError(f"Cannot connect to Ollama at {config.OLLAMA_HOST}: {error_msg}")

            raise

    def _parse_response(
        self,
        response_chunks: List[Dict[str, Any]],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Parse OpenInterpreter response chunks into standardized format

        Args:
            response_chunks: List of response chunks from interpreter
            start_time: Execution start time

        Returns:
            Standardized result dictionary
        """
        print(f"\n[OPENINTERPRETER] Parsing {len(response_chunks)} chunks...")

        stdout_parts = []
        stderr_parts = []
        success = True
        returncode = 0

        for i, chunk in enumerate(response_chunks):
            chunk_type = chunk.get("type", "unknown")
            content = chunk.get("content", "")

            print(f"[OPENINTERPRETER] Chunk {i}: type={chunk_type}")

            if chunk_type == "code":
                # Code execution chunk - log it
                print(f"[OPENINTERPRETER]   Code: {str(content)[:100]}...")

            elif chunk_type == "console":
                # Console output chunk
                format_type = chunk.get("format", "output")

                if format_type == "output":
                    stdout_parts.append(str(content))
                    print(f"[OPENINTERPRETER]   Output: {str(content)[:100]}...")
                elif format_type == "error":
                    stderr_parts.append(str(content))
                    success = False
                    returncode = 1
                    print(f"[OPENINTERPRETER]   Error: {str(content)[:100]}...")

            elif chunk_type == "message":
                # Message from interpreter
                stdout_parts.append(str(content))
                print(f"[OPENINTERPRETER]   Message: {str(content)[:100]}...")

        # Combine outputs
        stdout = "\n".join(stdout_parts).strip()
        stderr = "\n".join(stderr_parts).strip()

        # Limit output size
        if len(stdout) > self.max_output_size:
            stdout = stdout[:self.max_output_size] + "\n... (output truncated)"
        if len(stderr) > self.max_output_size:
            stderr = stderr[:self.max_output_size] + "\n... (output truncated)"

        execution_time = time.time() - start_time

        print(f"[OPENINTERPRETER] Parse complete:")
        print(f"  Success: {success}")
        print(f"  Return code: {returncode}")
        print(f"  STDOUT length: {len(stdout)}")
        print(f"  STDERR length: {len(stderr)}")
        print(f"  Execution time: {execution_time:.2f}s")

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "execution_time": execution_time,
            "error": stderr if not success else None
        }

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
            print(f"[OPENINTERPRETER] Warning: Failed to list files: {e}")

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
            print(f"[OPENINTERPRETER] Warning: Failed to read '{filename}': {e}")
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
            print(f"[OPENINTERPRETER] Warning: Failed to list files: {e}")
            return []

    def clear_workspace(self):
        """Clear all files in workspace"""
        import shutil

        try:
            if self.workspace.exists():
                shutil.rmtree(self.workspace)
            self.workspace.mkdir(parents=True, exist_ok=True)
            print(f"[OPENINTERPRETER] Workspace cleared: {self.workspace}")
        except Exception as e:
            print(f"[OPENINTERPRETER] Warning: Failed to clear workspace: {e}")
