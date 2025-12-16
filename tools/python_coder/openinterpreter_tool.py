"""
OpenInterpreter Python Code Executor
Wraps Open Interpreter for ReAct agent integration with automatic retry on errors
"""
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
    OpenInterpreter-based Python code executor with automatic error correction

    Features:
    - Automatic retry on execution errors (up to MAX_RETRIES)
    - Error context accumulation across retries
    - Same workspace as native executor (session-based)
    - Returns standardized ToolResponse format
    """

    def __init__(self, session_id: str):
        """
        Initialize OpenInterpreter executor for a session

        Args:
            session_id: Session ID for workspace isolation

        Raises:
            ImportError: If open-interpreter is not installed
        """
        super().__init__(session_id)

        # Import Open Interpreter (strict - raise error if not available)
        try:
            from interpreter import interpreter
            self.interpreter = interpreter
        except ImportError as e:
            raise ImportError(
                "Open Interpreter is not installed. "
                "Install it with: pip install open-interpreter\n"
                "Or set PYTHON_EXECUTOR_MODE='native' in config.py"
            ) from e

        # Configure workspace (same as native executor)
        self.workspace = config.PYTHON_WORKSPACE_DIR / session_id
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Configure Open Interpreter
        self._configure_interpreter()

        # Load system prompt
        self.system_prompt = self._load_system_prompt()

    def _configure_interpreter(self):
        """Configure Open Interpreter with settings from config"""
        # Get model from TOOL_MODELS config
        model = config.TOOL_MODELS.get("python_coder", config.OLLAMA_MODEL)

        # Configure LLM backend
        self.interpreter.llm.model = f"ollama/{model}"
        self.interpreter.llm.api_base = config.OLLAMA_HOST

        # Configure execution settings
        self.interpreter.auto_run = config.PYTHON_CODER_OPENINTERPRETER_AUTO_RUN
        self.interpreter.offline = config.PYTHON_CODER_OPENINTERPRETER_OFFLINE

        # Set safe mode if enabled
        if config.PYTHON_CODER_OPENINTERPRETER_SAFE_MODE:
            self.interpreter.safe_mode = "ask"  # or "auto" depending on OI version

        # Configure working directory
        self.interpreter.system_message = f"You are a Python code execution assistant. Working directory: {self.workspace}"

        print(f"[OpenInterpreter] Configured:")
        print(f"  Model: {model}")
        print(f"  API Base: {config.OLLAMA_HOST}")
        print(f"  Workspace: {self.workspace}")
        print(f"  Auto-run: {self.interpreter.auto_run}")
        print(f"  Offline: {self.interpreter.offline}")

    def _load_system_prompt(self) -> str:
        """Load system prompt from /prompts directory"""
        prompt_path = config.PROMPTS_DIR / "tools" / "python_coder_openinterpreter.txt"

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"[WARNING] Prompt file not found: {prompt_path}")
            print("[WARNING] Using default prompt")
            return (
                "You are a Python code execution assistant.\n"
                "Execute code in the workspace and report results clearly.\n"
                "If you encounter errors, analyze them and attempt to fix the code.\n"
                "Always use the workspace directory for file operations."
            )

    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code with automatic retry on errors

        Args:
            code: Python code to execute (or natural language instruction)
            timeout: Execution timeout (currently not enforced by OI)
            context: Additional context from ReAct agent

        Returns:
            Standardized execution result dictionary:
            {
                "success": bool,
                "stdout": str,
                "stderr": str,
                "returncode": int,
                "execution_time": float,
                "files": dict,
                "workspace": str,
                "error": Optional[str]
            }
        """
        start_time = time.time()
        max_retries = config.PYTHON_CODER_MAX_RETRIES
        error_history = []

        # Log to prompts.log
        log_to_prompts_file("\n\n")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"TOOL EXECUTION: python_coder (OpenInterpreter mode)")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_to_prompts_file(f"Session ID: {self.session_id}")
        log_to_prompts_file(f"Workspace: {self.workspace}")
        log_to_prompts_file(f"Max Retries: {max_retries}")
        log_to_prompts_file(f"Code/Instruction Length: {len(code)} chars")
        log_to_prompts_file(f"")
        log_to_prompts_file(f"INSTRUCTION:")
        for line in code.split('\n'):
            log_to_prompts_file(f"  {line}")

        # Console logging
        print("\n" + "=" * 80)
        print("[OPENINTERPRETER] execute() called")
        print("=" * 80)
        print(f"Session ID: {self.session_id}")
        print(f"Workspace: {self.workspace}")
        print(f"Instruction length: {len(code)} chars")
        print(f"Max retries: {max_retries}")

        # Retry loop
        for attempt in range(max_retries):
            try:
                print(f"\n[OPENINTERPRETER] Attempt {attempt + 1}/{max_retries}")
                log_to_prompts_file(f"\n--- Attempt {attempt + 1}/{max_retries} ---")

                # Build enhanced instruction with error context
                enhanced_instruction = self._build_instruction(code, error_history, context)

                # Execute via Open Interpreter
                print(f"[OPENINTERPRETER] Calling interpreter.chat()...")
                response = self.interpreter.chat(enhanced_instruction)

                # Parse response
                result = self._parse_response(response, attempt, start_time)

                # Check if execution was successful
                if result["success"]:
                    print(f"[OPENINTERPRETER] [SUCCESS] Completed in attempt {attempt + 1}")
                    log_to_prompts_file(f"\n[SUCCESS] Completed in attempt {attempt + 1}")
                    log_to_prompts_file(f"Execution time: {result['execution_time']:.2f}s")
                    log_to_prompts_file("=" * 80)
                    return result
                else:
                    # Execution failed, but we got a response
                    error_msg = result.get("error", "Unknown error")
                    error_history.append(f"Attempt {attempt + 1}: {error_msg}")

                    print(f"[OPENINTERPRETER] [FAILED] Attempt {attempt + 1}: {error_msg}")
                    log_to_prompts_file(f"[FAILED] {error_msg}")

                    # If last attempt, return failure
                    if attempt == max_retries - 1:
                        result["error"] = f"Failed after {max_retries} attempts. Last error: {error_msg}"
                        log_to_prompts_file(f"\n[FINAL FAILURE] All {max_retries} attempts exhausted")
                        log_to_prompts_file("=" * 80)
                        return result

                    # Continue to next retry
                    print(f"[OPENINTERPRETER] Retrying with error context...")

            except Exception as e:
                # Unexpected error during execution
                error_msg = f"Exception: {type(e).__name__}: {str(e)}"
                error_history.append(f"Attempt {attempt + 1}: {error_msg}")

                print(f"[OPENINTERPRETER] [ERROR] {error_msg}")
                log_to_prompts_file(f"[ERROR] {error_msg}")

                # If last attempt, return failure
                if attempt == max_retries - 1:
                    execution_time = time.time() - start_time
                    log_to_prompts_file(f"\n[FINAL FAILURE] All {max_retries} attempts exhausted")
                    log_to_prompts_file(f"Execution time: {execution_time:.2f}s")
                    log_to_prompts_file("=" * 80)

                    return {
                        "success": False,
                        "stdout": "",
                        "stderr": "\n".join(error_history),
                        "returncode": -1,
                        "execution_time": execution_time,
                        "files": self._get_workspace_files(),
                        "workspace": str(self.workspace),
                        "error": f"Failed after {max_retries} attempts. Last error: {error_msg}"
                    }

                # Continue to next retry
                print(f"[OPENINTERPRETER] Retrying after exception...")

        # Should never reach here, but just in case
        execution_time = time.time() - start_time
        return {
            "success": False,
            "stdout": "",
            "stderr": "Unexpected: retry loop completed without return",
            "returncode": -1,
            "execution_time": execution_time,
            "files": self._get_workspace_files(),
            "workspace": str(self.workspace),
            "error": "Retry loop error"
        }

    def _build_instruction(
        self,
        original_code: str,
        error_history: List[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build enhanced instruction with error context and system prompt

        Args:
            original_code: Original code or instruction
            error_history: List of previous error messages
            context: Additional context from ReAct agent

        Returns:
            Enhanced instruction string
        """
        parts = [self.system_prompt, ""]

        # Add workspace info
        parts.append(f"Working directory: {self.workspace}")
        parts.append("")

        # Add context if provided
        if context:
            if context.get("user_query"):
                parts.append(f"User query: {context['user_query']}")
            if context.get("current_thought"):
                parts.append(f"Current thought: {context['current_thought']}")
            parts.append("")

        # Add original instruction
        parts.append("Instruction:")
        parts.append(original_code)
        parts.append("")

        # Add error history if retrying
        if error_history:
            parts.append("Previous attempts failed with these errors:")
            for error in error_history:
                parts.append(f"  - {error}")
            parts.append("")
            parts.append("Please analyze the errors above and fix the code before executing.")
            parts.append("")

        return "\n".join(parts)

    def _parse_response(
        self,
        response: Any,
        attempt: int,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Parse Open Interpreter response into standardized format

        Args:
            response: Response from interpreter.chat()
            attempt: Current attempt number
            start_time: Execution start time

        Returns:
            Standardized result dictionary
        """
        execution_time = time.time() - start_time

        # Open Interpreter returns string response
        response_str = str(response) if response else ""

        # Simple heuristic: check for common error indicators
        # This is a simplified check - Open Interpreter's response format may vary
        is_error = any(keyword in response_str.lower() for keyword in [
            "error", "exception", "traceback", "failed", "syntax error"
        ])

        # Check for success indicators
        is_success = not is_error and len(response_str) > 0

        # Build result
        result = {
            "success": is_success,
            "stdout": response_str if is_success else "",
            "stderr": response_str if is_error else "",
            "returncode": 0 if is_success else 1,
            "execution_time": execution_time,
            "files": self._get_workspace_files(),
            "workspace": str(self.workspace),
            "error": response_str if is_error else None
        }

        # Log response
        log_to_prompts_file(f"\nRESPONSE:")
        for line in response_str.split('\n'):
            log_to_prompts_file(f"  {line}")

        return result

    def _get_workspace_files(self) -> Dict[str, Any]:
        """
        Get list of files in workspace (same as native executor)

        Returns:
            Dictionary of files with metadata
        """
        files = {}

        if not self.workspace.exists():
            return files

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
        if not self.workspace.exists():
            return []

        return [f.name for f in self.workspace.iterdir() if f.is_file()]

    def clear_workspace(self):
        """Clear all files in workspace"""
        import shutil
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
