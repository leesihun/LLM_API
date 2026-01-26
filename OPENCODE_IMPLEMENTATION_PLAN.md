# OpenCode Integration Plan (Revised)

## Design Decisions

Based on user requirements:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Providers | Ollama + llama.cpp | Match current LLM backend support |
| Sessions | Maintain across tool calls | More context for multi-step tasks |
| Output | JSON format | Machine-parseable |
| Server | Persistent mode | Reduced latency |
| Server Start | On tools_server startup | Always ready |
| Server Failure | Auto-restart once, then fail | Balance reliability + strictness |
| Session Cleanup | Let opencode manage | Simplicity |

## Principles

1. **Simple code** - No unnecessary abstractions
2. **No fallbacks** - Fail fast with clear errors
3. **All config in `config.py`** - Single source of truth
4. **All prompts in `/prompts`** - No hardcoded text
5. **Strict algorithm** - Predictable behavior

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     tools_server.py                         │
│  (starts OpenCodeServerManager on startup)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 OpenCodeServerManager                        │
│  - Singleton managing opencode server lifecycle             │
│  - Starts server on port from config                        │
│  - Auto-restart once on failure                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   OpenCodeExecutor                           │
│  - Implements BasePythonExecutor interface                  │
│  - Maintains session mapping (LLM session → opencode session)│
│  - Parses JSON output                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
tools/python_coder/
├── __init__.py              # Factory (modify)
├── base.py                  # Base class (no change)
├── native_tool.py           # Native executor (no change)
├── opencode_tool.py         # NEW: OpenCode executor
├── opencode_server.py       # NEW: Server manager
└── opencode_config.py       # NEW: Config generator

prompts/agents/
├── react_system_opencode.txt    # NEW: System prompt
└── react_thought_opencode.txt   # NEW: Thought prompt

config.py                    # Add OPENCODE_* settings
tools_server.py              # Start server on startup
```

---

## Part 1: Configuration (`config.py`)

Add these settings to `config.py`:

```python
# ============================================================================
# OpenCode Settings
# ============================================================================
# Path to opencode binary
OPENCODE_PATH: str = "opencode"

# Server settings
OPENCODE_SERVER_PORT: int = 4096
OPENCODE_SERVER_HOST: str = "127.0.0.1"

# Execution settings
OPENCODE_TIMEOUT: int = 864000  # 10 days (match other tools)

# Provider settings (used to generate opencode config)
# Provider name must match opencode provider naming: "ollama" or "llama.cpp"
OPENCODE_PROVIDER: str = "llama.cpp"  # or "ollama"
OPENCODE_MODEL: str = "default"  # model name within the provider

# Update PYTHON_EXECUTOR_MODE type hint
PYTHON_EXECUTOR_MODE: Literal["native", "nanocoder", "opencode"] = "opencode"
```

**Note**: Provider base URLs are derived from existing `OLLAMA_HOST` and `LLAMACPP_HOST` settings.

---

## Part 2: OpenCode Configuration Generator (`opencode_config.py`)

**Path**: `tools/python_coder/opencode_config.py`

```python
"""
OpenCode Configuration Generator
Generates opencode config.json based on config.py settings
"""
import json
import sys
from pathlib import Path

import config


def get_opencode_config_path() -> Path:
    """Get platform-specific opencode config directory"""
    if sys.platform == "win32":
        return Path.home() / ".config" / "opencode" / "config.json"
    else:
        return Path.home() / ".config" / "opencode" / "config.json"


def generate_opencode_config() -> Path:
    """
    Generate opencode config.json based on config.py LLM backend settings

    Returns:
        Path to generated config file

    Raises:
        ValueError: If LLM_BACKEND is invalid
    """
    config_path = get_opencode_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    providers = {}

    # Add Ollama provider
    if config.LLM_BACKEND in ["ollama", "auto"]:
        ollama_base = config.OLLAMA_HOST.rstrip("/")
        if not ollama_base.endswith("/v1"):
            ollama_base = f"{ollama_base}/v1"

        providers["ollama"] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Ollama",
            "options": {
                "baseURL": ollama_base
            },
            "models": {
                config.OLLAMA_MODEL: {
                    "name": config.OLLAMA_MODEL
                }
            }
        }

    # Add llama.cpp provider
    if config.LLM_BACKEND in ["llamacpp", "auto"]:
        llamacpp_base = config.LLAMACPP_HOST.rstrip("/")
        if not llamacpp_base.endswith("/v1"):
            llamacpp_base = f"{llamacpp_base}/v1"

        providers["llama.cpp"] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "llama.cpp",
            "options": {
                "baseURL": llamacpp_base
            },
            "models": {
                config.LLAMACPP_MODEL: {
                    "name": config.LLAMACPP_MODEL
                }
            }
        }

    if not providers:
        raise ValueError(
            f"No providers configured. LLM_BACKEND={config.LLM_BACKEND} "
            f"must be 'ollama', 'llamacpp', or 'auto'"
        )

    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": providers
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(opencode_config, f, indent=2)

    print(f"[OPENCODE] Config generated: {config_path}")
    return config_path


def ensure_opencode_config() -> Path:
    """
    Ensure opencode config exists

    Always regenerates to stay in sync with config.py

    Returns:
        Path to config file
    """
    return generate_opencode_config()
```

---

## Part 3: OpenCode Server Manager (`opencode_server.py`)

**Path**: `tools/python_coder/opencode_server.py`

```python
"""
OpenCode Server Manager
Manages the lifecycle of opencode persistent server
"""
import subprocess
import time
import threading
from typing import Optional

import config


class OpenCodeServerManager:
    """
    Singleton manager for opencode server lifecycle

    Responsibilities:
    - Start server on initialization
    - Auto-restart once on failure
    - Provide server URL for executors
    """

    _instance: Optional["OpenCodeServerManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "OpenCodeServerManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._process: Optional[subprocess.Popen] = None
        self._restart_attempted = False
        self._server_url = f"http://{config.OPENCODE_SERVER_HOST}:{config.OPENCODE_SERVER_PORT}"
        self._initialized = True

    @property
    def server_url(self) -> str:
        """Get server URL for --attach flag"""
        return self._server_url

    @property
    def is_running(self) -> bool:
        """Check if server process is running"""
        if self._process is None:
            return False
        return self._process.poll() is None

    def start(self) -> None:
        """
        Start opencode server

        Raises:
            RuntimeError: If server fails to start
        """
        if self.is_running:
            print(f"[OPENCODE SERVER] Already running on {self._server_url}")
            return

        print(f"[OPENCODE SERVER] Starting on port {config.OPENCODE_SERVER_PORT}...")

        cmd = [
            config.OPENCODE_PATH,
            "serve",
            "--port", str(config.OPENCODE_SERVER_PORT),
            "--hostname", config.OPENCODE_SERVER_HOST,
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to be ready (max 30 seconds)
        if not self._wait_for_server(timeout=30):
            self._process.terminate()
            self._process = None
            raise RuntimeError(
                f"OpenCode server failed to start on {self._server_url}. "
                f"Check if opencode is installed: npm install -g opencode-ai@latest"
            )

        print(f"[OPENCODE SERVER] Running on {self._server_url}")

    def _wait_for_server(self, timeout: int) -> bool:
        """Wait for server to be ready"""
        import urllib.request
        import urllib.error

        start = time.time()
        while time.time() - start < timeout:
            try:
                # Try to connect to server
                req = urllib.request.Request(f"{self._server_url}/health")
                with urllib.request.urlopen(req, timeout=2):
                    return True
            except (urllib.error.URLError, ConnectionRefusedError):
                time.sleep(0.5)
        return False

    def stop(self) -> None:
        """Stop opencode server"""
        if self._process is not None:
            print("[OPENCODE SERVER] Stopping...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            print("[OPENCODE SERVER] Stopped")

    def ensure_running(self) -> None:
        """
        Ensure server is running, restart once if needed

        Raises:
            RuntimeError: If server is down and restart fails
        """
        if self.is_running:
            return

        if self._restart_attempted:
            raise RuntimeError(
                "OpenCode server is down and restart already attempted. "
                "Manual intervention required."
            )

        print("[OPENCODE SERVER] Server down, attempting restart...")
        self._restart_attempted = True
        self.start()
        self._restart_attempted = False  # Reset on successful restart


# Global instance
_server_manager: Optional[OpenCodeServerManager] = None


def get_server_manager() -> OpenCodeServerManager:
    """Get the global server manager instance"""
    global _server_manager
    if _server_manager is None:
        _server_manager = OpenCodeServerManager()
    return _server_manager


def start_opencode_server() -> None:
    """Start the opencode server (call on tools_server startup)"""
    from tools.python_coder.opencode_config import ensure_opencode_config

    # Generate config first
    ensure_opencode_config()

    # Start server
    manager = get_server_manager()
    manager.start()


def stop_opencode_server() -> None:
    """Stop the opencode server (call on tools_server shutdown)"""
    global _server_manager
    if _server_manager is not None:
        _server_manager.stop()
        _server_manager = None
```

---

## Part 4: OpenCode Executor (`opencode_tool.py`)

**Path**: `tools/python_coder/opencode_tool.py`

```python
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
        instruction: str,
        timeout: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute natural language instruction using OpenCode

        Args:
            instruction: Natural language instruction (NOT Python code)
            timeout: Execution timeout in seconds
            context: Additional context (unused)

        Returns:
            Standardized execution result dictionary

        Raises:
            RuntimeError: If opencode server is unavailable
        """
        exec_timeout = timeout or self.timeout
        start_time = time.time()

        # Log execution start
        self._log_start(instruction, exec_timeout)

        # Ensure server is running
        self._server.ensure_running()

        # Build command
        cmd = self._build_command(instruction)

        print(f"\n[OPENCODE] Executing: {' '.join(cmd[:6])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=exec_timeout,
                cwd=str(self.workspace),
                encoding='utf-8',
                errors='replace'
            )

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
        cmd = [
            config.OPENCODE_PATH,
            "run",
            instruction,
            "--format", "json",
            "--attach", self._server.server_url,
            "--model", f"{config.OPENCODE_PROVIDER}/{config.OPENCODE_MODEL}",
        ]

        # Continue existing session if available
        if self.session_id in OpenCodeExecutor._session_map:
            opencode_session = OpenCodeExecutor._session_map[self.session_id]
            cmd.extend(["--session", opencode_session])

        return cmd

    def _parse_output(self, stdout: str) -> tuple:
        """
        Parse OpenCode JSON output

        Returns:
            Tuple of (text_output, opencode_session_id, error_message)
        """
        text_parts = []
        session_id = None
        error_msg = None

        for line in stdout.strip().split('\n'):
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Skip non-JSON lines (INFO logs)
                continue

            event_type = event.get("type")
            part = event.get("part", {})

            if event_type == "step_start":
                # Extract session ID
                session_id = event.get("sessionID") or part.get("sessionID")

            elif event_type == "text":
                text = part.get("text", "")
                if text:
                    text_parts.append(text)

            elif event_type == "tool_result":
                content = part.get("content", "")
                if content:
                    text_parts.append(f"[Tool Result]: {content}")

            elif event_type == "error":
                error_msg = part.get("message") or str(part)

        return "\n".join(text_parts), session_id, error_msg

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
```

---

## Part 5: Factory Update (`__init__.py`)

**Path**: `tools/python_coder/__init__.py`

```python
"""
Python Coder Tool - Factory for selecting execution backend
"""
import config
from tools.python_coder.base import BasePythonExecutor


def get_python_executor(session_id: str) -> BasePythonExecutor:
    """
    Factory function to get Python executor based on config

    Args:
        session_id: Session ID for workspace isolation

    Returns:
        Python executor instance

    Raises:
        ValueError: If PYTHON_EXECUTOR_MODE is invalid
    """
    mode = config.PYTHON_EXECUTOR_MODE

    if mode == "native":
        from tools.python_coder.native_tool import NativePythonExecutor
        return NativePythonExecutor(session_id)

    elif mode == "nanocoder":
        from tools.python_coder.nanocoder_tool import NanocoderExecutor
        return NanocoderExecutor(session_id)

    elif mode == "opencode":
        from tools.python_coder.opencode_tool import OpenCodeExecutor
        return OpenCodeExecutor(session_id)

    else:
        raise ValueError(
            f"Invalid PYTHON_EXECUTOR_MODE: '{mode}'. "
            f"Must be 'native', 'nanocoder', or 'opencode'"
        )


class PythonCoderTool:
    """Backward compatibility wrapper"""

    def __new__(cls, session_id: str):
        return get_python_executor(session_id)


__all__ = ["PythonCoderTool", "get_python_executor", "BasePythonExecutor"]
```

---

## Part 6: Tools Server Startup (`tools_server.py`)

Add to `tools_server.py` startup:

```python
# At the top, after imports
import atexit

# In the startup code (after app creation, before uvicorn.run)
if config.PYTHON_EXECUTOR_MODE == "opencode":
    from tools.python_coder.opencode_server import start_opencode_server, stop_opencode_server

    print("[TOOLS SERVER] Starting OpenCode server...")
    start_opencode_server()

    # Register shutdown handler
    atexit.register(stop_opencode_server)
```

---

## Part 7: Prompts

### 7.1 System Prompt

**Path**: `prompts/agents/react_system_opencode.txt`

```
You are a reasoning agent that uses tools to solve problems step-by-step.
Your job is to generate 'Thought, Action, and Action Input' triplet.

You have access to the following tools:
{tools}

Important:
- Keep your Thought focused and THOROUGH
- Use rag only when explicitly asked to search internal documents
- For python_coder: provide detailed natural language instructions
- For websearch and rag: provide a search query

Answer in this EXACT format:

Thought: [Your reasoning about what to do next]
Action: [Tool name to use]
Action Input: [Input for the tool]
```

### 7.2 Thought Prompt

**Path**: `prompts/agents/react_thought_opencode.txt`

```
==============================
{user_query}
==============================

Previous thoughts and actions:
{scratchpad}

----------------------------------------------------

Analyze the current situation and determine your next action.

Output EXACTLY ONE thought/action/action input pair:

Thought: [Your reasoning about what to do next]
Action: [Tool name]
Action Input: [Input for the tool]

RULES:
1. Generate EXACTLY ONE Thought/Action/Action Input triplet
2. Stop immediately after generating ONE action
3. The tool will execute and provide an observation
4. Action Input must be ONLY the input for the tool

For python_coder (OpenCode mode):
- Write clear, detailed natural language instructions
- Describe what the code should do, outputs to produce, files to create
- OpenCode will generate Python code, execute it, and handle errors
- OpenCode has full filesystem access in the workspace
- Example: "Create a function to calculate factorial of 10, test it with several values, and save results to output.txt"

For websearch and rag:
- Provide a pure search query

Output format:

Thought: [Your reasoning]
Action: [Tool name]
Action Input: [Input for the tool]
```

---

## Part 8: Tool Schema Update (`tools_config.py`)

**CRITICAL**: The tool description must tell the LLM what kind of input to provide.

**Path**: `tools_config.py`

Add mode-aware tool description function:

```python
import config

def get_python_coder_description() -> str:
    """Get python_coder description based on executor mode"""
    if config.PYTHON_EXECUTOR_MODE in ["nanocoder", "opencode"]:
        return (
            "Execute code tasks using an AI coding agent. Provide detailed natural language "
            "instructions describing what code to write and execute. The agent will generate "
            "Python code, execute it, handle errors, and iterate as needed. "
            "Example: 'Create a function to calculate factorial of numbers 1-10, test it, "
            "and save the results to a file called results.txt'"
        )
    else:  # native mode
        return (
            "Execute Python code in a sandboxed environment. Provide the complete Python "
            "code to execute. Can access files created in previous executions within the "
            "same session. Use for calculations, data analysis, visualizations, or any "
            "computational task."
        )


def get_python_coder_input_description() -> str:
    """Get python_coder input parameter description based on executor mode"""
    if config.PYTHON_EXECUTOR_MODE in ["nanocoder", "opencode"]:
        return (
            "Detailed natural language instruction describing the code task. "
            "Be specific about: what the code should do, expected outputs, "
            "files to create, and any constraints."
        )
    else:  # native mode
        return "The Python code to execute"
```

Update `TOOL_SCHEMAS` to use dynamic descriptions:

```python
# In TOOL_SCHEMAS, update python_coder entry:
"python_coder": {
    "name": "python_coder",
    "description": get_python_coder_description(),  # Dynamic based on mode
    "endpoint": "/api/tools/python_coder",
    "method": "POST",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": get_python_coder_input_description()  # Dynamic
            },
            # ... rest unchanged
        },
        "required": ["code", "session_id"]
    },
    # ... rest unchanged
}
```

**Alternative approach** (simpler): Since `TOOL_SCHEMAS` is loaded at import time, create a function that builds schemas dynamically:

```python
def get_tool_schemas() -> Dict[str, Dict[str, Any]]:
    """Get tool schemas with mode-aware descriptions"""
    schemas = {
        "websearch": { ... },  # unchanged
        "python_coder": {
            "name": "python_coder",
            "description": get_python_coder_description(),
            "endpoint": "/api/tools/python_coder",
            "method": "POST",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": get_python_coder_input_description()
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for workspace isolation (required)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (optional)",
                        "default": 30
                    }
                },
                "required": ["code", "session_id"]
            },
            "returns": {
                "success": "Boolean indicating if execution succeeded",
                "answer": "Human-readable execution result",
                "data": "Dictionary with stdout, stderr, files, workspace, returncode",
                "metadata": "Execution metadata including timing"
            }
        },
        "rag": { ... },  # unchanged
        "ppt_maker": { ... },  # unchanged
    }
    return schemas

# Replace TOOL_SCHEMAS constant with function call where needed
# Or keep TOOL_SCHEMAS but rebuild at runtime
```

---

## Part 9: ReAct Agent Update (`react_agent.py`)

**Path**: `backend/agents/react_agent.py`

### 9.1 Add Prompt Suffix Method

```python
def _get_prompt_suffix(self) -> str:
    """Get prompt file suffix based on executor mode"""
    mode = config.PYTHON_EXECUTOR_MODE
    if mode == "nanocoder":
        return "_nanocoder"
    elif mode == "opencode":
        return "_opencode"
    return ""  # native mode uses base prompts
```

### 9.2 Update `_step1_generate_action` (lines 181-186)

Replace current prompt selection:

```python
# BEFORE (current code):
if config.PYTHON_EXECUTOR_MODE == "nanocoder":
    system_prompt_file = "agents/react_system_nanocoder.txt"
    thought_prompt_file = "agents/react_thought_nanocoder.txt"
else:  # native mode (default)
    system_prompt_file = "agents/react_system.txt"
    thought_prompt_file = "agents/react_thought.txt"

# AFTER (updated code):
suffix = self._get_prompt_suffix()
system_prompt_file = f"agents/react_system{suffix}.txt"
thought_prompt_file = f"agents/react_thought{suffix}.txt"
```

### 9.3 Update `_convert_string_to_params` (CRITICAL)

**Current behavior**: Always calls `_extract_code_from_fenced_block()` which strips natural language formatting.

**Required behavior**: Only extract code for native mode; pass natural language directly for opencode/nanocoder.

```python
def _convert_string_to_params(self, tool_name: str, string_input: str) -> Dict[str, any]:
    """
    Convert string input to tool parameters (strict)

    Args:
        tool_name: Tool name
        string_input: String input

    Returns:
        Parameters dict

    Raises:
        ValueError: If conversion fails or input invalid
    """
    # Validate input
    if not string_input or not string_input.strip():
        raise ValueError(f"Tool input for '{tool_name}' is empty")

    clean_input = string_input.strip()

    # Convert based on tool
    if tool_name == "websearch":
        return {
            "query": clean_input,
            "max_results": config.WEBSEARCH_MAX_RESULTS
        }

    elif tool_name == "python_coder":
        # Mode-specific input handling
        if config.PYTHON_EXECUTOR_MODE == "native":
            # Native mode: extract Python code from fenced blocks
            code_input = self._extract_code_from_fenced_block(clean_input)
        else:
            # OpenCode/Nanocoder mode: pass natural language instruction directly
            # DO NOT extract from fenced blocks - preserve full instruction
            code_input = clean_input
            print(f"[REACT] OpenCode mode: passing natural language instruction ({len(code_input)} chars)")

        return {
            "code": code_input,
            "session_id": self.session_id or "auto",
            "timeout": config.PYTHON_CODER_TIMEOUT
        }

    elif tool_name == "rag":
        return {
            "query": clean_input,
            "collection_name": config.RAG_DEFAULT_COLLECTION,
            "max_results": config.RAG_MAX_RESULTS
        }

    else:
        raise ValueError(f"Unknown tool: '{tool_name}'")
```

---

## Part 10: Files Summary

### New Files to Create

| File | Purpose |
|------|---------|
| `tools/python_coder/opencode_tool.py` | Main executor class |
| `tools/python_coder/opencode_server.py` | Server lifecycle manager |
| `tools/python_coder/opencode_config.py` | Config generator |
| `prompts/agents/react_system_opencode.txt` | System prompt |
| `prompts/agents/react_thought_opencode.txt` | Thought prompt |

### Files to Modify

| File | Changes |
|------|---------|
| `config.py` | Add `OPENCODE_*` settings, update `PYTHON_EXECUTOR_MODE` type |
| `tools_config.py` | Add mode-aware description functions for python_coder |
| `tools/python_coder/__init__.py` | Add opencode case to factory |
| `tools_server.py` | Start/stop opencode server on startup/shutdown |
| `backend/agents/react_agent.py` | Add prompt suffix logic + fix `_convert_string_to_params` |

---

## Part 11: Implementation Order

1. **config.py** - Add all OPENCODE_* settings
2. **tools_config.py** - Add mode-aware tool descriptions (CRITICAL)
3. **opencode_config.py** - Config generator
4. **opencode_server.py** - Server manager
5. **opencode_tool.py** - Main executor
6. **__init__.py** - Update factory
7. **Prompts** - Create both prompt files
8. **react_agent.py** - Add prompt suffix + fix `_convert_string_to_params` (CRITICAL)
9. **tools_server.py** - Add startup/shutdown hooks

---

## Part 12: Testing Checklist

### Installation
- [ ] OpenCode installed: `npm install -g opencode-ai@latest`
- [ ] OpenCode version check: `opencode --version`

### Configuration
- [ ] Config generates correctly for Ollama backend
- [ ] Config generates correctly for llama.cpp backend
- [ ] Config path is correct on Windows

### Server Lifecycle
- [ ] Server starts on tools_server startup
- [ ] Server is reachable at configured host:port
- [ ] Server auto-restarts once on failure
- [ ] Server fails with clear error on second failure

### Tool Execution (CRITICAL)
- [ ] Simple natural language instruction executes successfully
- [ ] ReAct agent generates **natural language** (NOT Python code)
- [ ] Natural language is passed to opencode **without** code extraction
- [ ] Session continues across multiple tool calls
- [ ] Files are created in correct workspace directory
- [ ] JSON output parses correctly (text, tool_result, error events)
- [ ] Timeout works as expected

### Prompt Verification
- [ ] ReAct uses `react_system_opencode.txt` when mode is "opencode"
- [ ] ReAct uses `react_thought_opencode.txt` when mode is "opencode"
- [ ] Tool description shows natural language instructions (not Python code)

### End-to-End
- [ ] User query → ReAct agent → natural language instruction → opencode → result
- [ ] Multi-step task with multiple tool calls
- [ ] Error handling: opencode failure returns clear error to ReAct agent
