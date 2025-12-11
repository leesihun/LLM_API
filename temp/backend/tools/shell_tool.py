"""
Shell Tool
==========
Lightweight shell command executor for safe file navigation and inspection.

Provides a constrained set of commands (ls/dir, pwd, cd, cat/head/tail, find, grep, wc, echo)
scoped to the sandbox working directory used by the Python coder. This keeps navigation and
file inspection within the same session state as code execution.

Version: 0.1.0
Created: 2025-12-07
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.config.settings import settings
from backend.core.base_tool import BaseTool
from backend.core.result_types import ToolResult
from backend.tools.code_sandbox import SandboxManager
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Safe commands are intentionally conservative; extend only with review.
UNIX_SAFE_COMMANDS = {
    "ls", "pwd", "cd",
    "cat", "head", "tail",
    "find", "grep", "wc", "echo",
    "tree", "du", "df"
}

WINDOWS_SAFE_COMMANDS = {
    # Built-ins
    "dir", "cd", "type", "echo", "more", "tree",
    # Utilities
    "find", "findstr"
}

WINDOWS_ALIASES = {
    "ls": "dir",
    "pwd": "cd",
}

BLOCKED_TOKENS = [";", "&&", "||", "|", "`", "$(", ">", "<"]
MAX_OUTPUT_CHARS = 4000


@dataclass
class CommandValidation:
    valid: bool
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class ShellTool(BaseTool):
    """Execute whitelisted shell commands inside the sandbox working directory."""

    def __init__(self):
        super().__init__()
        self.session_cwds: Dict[str, Path] = {}

    def validate_inputs(self, **kwargs) -> bool:
        command = kwargs.get("command") or kwargs.get("query")
        return bool(command and str(command).strip())

    async def execute(
        self,
        query: str,
        session_id: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """Execute a safe shell command."""
        self._start_timer()

        if not settings.shell_tool_enabled:
            return self._handle_validation_error("Shell tool is disabled")

        if not self.validate_inputs(query=query):
            return self._handle_validation_error("Command is required", parameter="command")

        command = query.strip()
        session_id = session_id or f"shell_{int(time.time())}"
        sandbox = SandboxManager.get_sandbox(session_id)
        cwd = self._get_session_cwd(session_id, sandbox)
        safe_commands, aliases = self._command_policy()
        command = self._apply_alias(command, aliases)
        validation = self._validate_command(command, safe_commands)

        if not validation.valid:
            return ToolResult.failure_result(
                error=validation.error or "Invalid command",
                error_type="ValidationError",
                execution_time=self._elapsed_time()
            )

        effective_timeout = timeout or settings.shell_tool_timeout

        try:
            completed = await asyncio.to_thread(
                self._run_command,
                command,
                cwd,
                effective_timeout,
            )

            new_cwd = self._update_cwd(session_id, command, cwd, sandbox)
            stdout = self._truncate_output(completed.stdout)
            stderr = self._truncate_output(completed.stderr)

            observation_parts = []
            if stdout:
                observation_parts.append(stdout.strip())
            if stderr:
                observation_parts.append(f"[stderr] {stderr.strip()}")
            if not observation_parts:
                observation_parts.append("(no output)")
            observation_parts.append(f"(cwd: {new_cwd})")

            return ToolResult.success_result(
                output="\n".join(observation_parts),
                metadata={
                    "returncode": completed.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "cwd": str(new_cwd),
                    "command": command,
                },
                execution_time=self._elapsed_time()
            )

        except subprocess.TimeoutExpired:
            return ToolResult.failure_result(
                error=f"Command timed out after {effective_timeout} seconds",
                error_type="TimeoutError",
                metadata={"cwd": str(cwd), "command": command},
                execution_time=self._elapsed_time()
            )
        except Exception as e:
            return self._handle_error(e, "execute")

    def _command_policy(self) -> Tuple[set, Dict[str, str]]:
        """Return the allowed commands and aliases based on platform and settings."""
        if os.name == "nt" and settings.shell_windows_mode:
            return WINDOWS_SAFE_COMMANDS, WINDOWS_ALIASES
        return UNIX_SAFE_COMMANDS, {}

    def _apply_alias(self, command: str, aliases: Dict[str, str]) -> str:
        """Map user-friendly commands (e.g., ls -> dir) when allowed."""
        if not aliases:
            return command
        try:
            parts = shlex.split(command)
        except ValueError:
            return command
        if not parts:
            return command
        base = parts[0].lower()
        if base in aliases:
            parts[0] = aliases[base]
            # Reconstruct command safely; fall back to join on error
            try:
                command = shlex.join(parts)  # py3.8+; safe quoting
            except AttributeError:
                command = " ".join(parts)
        return command

    def _validate_command(self, command: str, safe_commands: set) -> CommandValidation:
        """Ensure the command is in the whitelist and free of dangerous tokens."""
        for token in BLOCKED_TOKENS:
            if token in command:
                return CommandValidation(False, f"Blocked token detected: '{token}'")

        try:
            parts = shlex.split(command)
        except ValueError as e:
            return CommandValidation(False, f"Could not parse command: {e}")

        if not parts:
            return CommandValidation(False, "Empty command after parsing")

        base = parts[0].lower()
        if base not in safe_commands:
            return CommandValidation(False, f"Command '{base}' is not allowed")

        return CommandValidation(True)

    def _get_session_cwd(self, session_id: str, sandbox) -> Path:
        """Return the current working directory for the session."""
        if session_id not in self.session_cwds:
            self.session_cwds[session_id] = sandbox.working_dir.resolve()
        return self.session_cwds[session_id]

    def _update_cwd(self, session_id: str, command: str, cwd: Path, sandbox) -> Path:
        """Handle `cd` commands while keeping paths inside the sandbox."""
        try:
            parts = shlex.split(command)
        except ValueError:
            return cwd

        if parts and parts[0].lower() == "cd":
            target = parts[1] if len(parts) > 1 else "."
            target_path = Path(target)
            new_path = (cwd / target_path).resolve() if not target_path.is_absolute() else target_path.resolve()

            base = sandbox.working_dir.resolve()
            if not (base == new_path or base in new_path.parents):
                raise PermissionError("Cannot change directory outside the sandbox working directory")

            new_path.mkdir(parents=True, exist_ok=True)
            self.session_cwds[session_id] = new_path
            return new_path

        return cwd

    def _run_command(self, command: str, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
        """Execute the command in a subprocess."""
        return subprocess.run(
            command,
            shell=True,  # Needed for built-ins like `dir` on Windows
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _truncate_output(self, text: Optional[str]) -> str:
        """Cap output length to prevent runaway responses."""
        if not text:
            return ""
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        excess = len(text) - MAX_OUTPUT_CHARS
        return f"{text[:MAX_OUTPUT_CHARS]}\n... (truncated, {excess} more chars)"


# Singleton for ease of import
shell_tool = ShellTool()

