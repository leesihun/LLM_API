"""
Secure-ish Python execution sandbox.

Goals:
- Keep per-session working directories under settings.python_code_execution_dir
- Validate code for obvious unsafe imports/tokens
- Enforce timeouts and capture stdout/stderr
- Optional persistent REPL state per session
"""

from __future__ import annotations

import asyncio
import builtins
import io
import os
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from backend.config.settings import settings
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Conservative import controls
BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "tempfile",
    "fcntl",
    "pty",
    "resource",
}

ALLOWED_IMPORTS = {
    "math",
    "json",
    "re",
    "random",
    "string",
    "itertools",
    "collections",
    "functools",
    "datetime",
    "typing",
    "statistics",
}


@dataclass
class ValidationResult:
    valid: bool
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: Optional[str]
    execution_time_ms: int
    files_written: Optional[list[str]] = None


class CodeValidator:
    """Basic AST-free validator to block obvious dangerous imports."""

    @staticmethod
    def validate(code: str, max_size_kb: int) -> ValidationResult:
        if not code.strip():
            return ValidationResult(False, "Code is empty")

        if len(code.encode("utf-8")) > max_size_kb * 1024:
            return ValidationResult(False, f"Code exceeds {max_size_kb} KB limit")

        lowered = code.lower()
        for banned in BLOCKED_IMPORTS:
            token = f"import {banned}"
            if token in lowered or f"from {banned}" in lowered:
                return ValidationResult(False, f"Blocked import: {banned}")

        return ValidationResult(True)


class CodeSandbox:
    """Manages execution for a single session directory."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.base_dir = Path(settings.python_code_execution_dir) / session_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir = self.base_dir  # compatibility for shell tool
        self.globals: Dict[str, Any] = {} if settings.python_code_use_persistent_repl else None

        # Preload optional libs
        if self.globals is not None:
            self.globals["__builtins__"] = self._safe_builtins()
            for lib in settings.python_code_preload_libraries:
                self._preload(lib)

    def _safe_builtins(self):
        allowed = {
            "abs",
            "min",
            "max",
            "sum",
            "len",
            "range",
            "enumerate",
            "zip",
            "sorted",
            "map",
            "filter",
            "all",
            "any",
            "print",
            "__import__",
        }
        return {k: getattr(builtins, k) for k in allowed if hasattr(builtins, k)}

    def _preload(self, import_stmt: str):
        try:
            exec(f"import {import_stmt}", self.globals)  # noqa: S102
        except Exception as exc:  # pragma: no cover - preload is best effort
            logger.warning(f"[CodeSandbox] Failed to preload {import_stmt}: {exc}")

    async def execute(self, code: str) -> ExecutionResult:
        validation = CodeValidator.validate(code, settings.python_code_max_file_size)
        if not validation.valid:
            return ExecutionResult(
                success=False,
                output="",
                error=validation.error,
                execution_time_ms=0,
            )

        # Dedent to reduce formatting issues
        code = textwrap.dedent(code)

        loop = asyncio.get_event_loop()
        start = time.time()

        try:
            if settings.python_code_allow_partial_execution:
                result = await loop.run_in_executor(None, lambda: self._exec_code_partial(code))
            else:
                result = await loop.run_in_executor(None, lambda: self._exec_code(code))
            elapsed_ms = int((time.time() - start) * 1000)
            return ExecutionResult(
                success=result[0],
                output=result[1],
                error=result[2],
                execution_time_ms=elapsed_ms,
                files_written=self._list_files(),
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            elapsed_ms = int((time.time() - start) * 1000)
            return ExecutionResult(
                success=False,
                output="",
                error=str(exc),
                execution_time_ms=elapsed_ms,
                files_written=self._list_files(),
            )

    def _exec_code(self, code: str) -> Tuple[bool, str, Optional[str]]:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        local_globals = self.globals if self.globals is not None else {"__builtins__": self._safe_builtins()}
        local_locals: Dict[str, Any] = {}

        try:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = stdout_capture, stderr_capture
            exec(code, local_globals, local_locals)  # noqa: S102
            return True, stdout_capture.getvalue(), None
        except Exception:
            tb = traceback.format_exc()
            return False, stdout_capture.getvalue(), tb
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            if self.globals is None:
                self.globals = None  # ensure no leak

    def _exec_code_partial(self, code: str) -> Tuple[bool, str, Optional[str]]:
        """Execute line by line; helpful for debugging. Still not fully safe."""
        output_lines = []
        for line in code.splitlines():
            res = self._exec_code(line)
            if res[2]:
                return res
            output_lines.append(res[1])
        return True, "\n".join(output_lines), None

    def _list_files(self) -> list[str]:
        files = []
        for p in self.base_dir.rglob("*"):
            if p.is_file():
                files.append(str(p.relative_to(self.base_dir)))
        return files


class SandboxManager:
    """Keeps per-session sandboxes."""

    _sandboxes: Dict[str, CodeSandbox] = {}

    @classmethod
    def get_sandbox(cls, session_id: str) -> CodeSandbox:
        if session_id not in cls._sandboxes:
            cls._sandboxes[session_id] = CodeSandbox(session_id)
        return cls._sandboxes[session_id]


# Singleton helper
sandbox_manager = SandboxManager()

