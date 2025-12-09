"""
Python coder tool: generate and optionally execute Python code inside the sandbox.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional

from backend.config.settings import settings
from backend.config.prompts import PYTHON_CODER_PROMPT
from backend.core.base_tool import BaseTool
from backend.core.result_types import ToolResult
from backend.tools.code_sandbox import SandboxManager


class PythonCoderTool(BaseTool):
    """LLM-powered Python code generator + sandbox executor."""

    def validate_inputs(self, **kwargs) -> bool:
        query = kwargs.get("query")
        return bool(query and str(query).strip())

    async def execute(
        self,
        query: str,
        session_id: Optional[str] = None,
        run: bool = True,
        **kwargs,
    ) -> ToolResult:
        self._start_timer()

        if not settings.python_code_enabled:
            return self._handle_validation_error("Python code execution is disabled")

        if not self.validate_inputs(query=query):
            return self._handle_validation_error("Query is required", parameter="query")

        llm = self._get_coder_llm(user_id=kwargs.get("user_id", "default"))

        prompt = PYTHON_CODER_PROMPT.format(task=query)

        code_text = await self._generate_code(llm, prompt)
        code = self._extract_code(code_text)

        if not code:
            return ToolResult.failure_result(
                error="LLM did not return code",
                error_type="NoCodeGenerated",
                execution_time=self._elapsed_time(),
            )

        if not run:
            return ToolResult.success_result(
                output=code,
                metadata={"generated_code": code},
                execution_time=self._elapsed_time(),
            )

        sandbox = SandboxManager.get_sandbox(session_id or "default")
        exec_result = await sandbox.execute(code)

        if exec_result.success:
            output = exec_result.output.strip()
            return ToolResult.success_result(
                output=output or "(no output)",
                metadata={
                    "generated_code": code,
                    "files_written": exec_result.files_written,
                },
                execution_time=self._elapsed_time(),
            )

        return ToolResult.failure_result(
            error=exec_result.error or "Execution failed",
            error_type="ExecutionError",
            metadata={
                "generated_code": code,
                "stdout": exec_result.output,
            },
            execution_time=self._elapsed_time(),
        )

    async def _generate_code(self, llm, prompt: str) -> str:
        if hasattr(llm, "ainvoke"):
            result = await llm.ainvoke(prompt)
            return getattr(result, "content", str(result))
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
        return getattr(result, "content", str(result))

    def _extract_code(self, text: str) -> str:
        match = re.search(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            body = match.group(1)
            return body.replace("```", "").strip()
        # fallback: any fenced block
        match = re.search(r"```(.*?```)", text, re.DOTALL)
        if match:
            body = match.group(1)
            return body.replace("```", "").strip()
        return text.strip()


# Singleton instance
python_coder_tool = PythonCoderTool()

