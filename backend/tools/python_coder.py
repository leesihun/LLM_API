"""
Python Coder Tool
=================
Generates and executes Python code to solve tasks autonomously.
- Accepts task description as input
- Generates Python code using LLM
- Executes code in sandboxed environment
- Evaluates outcomes and fixes errors with retry logic

Version: 3.0.0 - LLM-based code generation with retry and error fixing
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.messages import HumanMessage

from backend.core import BaseTool, ToolResult
from backend.config.settings import settings
from backend.utils.logging_utils import get_logger
from backend.utils.llm_response_parser import LLMResponseParser
from backend.tools.code_sandbox import CodeSandbox
from backend.runtime import extract_original_filename

logger = get_logger(__name__)


def _format_history(history: Optional[List[dict]]) -> str:
    """Format conversation history for context."""
    if not history:
        return "No previous conversation."

    lines = []
    for item in history[-5:]:  # Last 5 messages
        role = item.get("role", "user")
        content = str(item.get("content", ""))
        # Truncate long messages
        if len(content) > 400:
            content = content[:400] + "... (truncated)"
        lines.append(f"- {role}: {content}")

    return "\n".join(lines)


def _format_dict_block(title: str, data: Optional[dict]) -> str:
    """Format dictionary as JSON block with title."""
    if not data:
        return ""

    try:
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        return f"## {title}\n{pretty}"
    except Exception as e:
        logger.warning(f"Failed to format dict block: {e}")
        return f"## {title}\n{str(data)}"


def _build_code_generation_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    conversation_history: Optional[List[dict]],
    plan_context: Optional[dict],
    react_context: Optional[dict],
    previous_error: Optional[str] = None,
) -> str:
    """Build comprehensive prompt for code generation."""

    sections = [
        "You are a meticulous Python engineer. Generate runnable, production-quality code that solves the task.",
        "",
        f"## Task Description\n{query}",
    ]

    # Additional context from ReAct agent
    if context:
        sections.append(f"## Additional Context\n{context}")

    # Recent conversation history
    history_text = _format_history(conversation_history)
    sections.append(f"## Recent Conversation\n{history_text}")

    # Plan context (if executing within a plan)
    plan_block = _format_dict_block("Plan Context", plan_context)
    if plan_block:
        sections.append(plan_block)

    # ReAct context (previous attempts, observations)
    react_block = _format_dict_block("ReAct Context", react_context)
    if react_block:
        sections.append(react_block)

    # File information
    sections.append(f"## Available Files\n{file_context or 'No files provided.'}")

    # Previous error (for retry attempts)
    if previous_error:
        sections.append(
            f"## PREVIOUS ATTEMPT FAILED\n"
            f"The last code execution failed with the following error:\n"
            f"```\n{previous_error}\n```\n"
            f"**IMPORTANT**: Analyze this error carefully and fix the code. "
            f"Do NOT repeat the same mistake."
        )

    # Requirements
    sections.append(
        "## Requirements\n"
        "- Use only the files described above (paths are already absolute)\n"
        "- Prefer standard libraries: pandas, numpy, matplotlib, pillow, etc.\n"
        "- DO NOT attempt to install packages (pip install)\n"
        "- Print or display key results so output is visible\n"
        "- Handle edge cases: missing columns, empty files, wrong formats\n"
        "- Save results to files if requested (CSV, images, etc.)\n"
        "- Use descriptive variable names and add comments for clarity"
    )

    # Response format
    sections.append(
        "## Response Format\n"
        "Return ONLY a valid JSON object with exactly one key `code`:\n"
        "```json\n"
        "{\n"
        '  "code": "import pandas as pd\\n\\ndf = pd.read_csv(\'file.csv\')\\nprint(df.head())"\n'
        "}\n"
        "```\n"
        "**CRITICAL**: Do NOT wrap the code value in markdown code fences (no ```python). "
        "Use proper JSON string escaping (\\n for newlines, \\\" for quotes)."
    )

    # Self-verification checklist
    sections.append(
        "## Self-Verification Checklist (check before responding)\n"
        "- [ ] Does the code fully answer the task description?\n"
        "- [ ] Are all required files loaded correctly?\n"
        "- [ ] Are computations accurate and complete?\n"
        "- [ ] Are results printed or saved as requested?\n"
        "- [ ] Are errors and edge cases handled?\n"
        "- [ ] Is the response valid JSON with escaped code string?\n"
        "- [ ] If this is a retry, did I fix the previous error?"
    )

    return "\n\n".join(sections)


class PythonCoderTool(BaseTool):
    """
    Autonomous Python code generation and execution tool.

    Accepts natural language task descriptions, generates code using LLM,
    executes in sandbox, and retries with error fixing if needed.
    """

    def __init__(self):
        super().__init__()
        self.sandbox = CodeSandbox(
            timeout=settings.python_code_timeout,
            execution_base_dir=settings.python_code_execution_dir
        )
        self.max_retries = settings.python_code_max_iterations
        logger.info(f"[PythonCoder] Initialized with max_retries={self.max_retries}")

    def validate_inputs(self, **kwargs) -> bool:
        """Validate that query/description is provided."""
        query = kwargs.get("query", "")
        return bool(query and query.strip())

    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        stage_prefix: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
        plan_context: Optional[dict] = None,
        react_context: Optional[dict] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute a code generation and execution task.

        Args:
            query: Natural language description of the task
            context: Additional context from the agent
            file_paths: List of file paths to make available
            session_id: Session ID for execution isolation
            stage_prefix: Optional prefix for script naming
            conversation_history: Recent conversation messages
            plan_context: Context from plan executor (if in plan mode)
            react_context: Context from ReAct agent (previous steps)

        Returns:
            ToolResult with execution results
        """
        self._log_execution_start(query=query[:100] + "..." if len(query) > 100 else query)

        if not self.validate_inputs(query=query):
            return self._handle_validation_error("Task description cannot be empty", parameter="query")

        try:
            result = await self.execute_code_task(
                query=query,
                context=context,
                file_paths=file_paths,
                session_id=session_id,
                stage_prefix=stage_prefix,
                conversation_history=conversation_history,
                plan_context=plan_context,
                react_context=react_context
            )

            if result.get("success"):
                return ToolResult.success_result(
                    output=result,
                    execution_time=self._elapsed_time()
                )
            else:
                return ToolResult.failure_result(
                    error=result.get("error", "Code generation/execution failed"),
                    error_type="CodeExecutionError",
                    metadata={
                        "output": result.get("output", ""),
                        "attempt_history": result.get("attempt_history", [])
                    },
                    execution_time=self._elapsed_time()
                )

        except Exception as e:
            return self._handle_error(e, "execute")

    async def execute_code_task(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        stage_prefix: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
        plan_context: Optional[dict] = None,
        react_context: Optional[dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute code task with LLM-based generation and retry logic.

        Returns:
            Dict with success, output, code, execution_time, attempt_history
        """
        # Lazy import to avoid circular dependency
        from backend.utils.llm_factory import LLMFactory

        # Get coder LLM (optimized for code generation)
        llm = LLMFactory.create_coder_llm()

        # Prepare input files - map temp path to original filename
        input_files = {}
        file_metadata = {}

        if file_paths:
            for fp in file_paths:
                path = Path(fp)
                if path.exists():
                    # Extract original filename from temp path
                    original_name = extract_original_filename(str(path))
                    input_files[str(path)] = original_name

                    # Gather metadata for context
                    file_metadata[original_name] = {
                        "path": str(path),
                        "size": path.stat().st_size,
                        "extension": path.suffix
                    }
                    logger.debug(f"[PythonCoder] Added file: {path.name} -> {original_name}")

        # Format file context for prompt
        file_context_lines = []
        for name, meta in file_metadata.items():
            file_context_lines.append(
                f"- **{name}**: {meta['size']} bytes, type: {meta['extension']}, "
                f"path: `{meta['path']}`"
            )
        file_context = "\n".join(file_context_lines) if file_context_lines else "No files provided."

        # Track all attempts
        attempt_history = []
        previous_error = None

        # Retry loop
        for attempt in range(self.max_retries):
            logger.info(f"[PythonCoder] Attempt {attempt + 1}/{self.max_retries}")

            # Build prompt
            prompt = _build_code_generation_prompt(
                query=query,
                context=context,
                file_context=file_context,
                conversation_history=conversation_history,
                plan_context=plan_context,
                react_context=react_context,
                previous_error=previous_error
            )

            # Generate code using LLM
            try:
                logger.info(f"[PythonCoder] Invoking LLM for code generation...")
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                response_text = response.content if hasattr(response, 'content') else str(response)

                logger.debug(f"[PythonCoder] LLM response length: {len(response_text)} chars")

                # Parse response - expect JSON with "code" key
                parsed = LLMResponseParser.extract_json(response_text)

                if not parsed or "code" not in parsed:
                    # Fallback: try to extract code block
                    logger.warning("[PythonCoder] JSON parsing failed, trying code block extraction")
                    code = LLMResponseParser.extract_code(response_text)
                else:
                    code = parsed.get("code", "")

                if not code or not code.strip():
                    logger.warning(f"[PythonCoder] No code generated on attempt {attempt + 1}")
                    previous_error = "No code was generated in the response"
                    attempt_history.append({
                        "attempt": attempt + 1,
                        "code": None,
                        "success": False,
                        "error": "No code generated",
                        "output": ""
                    })
                    continue

            except Exception as e:
                logger.error(f"[PythonCoder] Code generation failed: {e}")
                previous_error = f"Code generation error: {str(e)}"
                attempt_history.append({
                    "attempt": attempt + 1,
                    "code": None,
                    "success": False,
                    "error": str(e),
                    "output": ""
                })
                continue

            # Execute generated code in sandbox
            logger.info(f"[PythonCoder] Executing generated code ({len(code)} chars)")

            stage_name = f"{stage_prefix}_att{attempt + 1}" if stage_prefix else f"att{attempt + 1}"

            exec_result = self.sandbox.execute(
                code=code,
                input_files=input_files,
                session_id=session_id,
                stage_name=stage_name
            )

            # Record attempt
            attempt_history.append({
                "attempt": attempt + 1,
                "code": code,
                "success": exec_result["success"],
                "error": exec_result.get("error"),
                "output": exec_result.get("output", ""),
                "execution_time": exec_result.get("execution_time", 0)
            })

            # Check if execution succeeded
            if exec_result["success"]:
                logger.info(
                    f"[PythonCoder] ✓ Success on attempt {attempt + 1}! "
                    f"Time: {exec_result.get('execution_time', 0):.2f}s"
                )

                return {
                    "success": True,
                    "output": exec_result["output"],
                    "code": code,
                    "execution_time": exec_result.get("execution_time", 0),
                    "created_files": exec_result.get("created_files", []),
                    "attempts": attempt + 1,
                    "attempt_history": attempt_history
                }
            else:
                # Execution failed - prepare for retry
                error_msg = exec_result.get("error", "Unknown execution error")
                logger.warning(f"[PythonCoder] ✗ Attempt {attempt + 1} failed: {error_msg}")
                previous_error = error_msg

        # All retries exhausted
        logger.error(f"[PythonCoder] All {self.max_retries} attempts failed")

        # Get the last error for reporting
        last_attempt = attempt_history[-1] if attempt_history else {}

        return {
            "success": False,
            "error": f"Failed after {self.max_retries} attempts. Last error: {last_attempt.get('error', 'Unknown')}",
            "output": last_attempt.get("output", ""),
            "code": last_attempt.get("code", ""),
            "execution_time": sum(a.get("execution_time", 0) for a in attempt_history),
            "attempts": self.max_retries,
            "attempt_history": attempt_history
        }


# Global instance
python_coder_tool = PythonCoderTool()
