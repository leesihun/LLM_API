"""
Python Coder Tool
=================
Generates and executes Python code to solve tasks.
Coordinates generation, execution (via CodeSandbox), and error handling.
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_core.messages import HumanMessage

from backend.core import BaseTool, ToolResult
from backend.config.settings import settings
from backend.utils.logging_utils import get_logger
from backend.utils.llm_manager import LLMManager
from backend.utils.llm_response_parser import LLMResponseParser
from backend.utils.error_classifier import ErrorClassifier
from backend.tools.code_sandbox import CodeSandbox
logger = get_logger(__name__)


def _format_history(history: Optional[List[dict]]) -> str:
    if not history:
        return ""
    lines = []
    for item in history[-5:]:
        role = item.get("role", "user")
        content = str(item.get("content", ""))[:400]
        if len(str(item.get("content", ""))) > 400:
            content += "... (truncated)"
        lines.append(f"- {role}: {content}")
    return "\n".join(lines)


def _format_dict_block(title: str, data: Optional[dict]) -> str:
    if not data:
        return ""
    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    return f"## {title}\n{pretty}"


def _build_code_generation_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    conversation_history: Optional[List[dict]],
    plan_context: Optional[dict],
    react_context: Optional[dict],
) -> str:
    sections = [
        "You are a meticulous Python engineer. Generate runnable code that solves the task.",
        f"## Query\n{query}",
    ]

    if context:
        sections.append(f"## Additional Context\n{context}")

    history_text = _format_history(conversation_history)
    if history_text:
        sections.append(f"## Recent Conversation\n{history_text}")

    plan_block = _format_dict_block("Plan Context", plan_context)
    if plan_block:
        sections.append(plan_block)

    react_block = _format_dict_block("ReAct Context", react_context)
    if react_block:
        sections.append(react_block)

    sections.append(f"## File Metadata\n{file_context or 'No files supplied.'}")

    sections.append(
        "## Requirements\n"
        "- Use only the files described above (absolute paths already resolved).\n"
        "- Prefer pandas/pyarrow/pillow/etc. as needed; install nothing.\n"
        "- Print key results so the user can read console output.\n"
        "- Never invent filenames or schema."
    )

    # Response format instruction - simplified to avoid escaping issues
    response_format = """## Response Format
CRITICAL: You MUST return valid JSON with exactly one key "code".

Example: {"code": "import pandas as pd\\ndf = pd.read_csv(path)\\nprint(df.head())"}

RULES:
- Return ONLY the JSON object, nothing else
- Do NOT return plain text calculations or results
- Do NOT wrap in markdown code fences
- The code value must be a complete Python script as a string
- Use \\n for newlines within the code string"""
    sections.append(response_format)

    sections.append(
        "## Self-Verification Checklist\n"
        "- Does the code fully answer the query (load files, compute results, save/print outputs)?\n"
        "- Are errors handled (missing columns, empty files, etc.)?\n"
        "- Are all file paths valid and joined with `os.path`?\n"
        "- Are results printed or saved as requested?\n"
        "Fix issues before returning."
    )

    return "\n\n".join(sections)


class PythonCoderTool(BaseTool):
    """
    Tools for generating and executing Python code.
    """

    def __init__(self):
        super().__init__()
        self.llm_manager = LLMManager()
        self.sandbox = CodeSandbox(
            timeout=settings.python_code_timeout,
            execution_base_dir=settings.python_code_execution_dir
        )
        self.max_retries = settings.python_code_max_iterations

        # Create a coder LLM with extended timeout
        from backend.utils.llm_factory import LLMFactory

        logger.info("[PythonCoder] Initializing coder LLM with extended timeout (300s)...")
        try:
            self.llm = LLMFactory.create_coder_llm(
                timeout=300000,  # 5 minutes timeout for code generation
                user_id="python_coder"
            )

            # Verify Ollama connection if using Ollama backend
            if settings.llm_backend == 'ollama':
                if not LLMFactory.check_connection(timeout=10000):
                    logger.warning(
                        "[PythonCoder] Ollama connection check failed. "
                        "Ensure Ollama is running: 'ollama serve'"
                    )
        except Exception as e:
            logger.error(f"[PythonCoder] Failed to initialize LLM: {e}")
            # Continue anyway - will retry on actual use
            self.llm = LLMFactory.create_coder_llm(
                timeout=300000,
                user_id="python_coder"
            )

    def validate_inputs(self, **kwargs) -> bool:
        """Validate Python coder inputs."""
        query = kwargs.get("query", "")
        return bool(query and query.strip())

    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute Python code generation and execution.

        Args:
            query: Task description
            context: Optional context
            **kwargs: Additional parameters (file_paths, session_id, etc.)

        Returns:
            ToolResult with execution results
        """
        self._log_execution_start(query=query)

        if not self.validate_inputs(query=query):
            return self._handle_validation_error("Query cannot be empty", parameter="query")

        try:
            result = await self.execute_code_task(
                query=query,
                context=context,
                **kwargs
            )

            if result.get("success"):
                return ToolResult.success_result(
                    output=result,
                    execution_time=self._elapsed_time()
                )
            else:
                return ToolResult.failure_result(
                    error=result.get("error", "Code execution failed"),
                    error_type="CodeExecutionError",
                    metadata={"attempt_history": result.get("attempt_history", [])},
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
        Execute a code task with retries.
        """
        # Use pre-configured LLM with extended timeout
        # Note: self.llm is already initialized in __init__ with proper timeout
        
        # Prepare files
        input_files = {}
        file_metadata = {}
        if file_paths:
            for fp in file_paths:
                path = Path(fp)
                if path.exists():
                    input_files[str(path)] = path.name
                    file_metadata[path.name] = {"path": str(path), "size": path.stat().st_size}

        # File context string
        file_ctx = "\n".join([f"- {name}: {meta}" for name, meta in file_metadata.items()])

        attempt_history = []
        
        for attempt in range(self.max_retries):
            logger.info(f"[PythonCoder] Attempt {attempt + 1}/{self.max_retries}")
            
            # Generate Code
            prompt = _build_code_generation_prompt(
                query=query,
                context=context,
                file_context=file_ctx,
                conversation_history=conversation_history,
                plan_context=plan_context,
                react_context=react_context
            )

            # Add retry context
            if attempt > 0:
                prev_error = attempt_history[-1]["error"]
                prompt += f"\n\nPREVIOUS ATTEMPT FAILED. Error:\n{prev_error}\n\nFIX THE CODE."

            # Retry LLM invocation with exponential backoff for connection errors
            max_llm_retries = 3
            response = None
            last_error = None

            for llm_retry in range(max_llm_retries):
                try:
                    response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                    break  # Success
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()

                    # Check if it's a connection error
                    if any(keyword in error_msg for keyword in [
                        'connection', 'timeout', 'refused', 'unreachable',
                        'network', 'http', 'socket'
                    ]):
                        wait_time = 2 ** llm_retry  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(
                            f"[PythonCoder] LLM connection error (retry {llm_retry + 1}/{max_llm_retries}): {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        if llm_retry < max_llm_retries - 1:
                            await asyncio.sleep(wait_time)
                    else:
                        # Non-connection error, re-raise immediately
                        raise

            # If all retries failed, raise the last error
            if response is None:
                logger.error(f"[PythonCoder] All LLM connection attempts failed: {last_error}")
                raise Exception(f"Failed to connect to LLM after {max_llm_retries} attempts: {last_error}")
            
            # Parse response
            try:
                parsed = LLMResponseParser.extract_json(response.content)
                if parsed and "code" in parsed:
                    code = parsed.get("code", "")
                else:
                    # Fallback: Try to extract code block from markdown
                    code = LLMResponseParser.extract_code(response.content)

                    # If still no code, check if LLM returned plain text (wrong format)
                    if not code or len(code.strip()) < 10:
                        logger.warning(
                            f"LLM returned invalid format. Expected JSON with 'code' key, got: {response.content[:200]}"
                        )
                        # Try to wrap response in simple print statement as last resort
                        if response.content.strip() and attempt == self.max_retries - 1:
                            logger.info("Creating fallback code from LLM response")
                            code = f"# LLM returned plain text instead of code\nresult = {response.content.strip()}\nprint('Result:', result)"
                        else:
                            continue
            except Exception as e:
                logger.error(f"Parsing failed: {e}")
                logger.debug(f"Response content: {response.content[:500]}")
                continue

            if not code:
                logger.warning("No code generated")
                continue

            # Execute
            result = self.sandbox.execute(
                code, input_files, session_id, 
                stage_name=f"{stage_prefix}_att{attempt}" if stage_prefix else f"att{attempt}"
            )
            
            attempt_history.append({
                "attempt": attempt + 1,
                "code": code,
                "success": result["success"],
                "error": result.get("error"),
                "output": result.get("output")
            })

            if result["success"]:
                return {
                    "success": True,
                    "output": result["output"],
                    "code": code,
                    "execution_time": result["execution_time"],
                    "attempt_history": attempt_history
                }
            
            logger.warning(f"Execution failed: {result.get('error')}")

        return {
            "success": False,
            "error": "Max retries reached",
            "attempt_history": attempt_history
        }

python_coder_tool = PythonCoderTool()

