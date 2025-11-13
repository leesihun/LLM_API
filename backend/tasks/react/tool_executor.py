"""
Tool Executor Module for ReAct Agent
Handles execution and routing of tool actions (web_search, rag_retrieval, python_coder, file_analyzer)
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_ollama import ChatOllama

from backend.tools.web_search import web_search_tool
from backend.tools.rag_retriever import rag_retriever
from backend.tools.python_coder import python_coder_tool
from backend.tools.file_analyzer_tool import file_analyzer
from backend.utils.logging_utils import get_logger
from backend.tasks.react.models import ToolName

logger = get_logger(__name__)


class ToolExecutor:
    """
    Executes tools for ReAct agent with intelligent routing and guard logic.

    Features:
    - Routes actions to appropriate tools (web_search, rag_retrieval, python_coder, file_analyzer)
    - GUARD LOGIC: If RAG requested but files present, tries python_coder first
    - Builds context from execution history for python_coder
    - Handles all tool-specific formatting and error handling
    """

    def __init__(self, llm: ChatOllama):
        """
        Initialize ToolExecutor

        Args:
            llm: ChatOllama instance for LLM operations
        """
        self.llm = llm

    async def execute(
        self,
        action: str,
        action_input: str,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        attempted_coder: bool = False,
        steps: Optional[List[Any]] = None
    ) -> tuple[str, bool]:
        """
        Execute the selected action and return observation

        Args:
            action: Tool name to execute (from ToolName enum)
            action_input: Input for the tool
            file_paths: Optional list of file paths for code execution
            session_id: Optional session ID for code execution
            attempted_coder: Whether python_coder has been attempted (for guard logic)
            steps: Optional list of ReActStep objects for context building

        Returns:
            Tuple of (observation, attempted_coder_updated)
            - observation: Result from tool execution
            - attempted_coder_updated: Updated value of attempted_coder flag
        """
        try:
            logger.info("")
            logger.info(f"EXECUTING TOOL: {action}")
            logger.info("")
            logger.info("Tool Input:")
            for _line in str(action_input).splitlines():
                logger.info(_line)
            logger.info("")

            if action == ToolName.WEB_SEARCH:
                observation = await self._execute_web_search(action_input)
                return observation, attempted_coder

            elif action == ToolName.RAG_RETRIEVAL:
                observation, attempted_coder = await self._execute_rag_retrieval(
                    action_input,
                    file_paths,
                    session_id,
                    attempted_coder,
                    steps
                )
                return observation, attempted_coder

            elif action == ToolName.PYTHON_CODER:
                observation = await self._execute_python_coder(
                    action_input,
                    file_paths,
                    session_id,
                    steps
                )
                return observation, True  # Mark attempted_coder as True

            elif action == ToolName.FILE_ANALYZER:
                observation = await self._execute_file_analyzer(action_input, file_paths)
                return observation, attempted_coder

            else:
                logger.warning(f"\n[ToolExecutor] Invalid action attempted: {action}\n")
                return "Invalid action.", attempted_coder

        except Exception as e:
            logger.error("")
            logger.error(f"ERROR EXECUTING ACTION: {action}")
            logger.error(f"Exception Type: {type(e).__name__}")
            logger.error(f"Exception Message: {str(e)}")
            import traceback
            logger.error("Traceback:")
            for _line in traceback.format_exc().splitlines():
                logger.error(_line)
            return f"Error executing action: {str(e)}", attempted_coder

    async def _execute_web_search(self, query: str) -> str:
        """Execute web search tool"""
        # Perform search with temporal context
        results, context_metadata = await web_search_tool.search(
            query,
            max_results=5,
            include_context=True,
            user_location=None  # Can be extended to accept user location from session
        )
        observation = web_search_tool.format_results(results)

        # Include context metadata in observation
        if context_metadata.get('query_enhanced'):
            observation = f"[Search performed with contextual enhancement]\n{observation}"

        final_observation = observation if observation else "No web search results found."

        logger.info("")
        logger.info("TOOL OUTPUT (Web Search):")
        logger.info(f"Context: {context_metadata.get('current_datetime', 'N/A')}")
        if context_metadata.get('enhanced_query'):
            logger.info(f"Enhanced query: {context_metadata['enhanced_query']}")
        for _line in str(final_observation).splitlines():
            logger.info(_line)
        logger.info("")

        return final_observation

    async def _execute_rag_retrieval(
        self,
        query: str,
        file_paths: Optional[List[str]],
        session_id: Optional[str],
        attempted_coder: bool,
        steps: Optional[List[Any]]
    ) -> tuple[str, bool]:
        """
        Execute RAG retrieval tool with GUARD LOGIC

        GUARD LOGIC: If files exist and python_coder not yet attempted, try it first
        """
        # Guarded fallback: if files exist and python_coder not yet attempted, try it first
        if file_paths and not attempted_coder:
            logger.info("[ToolExecutor] Guard: Files present and coder not attempted; trying python_coder before RAG")
            try:
                attempted_coder = True
                # Build context from current execution history
                context = self._build_context_for_python_coder(steps, session_id)

                coder_result = await python_coder_tool.execute_code_task(
                    query=query,
                    file_paths=file_paths,
                    session_id=session_id,
                    context=context
                )
                if coder_result.get("success") and str(coder_result.get("output", "")).strip():
                    final_observation = f"Code executed successfully:\n{coder_result.get('output','')}"
                    logger.info("")
                    logger.info("TOOL OUTPUT (Python Coder via Guard):")
                    for _line in str(final_observation).splitlines():
                        logger.info(_line)
                    logger.info("")
                    return final_observation, attempted_coder
                else:
                    logger.info("[ToolExecutor] Guard coder attempt insufficient; proceeding with RAG")
            except Exception as e:
                logger.warning(f"[ToolExecutor] Guard coder attempt failed: {e}; proceeding with RAG")

        results = await rag_retriever.retrieve(query, top_k=5)
        observation = rag_retriever.format_results(results)
        final_observation = observation if observation else "No relevant documents found."

        logger.info("")
        logger.info("TOOL OUTPUT (RAG Retrieval):")
        for _line in str(final_observation).splitlines():
            logger.info(_line)
        logger.info("")

        return final_observation, attempted_coder

    async def _execute_python_coder(
        self,
        query: str,
        file_paths: Optional[List[str]],
        session_id: Optional[str],
        steps: Optional[List[Any]]
    ) -> str:
        """Execute python_coder tool"""
        # Build context from current execution history
        context = self._build_context_for_python_coder(steps, session_id)

        # Pass attached file paths, session_id, execution context, and step number to python_coder
        # Use ReAct step number for stage prefix (e.g., "step1", "step2")
        current_step_num = len(steps) + 1 if steps else 1  # Current iteration number
        stage_prefix = f"step{current_step_num}"

        result = await python_coder_tool.execute_code_task(
            query=query,
            file_paths=file_paths,
            session_id=session_id,
            context=context,
            stage_prefix=stage_prefix
        )

        logger.info("")
        logger.info("TOOL OUTPUT (Python Coder - Detailed):")
        logger.info(f"Success: {result['success']}")
        logger.info(f"Verification Iterations: {result.get('verification_iterations', 'N/A')}")
        logger.info(f"Execution Attempts: {result.get('execution_attempts', 'N/A')}")
        logger.info(f"Execution Time: {result.get('execution_time', 'N/A'):.2f}s" if isinstance(result.get('execution_time'), (int, float)) else f"Execution Time: {result.get('execution_time', 'N/A')}")
        if file_paths:
            logger.info(f"Files Used: {len(file_paths)} files")
        logger.info("")

        if result["success"]:
            logger.info("Generated Code:")
            for _line in str(result.get('code', 'N/A')).splitlines():
                logger.info(_line)
            logger.info("")
            logger.info("Execution Output:")
            for _line in str(result['output']).splitlines():
                logger.info(_line)
            logger.info("")
            if result.get('verification_issues'):
                logger.info("Verification Issues:")
                for _line in str(result['verification_issues']).splitlines():
                    logger.info(_line)
                logger.info("")

            # Build execution details string safely
            exec_details_parts = []
            if result.get('verification_iterations'):
                exec_details_parts.append(f"{result['verification_iterations']} verification iterations")
            if result.get('execution_attempts'):
                exec_details_parts.append(f"{result['execution_attempts']} execution attempts")
            if isinstance(result.get('execution_time'), (int, float)):
                exec_details_parts.append(f"{result['execution_time']:.2f}s")

            exec_details = ", ".join(exec_details_parts) if exec_details_parts else "completed"
            final_observation = f"Code executed successfully:\n{result['output']}\n\nExecution details: {exec_details}"
        else:
            logger.info("Error Details:")
            for _line in str(result.get('error', 'Unknown error')).splitlines():
                logger.info(_line)
            logger.info("")
            if result.get('code'):
                logger.info("Failed Code:")
                for _line in str(result['code']).splitlines():
                    logger.info(_line)
                logger.info("")

            final_observation = f"Code execution failed: {result.get('error', 'Unknown error')}"

        logger.info("")
        return final_observation

    async def _execute_file_analyzer(
        self,
        query: str,
        file_paths: Optional[List[str]]
    ) -> str:
        """Execute file_analyzer tool"""
        # FILE_ANALYZER should only be used in pre-step, but handle it just in case
        logger.warning("[ToolExecutor] FILE_ANALYZER called during ReAct loop (should only be in pre-step)")

        if not file_paths:
            return "No files attached to analyze."

        result = file_analyzer.analyze(
            file_paths=file_paths,
            user_query=query
        )

        logger.info("")
        logger.info("TOOL OUTPUT (File Analyzer):")
        logger.info(f"Success: {result.get('success', False)}")
        logger.info(f"Files Analyzed: {result.get('files_analyzed', 0)}")
        logger.info("")

        if result.get("success"):
            logger.info("Analysis Summary:")
            for _line in str(result.get('summary', '')).splitlines():
                logger.info(_line)
            logger.info("")
            final_observation = f"File analysis completed:\n{result.get('summary','')}"
        else:
            logger.info("Error Details:")
            for _line in str(result.get('error', 'Unknown error')).splitlines():
                logger.info(_line)
            logger.info("")
            final_observation = f"File analysis failed: {result.get('error','Unknown error')}"

        logger.info("")
        return final_observation

    def _build_context_for_python_coder(
        self,
        steps: Optional[List[Any]],
        session_id: Optional[str]
    ) -> str:
        """
        Build context string from ReAct execution history for python_coder

        Args:
            steps: List of ReActStep objects from execution history
            session_id: Session ID for retrieving code history

        Returns:
            Formatted context string with previous steps and observations
        """
        context_parts = []

        # Load previous code history if available
        code_history = python_coder_tool.get_previous_code_history(session_id, max_versions=3)
        if code_history:
            context_parts.append("=== Previous Code Versions ===\n")
            for idx, code_entry in enumerate(code_history, 1):
                stage_name = code_entry.get('stage_name', 'unknown')
                code = code_entry.get('code', '')

                # Show code preview (first 20 lines)
                code_lines = code.split('\n')
                code_preview = '\n'.join(code_lines[:20])
                if len(code_lines) > 20:
                    code_preview += f"\n... ({len(code_lines) - 20} more lines)"

                context_parts.append(f"Version {idx} ({stage_name}):")
                context_parts.append("```python")
                context_parts.append(code_preview)
                context_parts.append("```")
                context_parts.append("")

            context_parts.append("You can reference and build upon these previous code versions.")
            context_parts.append("Use them as a starting point to avoid repeating work.\n")

        if not steps:
            return "\n".join(context_parts) if context_parts else ""

        context_parts.append("=== Previous Agent Activity ===\n")

        # Include recent steps (last 3 steps or all if less than 3)
        recent_steps = steps[-3:] if len(steps) > 3 else steps

        for step in recent_steps:
            context_parts.append(f"Step {step.step_num}:")
            context_parts.append(f"  Thought: {step.thought[:300]}")
            context_parts.append(f"  Action: {step.action}")

            # Include observation summary
            obs_preview = step.observation[:500] if len(step.observation) > 500 else step.observation
            context_parts.append(f"  Result: {obs_preview}")

            # Highlight if there were errors
            if "error" in step.observation.lower() or "failed" in step.observation.lower():
                context_parts.append(f"  ⚠ Note: This action encountered errors")

            context_parts.append("")

        # Add summary of what's been tried
        tools_tried = [step.action for step in steps if step.action != ToolName.FINISH]
        if tools_tried:
            context_parts.append(f"Tools already attempted: {', '.join(set(tools_tried))}")

        context_parts.append("\nThis context shows what has already been tried. Use this information to:")
        context_parts.append("- Avoid repeating failed approaches")
        context_parts.append("- Build upon partial results from previous steps")
        context_parts.append("- Generate more targeted code based on what's already known")
        if code_history:
            context_parts.append("- Reference the previous code versions shown above")

        return "\n".join(context_parts)
