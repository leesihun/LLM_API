"""
Tool Executor Module for ReAct Agent

Handles execution and routing of tool actions with intelligent guard logic.
Uses llm_factory for LLM instances and tool_metadata models from Phase 2.
"""

from typing import List, Optional, Tuple, Any

from backend.tools.web_search import web_search_tool
from backend.tools.rag_retriever import rag_retriever_tool
from backend.tools.python_coder import python_coder_tool
from backend.tools.file_analyzer import file_analyzer
from backend.utils.logging_utils import get_logger
from .models import ToolName

logger = get_logger(__name__)


class ToolExecutor:
    """
    Executes tools for ReAct agent with direct routing.

    Features:
    - Routes actions to appropriate tools
    - Direct execution: Each tool executes when requested (no guard logic)
    - Handles all tool-specific formatting and error handling
    """

    def __init__(self, llm):
        """
        Initialize ToolExecutor.

        Args:
            llm: LLM instance for LLM operations
        """
        self.llm = llm

    async def execute(
        self,
        action: str,
        action_input: str,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        steps: Optional[List[Any]] = None,
        plan_context: Optional[dict] = None
    ) -> str:
        """
        Execute the selected action and return observation.

        Args:
            action: Tool name to execute (from ToolName enum)
            action_input: Input for the tool
            file_paths: Optional list of file paths for code execution
            session_id: Optional session ID for code execution
            steps: Optional list of ReActStep objects for context building
            plan_context: Optional plan context from Plan-Execute agent

        Returns:
            Observation string from tool execution
        """
        try:
            logger.info(f"EXECUTING TOOL: {action}")
            logger.info(f"Tool Input: {action_input[:200]}...")

            if action == ToolName.WEB_SEARCH:
                return await self._execute_web_search(action_input)

            elif action == ToolName.RAG_RETRIEVAL:
                return await self._execute_rag_retrieval(action_input)

            elif action == ToolName.PYTHON_CODER:
                return await self._execute_python_coder(
                    action_input, file_paths, session_id, steps, plan_context
                )

            elif action == ToolName.FILE_ANALYZER:
                return await self._execute_file_analyzer(action_input, file_paths)

            else:
                logger.warning(f"Invalid action: {action}")
                return "Invalid action."

        except Exception as e:
            logger.error(f"ERROR EXECUTING ACTION: {action}")
            logger.error(f"Exception: {type(e).__name__}: {str(e)}")
            return f"Error executing action: {str(e)}"

    async def _execute_web_search(self, query: str) -> str:
        """Execute web search tool."""
        results, context_metadata = await web_search_tool.search(
            query,
            max_results=5,
            include_context=True,
            user_location=None
        )
        observation = web_search_tool.format_results(results)

        # Include context metadata
        if context_metadata.get('query_enhanced'):
            observation = f"[Search performed with contextual enhancement]\n{observation}"

        final_observation = observation if observation else "No web search results found."

        logger.info(f"Web Search Results: {len(results)} results")
        if context_metadata.get('enhanced_query'):
            logger.info(f"Enhanced query: {context_metadata['enhanced_query']}")

        return final_observation

    async def _execute_rag_retrieval(self, query: str) -> str:
        """
        Execute RAG retrieval tool.

        Args:
            query: Search query for document retrieval

        Returns:
            Formatted observation with retrieved documents
        """
        results = await rag_retriever_tool.retrieve(query, top_k=5)
        observation = rag_retriever_tool.format_results(results)
        final_observation = observation if observation else "No relevant documents found."

        logger.info(f"RAG Retrieval: {len(results)} documents")

        return final_observation

    async def _execute_python_coder(
        self,
        query: str,
        file_paths: Optional[List[str]],
        session_id: Optional[str],
        steps: Optional[List[Any]],
        plan_context: Optional[dict] = None
    ) -> str:
        """Execute python_coder tool."""
        # Build context from execution history
        context = self._build_context_for_python_coder(steps, session_id)

        # Build react_context with failed attempts (including saved code)
        react_context = self._build_react_context(steps, session_id)

        # Load conversation_history from conversation store
        conversation_history = self._load_conversation_history(session_id)

        # Use ReAct step number for hierarchical prompt organization
        current_step_num = len(steps) + 1 if steps else 1
        stage_prefix = f"step{current_step_num}"

        # Extract plan_step if available from plan_context
        plan_step_num = None
        if plan_context and 'current_step' in plan_context:
            plan_step_num = plan_context['current_step']

        result = await python_coder_tool.execute_code_task(
            query=query,
            file_paths=file_paths,
            session_id=session_id,
            context=context,
            stage_prefix=stage_prefix,
            react_context=react_context,
            plan_context=plan_context,
            conversation_history=conversation_history,
            react_step=current_step_num,  # NEW: explicit react step
            plan_step=plan_step_num        # NEW: explicit plan step if available
        )

        logger.info(f"Python Coder - Success: {result['success']}")
        logger.info(f"Verification Iterations: {result.get('verification_iterations', 'N/A')}")
        logger.info(f"Execution Attempts: {result.get('execution_attempts', 'N/A')}")

        if result["success"]:
            # Build execution details safely
            exec_details_parts = []
            if result.get('verification_iterations'):
                exec_details_parts.append(f"{result['verification_iterations']} verification iterations")
            if result.get('execution_attempts'):
                exec_details_parts.append(f"{result['execution_attempts']} execution attempts")
            if isinstance(result.get('execution_time'), (int, float)):
                exec_details_parts.append(f"{result['execution_time']:.2f}s")

            exec_details = ", ".join(exec_details_parts) if exec_details_parts else "completed"
            return f"Code executed successfully:\n{result['output']}\n\nExecution details: {exec_details}"
        else:
            return f"Code execution failed: {result.get('error', 'Unknown error')}"

    async def _execute_file_analyzer(
        self,
        query: str,
        file_paths: Optional[List[str]]
    ) -> str:
        """Execute file_analyzer tool."""
        logger.warning("FILE_ANALYZER called during ReAct loop (should only be in pre-step)")

        if not file_paths:
            return "No files attached to analyze."

        result = file_analyzer.analyze(file_paths=file_paths, user_query=query)

        logger.info(f"File Analyzer - Success: {result.get('success', False)}")
        logger.info(f"Files Analyzed: {result.get('files_analyzed', 0)}")

        if result.get("success"):
            return f"File analysis completed:\n{result.get('summary','')}"
        else:
            return f"File analysis failed: {result.get('error','Unknown error')}"

    def _load_conversation_history(
        self,
        session_id: Optional[str]
    ) -> Optional[List[dict]]:
        """
        Load conversation history from conversation store.

        Args:
            session_id: Session ID for conversation lookup

        Returns:
            List of conversation messages as dicts, or None if not available
        """
        if not session_id:
            return None

        try:
            from backend.storage.conversation_store import conversation_store

            # Load conversation messages
            messages = conversation_store.get_messages(session_id, limit=10)  # Last 10 messages
            if not messages:
                return None

            # Convert to dict format for prompt
            history = []
            for msg in messages:
                history.append({
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat() if msg.timestamp else ""
                })

            return history

        except Exception as e:
            logger.warning(f"[ToolExecutor] Failed to load conversation history: {e}")
            return None

    def _build_react_context(
        self,
        steps: Optional[List[Any]],
        session_id: Optional[str] = None
    ) -> Optional[dict]:
        """
        Build structured react_context with failed attempts for python_coder prompt.

        Args:
            steps: List of ReActStep objects from execution history
            session_id: Session ID for loading saved code files

        Returns:
            Dict with iteration history including failed code and errors, or None if no steps
        """
        if not steps:
            return None

        current_iteration = len(steps) + 1
        history = []

        # Extract failed python_coder attempts from steps
        for step in steps:
            step_info = {
                'thought': step.thought,
                'action': str(step.action),
                'tool_input': step.action_input
            }

            # Check if this was a python_coder action
            if step.action == ToolName.PYTHON_CODER:
                step_info['observation'] = step.observation

                # Try to load the actual code from session directory
                code = self._load_code_for_step(session_id, step.step_num)
                if code:
                    step_info['code'] = code

                # Determine if failed or succeeded
                if "failed" in step.observation.lower() or "error" in step.observation.lower():
                    step_info['status'] = 'error'
                    # Try to extract error message
                    if "error:" in step.observation.lower():
                        error_parts = step.observation.split("error:", 1)
                        if len(error_parts) > 1:
                            step_info['error_reason'] = error_parts[1].strip()[:500]
                    else:
                        step_info['error_reason'] = step.observation[:500]
                else:
                    step_info['status'] = 'success'
                    step_info['observation'] = step.observation[:500]

                history.append(step_info)
            elif "error" in step.observation.lower() or "failed" in step.observation.lower():
                # Include other failed attempts too
                step_info['observation'] = step.observation[:500]
                step_info['status'] = 'error'
                history.append(step_info)

        # Only return context if there are failed attempts
        if not history:
            return None

        return {
            'iteration': current_iteration,
            'history': history
        }

    def _load_code_for_step(
        self,
        session_id: Optional[str],
        step_num: int
    ) -> Optional[str]:
        """
        Load saved code for a specific ReAct step from session directory.

        Args:
            session_id: Session ID
            step_num: Step number to find code for

        Returns:
            Code string if found, None otherwise
        """
        if not session_id:
            return None

        try:
            from pathlib import Path
            from backend.config.settings import settings

            session_dir = Path(settings.python_code_execution_dir) / session_id
            if not session_dir.exists():
                return None

            # Look for code files matching this step (script_step{N}_*.py)
            pattern = f"script_step{step_num}_*.py"
            matching_files = list(session_dir.glob(pattern))

            if not matching_files:
                return None

            # Get the most recent file (by modification time)
            latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)

            # Read the code
            code = latest_file.read_text(encoding='utf-8')
            logger.info(f"[ToolExecutor] Loaded code for step {step_num} from {latest_file.name}")
            return code

        except Exception as e:
            logger.warning(f"[ToolExecutor] Failed to load code for step {step_num}: {e}")
            return None

    def _build_context_for_python_coder(
        self,
        steps: Optional[List[Any]],
        session_id: Optional[str]
    ) -> str:
        """
        Build context string from ReAct execution history for python_coder.

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

            context_parts.append("You can reference and build upon these previous code versions.\n")

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

            # Highlight errors
            if "error" in step.observation.lower() or "failed" in step.observation.lower():
                context_parts.append(f"  ⚠ Note: This action encountered errors")

            context_parts.append("")

        # Add summary of tools tried
        tools_tried = [step.action for step in steps if step.action != ToolName.FINISH]
        if tools_tried:
            context_parts.append(f"Tools already attempted: {', '.join(set(tools_tried))}")

        context_parts.append("\nUse this context to:")
        context_parts.append("- Avoid repeating failed approaches")
        context_parts.append("- Build upon partial results from previous steps")
        context_parts.append("- Generate more targeted code based on what's already known")
        if code_history:
            context_parts.append("- Reference the previous code versions shown above")

        return "\n".join(context_parts)
