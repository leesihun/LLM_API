"""
ReAct Execution Module

Handles tool execution, verification, and auto-finish detection.

Consolidated from:
- tool_executor.py
- verification.py
- utils.py
"""

from typing import List, Optional, Tuple, Any
from langchain_core.messages import HumanMessage
import json

from backend.tools.web_search import web_search_tool
from backend.tools.rag_retriever import rag_retriever_tool
from backend.tools.python_coder import python_coder_tool
from backend.tools.file_analyzer import file_analyzer
from backend.tools.vision_analyzer import vision_analyzer_tool
from backend.utils.logging_utils import get_logger
from .models import ToolName

logger = get_logger(__name__)


class ToolExecutor:
    """
    Executes tools for ReAct agent with direct routing.

    Routes actions to appropriate tools and handles tool-specific formatting.
    """

    def __init__(self, llm, user_id: str = "default"):
        """
        Initialize ToolExecutor.

        Args:
            llm: LLM instance for operations
            user_id: User ID for tracking and logging
        """
        self.llm = llm
        self.user_id = user_id

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
            action: Tool name to execute
            action_input: Input for the tool
            file_paths: Optional list of file paths
            session_id: Optional session ID
            steps: Optional list of ReActStep objects
            plan_context: Optional plan context

        Returns:
            Observation string from tool execution
        """
        try:
            logger.info(f"EXECUTING TOOL: {action}")
            logger.info(f"Tool Input: {action_input}")

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

            elif action == ToolName.VISION_ANALYZER:
                return await self._execute_vision_analyzer(action_input, file_paths)

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

        if context_metadata.get('query_enhanced'):
            observation = f"[Search performed with contextual enhancement]\n{observation}"

        final_observation = observation if observation else "No web search results found."

        logger.info(f"Web Search Results: {len(results)} results")
        if context_metadata.get('enhanced_query'):
            logger.info(f"Enhanced query: {context_metadata['enhanced_query']}")

        return final_observation

    async def _execute_rag_retrieval(self, query: str) -> str:
        """Execute RAG retrieval tool."""
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
        # Build context from execution history (high-level summary only)
        context = self._build_context_for_python_coder(steps)

        # Build react_context with ALL code attempts and session history
        react_context = self._build_react_context(steps, session_id)

        # Load conversation history
        conversation_history = self._load_conversation_history(session_id)

        # Use ReAct step number for hierarchical prompt organization
        current_step_num = len(steps) + 1 if steps else 1
        stage_prefix = f"step{current_step_num}"

        # Extract plan_step if available
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
            react_step=current_step_num,
            plan_step=plan_step_num
        )

        logger.info(f"Python Coder - Success: {result['success']}")
        logger.info(f"Verification Iterations: {result.get('verification_iterations', 'N/A')}")
        logger.info(f"Execution Attempts: {result.get('execution_attempts', 'N/A')}")

        if result["success"]:
            # Build execution details
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

    async def _execute_vision_analyzer(
        self,
        query: str,
        file_paths: Optional[List[str]]
    ) -> str:
        """Execute vision_analyzer tool."""
        if not file_paths:
            return "No files attached for vision analysis."

        result = await vision_analyzer_tool(
            query=query,
            file_paths=file_paths,
            user_id=self.user_id
        )

        logger.info(f"Vision Analyzer - Success: {result.get('success', False)}")
        logger.info(f"Images Analyzed: {result.get('image_count', 0)}")

        if result.get("success"):
            analysis = result.get('analysis', '')
            images = result.get('images_analyzed', [])
            return f"Vision analysis completed for {len(images)} image(s):\n{analysis}"
        else:
            return f"Vision analysis failed: {result.get('error', 'Unknown error')}"

    def _load_conversation_history(
        self,
        session_id: Optional[str]
    ) -> Optional[List[dict]]:
        """Load conversation history from conversation store."""
        if not session_id:
            return None

        try:
            from backend.storage.conversation_store import conversation_store

            messages = conversation_store.get_messages(session_id, limit=10)
            if not messages:
                return None

            history = []
            for msg in messages:
                history.append({
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat() if msg.timestamp else ""
                })

            return history

        except Exception as e:
            logger.warning(f"Failed to load conversation history: {e}")
            return None

    def _build_react_context(
        self,
        steps: Optional[List[Any]],
        session_id: Optional[str] = None
    ) -> Optional[dict]:
        """
        Build structured react_context with ALL code attempts across the entire session.

        NEW BEHAVIOR: Captures complete code execution history from the session,
        not just the current ReAct execution. This includes:
        - All attempts from ALL ReAct steps in current execution
        - All code from previous ReAct executions in the same session
        - Full error messages and observations

        Returns:
            Dict with iteration history including all code versions and errors
        """
        if not steps and not session_id:
            return None

        current_iteration = len(steps) + 1 if steps else 1
        history = []

        # First, load ALL code history from session (across all executions)
        all_session_code = self._load_all_session_code(session_id)

        # Extract failed python_coder attempts from current ReAct execution
        if steps:
            for step in steps:
                step_info = {
                    'thought': step.thought,
                    'action': str(step.action),
                    'tool_input': step.action_input
                }

                if step.action == ToolName.PYTHON_CODER:
                    step_info['observation'] = step.observation

                    # Load ALL code attempts for this step (not just latest)
                    code_attempts = self._load_all_attempts_for_step(session_id, step.step_num)
                    if code_attempts:
                        if len(code_attempts) == 1:
                            step_info['code'] = code_attempts[0]['code']
                        else:
                            # Multiple attempts - include all
                            step_info['code_attempts'] = code_attempts

                    # Determine status
                    if "failed" in step.observation.lower() or "error" in step.observation.lower():
                        step_info['status'] = 'error'
                        if "error:" in step.observation.lower():
                            error_parts = step.observation.split("error:", 1)
                            if len(error_parts) > 1:
                                step_info['error_reason'] = error_parts[1].strip()
                        else:
                            step_info['error_reason'] = step.observation
                    else:
                        step_info['status'] = 'success'
                        step_info['observation'] = step.observation

                    history.append(step_info)
                elif "error" in step.observation.lower() or "failed" in step.observation.lower():
                    step_info['observation'] = step.observation
                    step_info['status'] = 'error'
                    history.append(step_info)

        # Add session-wide code history if available
        context = {
            'iteration': current_iteration,
            'history': history if history else []
        }

        if all_session_code:
            context['session_code_history'] = all_session_code

        return context if (history or all_session_code) else None

    def _load_all_session_code(
        self,
        session_id: Optional[str]
    ) -> Optional[List[dict]]:
        """
        Load ALL code files from the entire session (across all executions).

        This captures the complete code execution history for the session,
        including code from previous ReAct executions.

        Returns:
            List of dicts with code, filename, timestamp, organized by recency
        """
        if not session_id:
            return None

        try:
            from pathlib import Path
            from backend.config.settings import settings

            session_dir = Path(settings.python_code_execution_dir) / session_id
            if not session_dir.exists():
                return None

            # Find ALL script_*.py files in session
            all_code_files = list(session_dir.glob("script_*.py"))

            if not all_code_files:
                return None

            # Sort by modification time (most recent first)
            all_code_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            code_history = []
            for code_file in all_code_files:
                try:
                    code = code_file.read_text(encoding='utf-8')
                    stage_name = code_file.stem.replace('script_', '')
                    timestamp = code_file.stat().st_mtime

                    code_history.append({
                        'code': code,
                        'filename': code_file.name,
                        'stage_name': stage_name,
                        'timestamp': timestamp
                    })
                except Exception as e:
                    logger.warning(f"Failed to read {code_file.name}: {e}")

            if code_history:
                logger.info(f"Loaded {len(code_history)} code file(s) from session history")

            return code_history if code_history else None

        except Exception as e:
            logger.warning(f"Failed to load session code history: {e}")
            return None

    def _load_all_attempts_for_step(
        self,
        session_id: Optional[str],
        step_num: int
    ) -> Optional[List[dict]]:
        """
        Load ALL code attempts for a specific ReAct step.

        This captures multiple retry attempts within the same step
        (e.g., step1_attempt1.py, step1_attempt2.py, step1_attempt3.py).

        Returns:
            List of dicts with code, attempt_num, timestamp, sorted by attempt number
        """
        if not session_id:
            return None

        try:
            from pathlib import Path
            from backend.config.settings import settings
            import re

            session_dir = Path(settings.python_code_execution_dir) / session_id
            if not session_dir.exists():
                return None

            # Look for all code files matching this step
            pattern = f"script_step{step_num}_*.py"
            matching_files = list(session_dir.glob(pattern))

            if not matching_files:
                return None

            # Parse attempt numbers and sort
            attempts = []
            for file_path in matching_files:
                try:
                    code = file_path.read_text(encoding='utf-8')

                    # Extract attempt number from filename
                    # Format: script_step{N}_attempt{M}.py or script_step{N}_{stage}.py
                    match = re.search(r'step\d+_attempt(\d+)', file_path.name)
                    if match:
                        attempt_num = int(match.group(1))
                    else:
                        # No attempt number - treat as attempt 1
                        attempt_num = 1

                    attempts.append({
                        'code': code,
                        'attempt_num': attempt_num,
                        'filename': file_path.name,
                        'timestamp': file_path.stat().st_mtime
                    })
                except Exception as e:
                    logger.warning(f"Failed to read {file_path.name}: {e}")

            if not attempts:
                return None

            # Sort by attempt number
            attempts.sort(key=lambda x: x['attempt_num'])

            logger.info(f"Loaded {len(attempts)} attempt(s) for step {step_num}")
            return attempts

        except Exception as e:
            logger.warning(f"Failed to load attempts for step {step_num}: {e}")
            return None

    def _build_context_for_python_coder(
        self,
        steps: Optional[List[Any]]
    ) -> str:
        """
        Build context string from ReAct execution history.

        NOTE: Code history is now fully handled by react_context parameter,
        which includes ALL code from session (via _load_all_session_code).
        This method only provides high-level agent activity summary.
        """
        if not steps:
            return ""

        context_parts = []
        context_parts.append("=== Previous Agent Activity ===\n")

        # Include recent steps (last 3)
        recent_steps = steps[-3:] if len(steps) > 3 else steps

        for step in recent_steps:
            context_parts.append(f"Step {step.step_num}:")
            context_parts.append(f"  Thought: {step.thought}")
            context_parts.append(f"  Action: {step.action}")
            context_parts.append(f"  Result: {step.observation[:200]}...")  # Truncate long observations

            if "error" in step.observation.lower() or "failed" in step.observation.lower():
                context_parts.append(f"  Status: FAILED")

            context_parts.append("")

        # Add summary of tools tried
        tools_tried = [step.action for step in steps if step.action != ToolName.FINISH]
        if tools_tried:
            context_parts.append(f"Tools already attempted: {', '.join(set(tools_tried))}")

        context_parts.append("\nUse this context to:")
        context_parts.append("- Avoid repeating failed approaches")
        context_parts.append("- Build upon partial results from previous steps")
        context_parts.append("- Generate more targeted code based on what's already known")

        return "\n".join(context_parts)


class StepVerifier:
    """
    Handles step verification and auto-finish detection.

    Features:
    - Heuristic-based auto-finish (fast)
    - LLM-enhanced auto-finish with confidence scoring
    - Plan step verification
    """

    def __init__(self, llm):
        """
        Initialize step verifier.

        Args:
            llm: LangChain LLM instance
        """
        self.llm = llm

    def should_auto_finish(self, observation: str, step_num: int) -> bool:
        """
        Heuristic-based auto-finish detection (fast, no LLM call).

        Checks:
        1. Length check (> 200 chars)
        2. No error keywords
        3. Has answer keywords OR very long (> 500 chars)

        Args:
            observation: Latest observation
            step_num: Current step number

        Returns:
            True if should auto-finish
        """
        if step_num < 2:
            return False

        if not observation or len(observation) < 200:
            return False

        observation_lower = observation.lower()

        # Has error - don't finish
        error_keywords = ['error', 'failed', 'exception', 'traceback']
        if any(keyword in observation_lower for keyword in error_keywords):
            return False

        # Has answer keywords - finish
        answer_keywords = ['answer is', 'result is', 'found that', 'shows that', 'indicates']
        if any(keyword in observation_lower for keyword in answer_keywords):
            logger.info("[EARLY EXIT] Observation contains answer keywords")
            return True

        # Very long observation with data - probably complete
        if len(observation) > 500:
            logger.info("[EARLY EXIT] Observation is substantial (> 500 chars)")
            return True

        return False

    async def should_auto_finish_enhanced(
        self,
        observation: str,
        user_query: str,
        iteration: int,
        steps_context: str
    ) -> Tuple[bool, float, str]:
        """
        Enhanced auto-finish with LLM-based adequacy check.

        Checks heuristics first, then uses LLM for borderline cases.

        Args:
            observation: Latest observation
            user_query: Original user query
            iteration: Current iteration number
            steps_context: Formatted context from steps

        Returns:
            Tuple of (should_finish, confidence_score, reason)
        """
        # LLM adequacy check for borderline cases
        try:
            prompt = f"""Assess if this observation adequately answers the user's query.

User Query:
{user_query}

Latest Observation:
{observation}

Previous Steps Context:
{steps_context}

Question: Does the observation contain enough information to FULLY and ACCURATELY answer the user's query?

Respond with JSON only:
{{
  "adequate": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation",
  "missing_info": "what's still needed if inadequate, otherwise empty string"
}}
"""

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            # Parse JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            result = json.loads(response_text.strip())

            adequate = result.get("adequate", False)
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "No reason provided")
            missing_info = result.get("missing_info", "")

            # Require high confidence (>= 0.8) to auto-finish
            should_finish = adequate and confidence >= 0.9

            if should_finish:
                logger.info(f"[AUTO-FINISH ENHANCED] Confidence: {confidence:.2f} - {reason}")
            else:
                logger.info(f"[CONTINUE] Confidence: {confidence:.2f} - {reason}")
                if missing_info:
                    logger.info(f"[CONTINUE] Missing: {missing_info}")

            return should_finish, confidence, reason

        except Exception as e:
            logger.error(f"[AUTO-FINISH ENHANCED] LLM check failed: {e}")
            raise
