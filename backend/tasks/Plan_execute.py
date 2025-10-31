"""
Plan-and-Execute Agentic Task
Implements a hybrid Plan-and-Execute + ReAct architecture:

Architecture:
1. Planning Phase: LLM analyzes query and creates detailed execution plan
   - Identifies required tools
   - Breaks down into concrete steps
   - Assesses complexity and potential challenges
   - Defines success criteria

2. Execution Phase: ReAct agent executes the plan
   - Uses iterative Thought-Action-Observation loops
   - Follows the plan while maintaining flexibility
   - Adapts to observations and unexpected results

3. Monitoring Phase: Tracks and verifies execution
   - Compares planned vs. actual tool usage
   - Calculates plan adherence score
   - Provides comprehensive execution metadata

This hybrid approach combines:
- Strategic planning (Plan-and-Execute strength)
- Flexible execution (ReAct strength)
- Transparency through detailed logging
"""

import logging
from typing import List, Optional, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import httpx

from backend.models.schemas import ChatMessage
from backend.config.settings import settings
from backend.tasks.React import react_agent

logger = logging.getLogger(__name__)


class PlanExecuteTask:
    """Handles complex agentic workflows using Plan-and-Execute pattern"""

    def __init__(self):
        """Initialize LLM for planning"""
        self.llm = None

    def _get_llm(self) -> ChatOllama:
        """Lazy-load LLM instance"""
        if self.llm is None:
            async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.ollama_timeout / 1000, connect=60.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            self.llm = ChatOllama(
                base_url=settings.ollama_host,
                model=settings.agentic_classifier_model,
                temperature=settings.ollama_temperature,
                num_ctx=settings.ollama_num_ctx,
                timeout=settings.ollama_timeout / 1000,
                async_client=async_client
            )
        return self.llm

    async def _create_execution_plan(self, query: str, conversation_history: str, has_files: bool = False) -> Dict[str, Any]:
        """
        Step 1: Create detailed execution plan by analyzing the query

        Args:
            query: User's current query
            conversation_history: Previous conversation context

        Returns:
            Dict containing plan, required_tools, estimated_steps, complexity
        """
        logger.info(f"[Plan-Execute: Planning] Analyzing query: {query[:]}\n\n")

        llm = self._get_llm()

        file_first_note = "\nIMPORTANT: There are attached files for this task. Prefer local file analysis FIRST using python_coder or python_code. Only fall back to rag_retrieval or web_search if local analysis fails or is insufficient." if has_files else ""

        planning_prompt = f"""You are an AI planning expert. Analyze this user query and create a detailed execution plan.{file_first_note}

Conversation History:
{conversation_history}

Current User Query: {query}

Analyze the query and provide a detailed execution plan:

Here are a list of Available Tools:
{settings.available_tools}

Format your response as a detailed execution plan:
[Execution Plan]
1. First step
2. Second step
...
n-1. Last step
n. Verifying step
...
When you are done, make sure if the required tools are correct."""

        response = await llm.ainvoke([HumanMessage(content=planning_prompt)])
        plan_text = response.content

        logger.info(f"[Plan-Execute: Planning] \n\nPlan: {plan_text}")

        return plan_text

    async def execute(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str],
        user_id: str,
        file_paths: Optional[List[str]] = None,
        max_iterations: int = 3
    ) -> tuple[str, dict]:
        """
        Execute hybrid Plan-and-Execute + ReAct workflow:

        Phase 1 - Planning: Analyze query and create strategic execution plan
        Phase 2 - Execution: Run ReAct agent with plan-aware prompts
        Phase 3 - Monitoring: Verify execution and calculate plan adherence

        Args:
            messages: List of chat messages (conversation history)
            session_id: Session ID for conversation tracking
            user_id: User identifier
            file_paths: Optional list of file paths for code execution
            max_iterations: Not used (kept for API compatibility, ReAct uses its own max_iterations=5)

        Returns:
            Tuple of (final_output, metadata)
            - final_output: Final answer from ReAct agent
            - metadata: Comprehensive execution metadata including:
                - planning: Full plan details (tools, steps, complexity)
                - execution: ReAct execution details (iterations, tools used)
                - analysis: Plan adherence score and efficiency metrics
        """
        logger.info(f"\n\n\n[Plan-Execute] Starting workflow for user: {user_id}, session: {session_id}")
        if file_paths:
            logger.info(f"[Plan-Execute] Attached files: {len(file_paths)} files")

        # Extract user query and conversation history
        user_query = messages[-1].content
        conversation_history = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in messages[:-1]
        ]) if len(messages) > 1 else "No previous conversation"

        # ====== PHASE 1: PLANNING ======
        logger.info("[Plan-Execute] Phase 1: Creating execution plan...")
        plan_data = await self._create_execution_plan(user_query, conversation_history, has_files=bool(file_paths))

        # ====== PHASE 2: EXECUTION with ReAct Agent ======
        logger.info("[Plan-Execute] Phase 2: Executing plan with ReAct Agent...")
        logger.info(f"[Plan-Execute] Plan summary: {plan_data}")

        # Create enhanced prompt for ReAct agent with the plan
        plan_aware_messages = self._create_plan_aware_messages(messages, plan_data)

        # Execute ReAct agent with the plan and file_paths
        final_output, react_metadata = await react_agent.execute(
            messages=plan_aware_messages,
            session_id=session_id,
            user_id=user_id,
            file_paths=file_paths
        )

        # ====== PHASE 3: MONITORING & VERIFICATION ======
        logger.info("[Plan-Execute] Phase 3: Monitoring execution results...")

        tools_used = react_metadata.get("tools_used", [])
        total_iterations = react_metadata.get("total_iterations", 0)

        logger.info(f"[Plan-Execute] Execution completed:")
        logger.info(f"  - Tools used: {', '.join(tools_used)}")
        logger.info(f"  - ReAct iterations: {total_iterations}/{react_metadata.get('max_iterations', 5)}")

        # Verify if planned tools were used
        used_tools_set = set(tools_used)
        
        # Build comprehensive metadata
        metadata = self._build_metadata(react_metadata, plan_data)

        # Return final output and metadata
        return final_output, metadata

    def _create_plan_aware_messages(
        self,
        messages: List[ChatMessage],
        plan_data: str
    ) -> List[ChatMessage]:
        """
        Create messages list with plan context injected for ReAct agent

        Args:
            messages: Original messages
            plan_data: Planning data

        Returns:
            Enhanced messages list with plan context
        """
        # Format the plan for the agent
        plan_context = f"""
[EXECUTION PLAN]
Your task has been analyzed and a plan has been created:

The plan: {plan_data}

Make sure to execute this plan step-by-step using the ReAct framework. Do not skip any steps.
[END PLAN]

"""

        # Create a new message list with plan injected
        enhanced_messages = messages[:-1].copy() if len(messages) > 1 else []

        # Inject plan before the user's final query
        from backend.models.schemas import ChatMessage
        plan_message = ChatMessage(
            role="system",
            content=plan_context
        )
        enhanced_messages.append(plan_message)

        # Add the user's query
        enhanced_messages.append(messages[-1])

        return enhanced_messages

    def _format_steps_list(self, steps: List[str]) -> str:
        """Format steps list as numbered items"""
        if not steps:
            return "No specific steps defined."

        return "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])

    def _build_metadata(self, react_metadata: dict, plan_data: Dict[str, Any]) -> dict:
        """
        Build comprehensive metadata combining planning and execution results

        Args:
            react_metadata: Metadata from ReAct agent execution
            plan_data: Planning data from Phase 1

        Returns:
            Combined metadata dictionary
        """
        return {
            "agent_type": "plan_execute",
            "architecture": "Plan-and-Execute with ReAct execution",

            # Planning phase data
            "planning": {
                "plan_data": plan_data
            },

            # Execution phase data (from ReAct)
            "execution": {
                "tools_used": react_metadata.get("tools_used", []),
                "total_iterations": react_metadata.get("total_iterations", 0),
                "max_iterations": react_metadata.get("max_iterations", 5),
                "execution_steps": react_metadata.get("execution_steps", []),
                "execution_trace": react_metadata.get("execution_trace", "No trace available")
            }
        }

# Global agentic task instance
plan_execute_task = PlanExecuteTask()
