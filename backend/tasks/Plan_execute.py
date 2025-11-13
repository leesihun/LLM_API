"""
Plan-and-Execute Agentic Task
Implements a hybrid Plan-and-Execute + ReAct architecture:

Architecture:
1. Planning Phase: LLM analyzes query and creates detailed execution plan
   - Identifies required tools
   - Breaks down into concrete steps
   - Assesses complexity and potential challenges
   - Defines success criteria

2. Execution Phase: ReAct agent executes the plan step-by-step
   - Each plan step is executed by ReAct with specific goal
   - Uses iterative Thought-Action-Observation loops per step
   - Follows the plan while maintaining flexibility
   - Adapts to observations and unexpected results
   - Automatic tool fallback on failure

3. Monitoring Phase: Tracks and verifies execution
   - Compares planned vs. actual tool usage
   - Tracks step-by-step success/failure
   - Provides comprehensive execution metadata

This hybrid approach combines:
- Strategic planning (Plan-and-Execute strength)
- Flexible execution (ReAct strength)
- Tool fallback on failure
- Transparency through detailed logging
"""

import json
import re
from typing import List, Optional, Dict, Any

from langchain_core.messages import HumanMessage

from backend.models.schemas import ChatMessage, PlanStep, StepResult
from backend.config.settings import settings
from backend.config import prompts
from backend.tasks.react import react_agent
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PlanExecuteTask:
    """
    Plan-and-Execute agent that creates structured plans and executes them.

    This agent combines strategic planning with flexible execution by:
    1. Analyzing queries and creating structured plans
    2. Executing plans step-by-step using ReAct agent
    3. Tracking and verifying execution with comprehensive metadata
    """

    def __init__(self):
        """Initialize the Plan-and-Execute agent."""
        self.llm = None

    def _get_llm(self):
        """Lazy-load the LLM."""
        if self.llm is None:
            from backend.utils.llm_factory import LLMFactory
            self.llm = LLMFactory.create_llm(temperature=0.3)
        return self.llm

    async def _create_execution_plan(
        self,
        query: str,
        conversation_history: str,
        has_files: bool = False
    ) -> List[PlanStep]:
        """
        Create detailed structured execution plan by analyzing the query.

        Args:
            query: User's current query
            conversation_history: Previous conversation context
            has_files: Whether files are attached

        Returns:
            List of PlanStep objects with structured plan
        """
        logger.info(f"[Plan-Execute: Planning] Analyzing query: {query[:200]}...")

        llm = self._get_llm()

        # Use centralized prompt from prompts module
        planning_prompt = prompts.get_execution_plan_prompt(
            query=query,
            conversation_history=conversation_history,
            available_tools=settings.available_tools,
            has_files=has_files
        )

        response = await llm.ainvoke([HumanMessage(content=planning_prompt)])
        plan_text = response.content.strip()

        logger.info(f"[Plan-Execute: Planning] Raw LLM response:\n{plan_text[:500]}...")

        # Parse JSON plan
        try:
            # Extract JSON if wrapped in markdown or other text
            json_match = re.search(r'\[\s*\{.*\}\s*\]', plan_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = plan_text

            plan_data = json.loads(json_str)
            
            # Convert to PlanStep objects
            plan_steps = []
            for step_data in plan_data:
                plan_step = PlanStep(
                    step_num=step_data.get("step_num", len(plan_steps) + 1),
                    goal=step_data.get("goal", ""),
                    primary_tools=step_data.get("primary_tools", []),
                    fallback_tools=step_data.get("fallback_tools", []),
                    success_criteria=step_data.get("success_criteria", ""),
                    context=step_data.get("context")
                )
                plan_steps.append(plan_step)

            logger.info(f"[Plan-Execute: Planning] Successfully parsed {len(plan_steps)} steps")
            for step in plan_steps:
                logger.info(f"  Step {step.step_num}: {step.goal}")
                logger.info(f"    Primary tools: {step.primary_tools}")
                logger.info(f"    Fallback tools: {step.fallback_tools}")

            return plan_steps

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"[Plan-Execute: Planning] Failed to parse plan JSON: {e}")
            logger.error(f"[Plan-Execute: Planning] Raw response: {plan_text}")
            
            # Fallback: Create a simple single-step plan
            logger.warning("[Plan-Execute: Planning] Creating fallback single-step plan")
            fallback_tools = ["python_coder", "python_code"] if has_files else ["web_search"]
            fallback_step = PlanStep(
                step_num=1,
                goal=f"Answer the query: {query[:100]}",
                primary_tools=fallback_tools,
                fallback_tools=["rag_retrieval"],
                success_criteria="Query answered with relevant information",
                context="Fallback plan due to parsing failure"
            )
            return [fallback_step]

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

        Phase 1 - Planning: Analyze query and create structured execution plan
        Phase 2 - Execution: Run ReAct agent in guided mode (step-by-step)
        Phase 3 - Monitoring: Track step results and build comprehensive metadata

        Args:
            messages: List of chat messages (conversation history)
            session_id: Session ID for conversation tracking
            user_id: User identifier
            file_paths: Optional list of file paths for code execution
            max_iterations: Max iterations per step (passed to ReAct)

        Returns:
            Tuple of (final_output, metadata)
            - final_output: Final answer from ReAct agent
            - metadata: Comprehensive execution metadata including:
                - planning: Structured plan with all steps
                - execution: Step-by-step execution results
                - summary: Overall success rate and tools used
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
        logger.info("[Plan-Execute] Phase 1: Creating structured execution plan...")
        plan_steps = await self._create_execution_plan(user_query, conversation_history, has_files=bool(file_paths))

        logger.info(f"[Plan-Execute] Plan created with {len(plan_steps)} steps")

        # ====== PHASE 2: EXECUTION with ReAct Agent (Guided Mode) ======
        logger.info("[Plan-Execute] Phase 2: Executing plan step-by-step with ReAct Agent...")

        # Execute ReAct agent in guided mode with structured plan
        final_output, step_results = await react_agent.execute_with_plan(
            plan_steps=plan_steps,
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            file_paths=file_paths,
            max_iterations_per_step=max_iterations
        )

        # ====== PHASE 3: MONITORING & VERIFICATION ======
        logger.info("[Plan-Execute] Phase 3: Analyzing execution results...")

        # Calculate success statistics
        total_steps = len(step_results)
        successful_steps = sum(1 for r in step_results if r.success)
        success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0

        # Collect all tools used
        tools_used = list(set([r.tool_used for r in step_results if r.tool_used]))

        logger.info(f"[Plan-Execute] Execution completed:")
        logger.info(f"  - Total steps: {total_steps}")
        logger.info(f"  - Successful steps: {successful_steps}/{total_steps} ({success_rate:.1f}%)")
        logger.info(f"  - Tools used: {', '.join(tools_used)}")

        # Build comprehensive metadata
        metadata = self._build_metadata_from_steps(plan_steps, step_results, tools_used)

        # Return final output and metadata
        return final_output, metadata

    def _build_metadata_from_steps(
        self,
        plan_steps: List[PlanStep],
        step_results: List[StepResult],
        tools_used: List[str]
    ) -> dict:
        """
        Build comprehensive metadata from structured plan and results

        Args:
            plan_steps: Original plan steps
            step_results: Results from each step execution
            tools_used: List of unique tools used

        Returns:
            Combined metadata dictionary
        """
        # Convert PlanStep objects to dicts for JSON serialization
        plan_steps_dict = [
            {
                "step_num": step.step_num,
                "goal": step.goal,
                "primary_tools": step.primary_tools,
                "fallback_tools": step.fallback_tools,
                "success_criteria": step.success_criteria,
                "context": step.context
            }
            for step in plan_steps
        ]

        # Convert StepResult objects to dicts
        step_results_dict = [
            {
                "step_num": result.step_num,
                "goal": result.goal,
                "success": result.success,
                "tool_used": result.tool_used,
                "attempts": result.attempts,
                "observation": result.observation[:500] if result.observation else "",  # Truncate for metadata
                "error": result.error
            }
            for result in step_results
        ]

        # Calculate statistics
        total_steps = len(step_results)
        successful_steps = sum(1 for r in step_results if r.success)
        success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0
        total_attempts = sum(r.attempts for r in step_results)

        return {
            "agent_type": "plan_execute",
            "architecture": "Plan-and-Execute with ReAct step-by-step execution",

            # Planning phase data
            "planning": {
                "total_steps": len(plan_steps),
                "plan_steps": plan_steps_dict
            },

            # Execution phase data
            "execution": {
                "step_results": step_results_dict,
                "tools_used": tools_used,
                "total_attempts": total_attempts
            },

            # Summary statistics
            "summary": {
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "failed_steps": total_steps - successful_steps,
                "success_rate": round(success_rate, 2),
                "total_tool_attempts": total_attempts
            }
        }

# Global agentic task instance
plan_execute_task = PlanExecuteTask()
