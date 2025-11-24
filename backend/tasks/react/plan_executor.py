"""
ReAct Plan Executor Module

Handles guided plan execution where the ReAct agent executes
a structured plan step-by-step.

Features:
- Dynamic plan adaptation (re-planning when steps fail or observations reveal new requirements)
- Preserves completed work when adapting plans
- Multi-trigger re-planning detection

Extracted from React.py for improved modularity.
"""

from typing import List, Tuple, Optional
from langchain_core.messages import HumanMessage

from .models import ToolName
from .plan_adapter import PlanAdapter
from backend.config.prompts import PromptRegistry
from backend.models.schemas import PlanStep, StepResult, ChatMessage
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PlanExecutor:
    """
    Executes structured plans step-by-step using ReAct-style tool execution.

    This class coordinates plan execution by:
    1. Iterating through plan steps
    2. Executing each step with its primary tool (no fallback)
    3. Verifying step success
    4. Accumulating observations for final answer

    Simplified: Each step uses only its primary tool. No fallback logic.
    If a step fails, the plan should explicitly define retry steps.

    Attributes:
        tool_executor: ToolExecutor instance for running tools
        context_manager: ContextManager for building execution context
        verifier: StepVerifier for validating step success
        answer_generator: AnswerGenerator for final synthesis
        llm: Language model instance
    """

    def __init__(
        self,
        tool_executor,
        context_manager,
        verifier,
        answer_generator,
        llm
    ):
        """
        Initialize plan executor with dynamic plan adaptation.

        Args:
            tool_executor: ToolExecutor instance
            context_manager: ContextManager instance
            verifier: StepVerifier instance
            answer_generator: AnswerGenerator instance
            llm: Language model instance
        """
        self.tool_executor = tool_executor
        self.context_manager = context_manager
        self.verifier = verifier
        self.answer_generator = answer_generator
        self.llm = llm
        self.plan_adapter = PlanAdapter(llm)  # New: Dynamic plan adaptation

    async def execute_plan(
        self,
        plan_steps: List[PlanStep],
        user_query: str,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        max_iterations_per_step: int = 3
    ) -> Tuple[str, List[StepResult]]:
        """
        Execute a structured plan step-by-step.

        Args:
            plan_steps: List of plan steps to execute
            user_query: Original user query
            file_paths: Optional list of file paths
            session_id: Optional session ID
            max_iterations_per_step: Max iterations to try per step

        Returns:
            Tuple of (final_answer, step_results)
        """
        logger.info("\n\n\n" + "=" * 100)
        logger.info("[PlanExecutor] EXECUTION STARTED")
        logger.info(f"Plan has {len(plan_steps)} steps")
        if file_paths:
            logger.info(f"Attached Files: {len(file_paths)} files")
        logger.info("=" * 100 + "\n\n\n")

        # Track results from each step
        step_results: List[StepResult] = []
        accumulated_observations = []

        # Execute each plan step
        for step_idx, plan_step in enumerate(plan_steps):
            logger.info("\n" + "#" * 100)
            logger.info(f"EXECUTING PLAN STEP {plan_step.step_num}/{len(plan_steps)}")
            logger.info(f"Goal: {plan_step.goal}")
            logger.info(f"Tool: {plan_step.primary_tools[0] if plan_step.primary_tools else 'None'}")
            logger.info("#" * 100 + "\n")

            # Execute this step
            step_result = await self.execute_step(
                plan_step=plan_step,
                user_query=user_query,
                accumulated_observations=accumulated_observations,
                file_paths=file_paths,
                session_id=session_id,
                all_plan_steps=plan_steps,
                step_results=step_results
            )

            step_results.append(step_result)

            # Add observation to accumulated context
            if step_result.observation:
                accumulated_observations.append(
                    f"Step {step_result.step_num} ({step_result.goal}): {step_result.observation}"
                )

                # Apply pruning if observations grow too large (keep last 5 steps)
                if len(accumulated_observations) > 5:
                    pruned_observations = accumulated_observations[-5:]
                    logger.info(f"[PlanExecutor] Context pruned: keeping last 5 observations (removed {len(accumulated_observations) - 5})")
                    accumulated_observations = pruned_observations

            logger.info(f"\n[PlanExecutor] Step {plan_step.step_num} {'✓ SUCCESS' if step_result.success else '✗ FAILED'}")
            logger.info(f"Tool used: {step_result.tool_used}")
            logger.info(f"Attempts: {step_result.attempts}\n")

            # Check if re-planning is needed (skip if this is the last step)
            remaining_steps = plan_steps[step_idx + 1:] if step_idx + 1 < len(plan_steps) else []
            if remaining_steps:
                needs_replan, reason = await self.plan_adapter.should_replan(
                    current_step_result=step_result,
                    remaining_steps=remaining_steps,
                    original_query=user_query
                )

                if needs_replan:
                    logger.warning(f"\n{'='*100}")
                    logger.warning(f"RE-PLANNING TRIGGERED: {reason}")
                    logger.warning(f"{'='*100}\n")

                    # Generate adapted plan
                    adapted_plan = await self.plan_adapter.adapt_plan(
                        original_plan=plan_steps,
                        completed_steps=step_results,
                        current_step_result=step_result,
                        remaining_steps=remaining_steps,
                        original_query=user_query
                    )

                    logger.info(f"\n[PlanExecutor] Adapted plan generated: {len(adapted_plan)} new steps")
                    for i, step in enumerate(adapted_plan, 1):
                        logger.info(f"  {i}. {step.goal}")

                    # Replace remaining steps with adapted plan
                    # Renumber adapted plan steps to continue from current step number
                    next_step_num = plan_step.step_num + 1
                    for i, step in enumerate(adapted_plan):
                        step.step_num = next_step_num + i

                    # Update plan_steps to include completed + adapted
                    plan_steps = plan_steps[:step_idx + 1] + adapted_plan

                    logger.info(f"[PlanExecutor] Updated plan: {len(plan_steps)} total steps\n")
                else:
                    logger.info(f"[PlanExecutor] Plan execution on track - continuing as planned\n")

        # Generate final answer
        logger.info("\n" + "=" * 100)
        logger.info("[PlanExecutor] Finalizing answer...")
        logger.info("=" * 100 + "\n")

        final_answer = await self._generate_final_answer_from_steps(
            user_query=user_query,
            step_results=step_results,
            accumulated_observations=accumulated_observations
        )

        logger.info("\n[PlanExecutor] EXECUTION COMPLETED")
        logger.info(f"Total steps executed: {len(step_results)}")
        logger.info(f"Successful steps: {sum(1 for r in step_results if r.success)}/{len(step_results)}\n")

        return final_answer, step_results

    async def execute_step(
        self,
        plan_step: PlanStep,
        user_query: str,
        accumulated_observations: List[str],
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        all_plan_steps: Optional[List[PlanStep]] = None,
        step_results: Optional[List[StepResult]] = None
    ) -> StepResult:
        """
        Execute a single plan step with single tool (no fallback).

        Simplified: Each step uses only its primary tool. No fallback tools.
        If execution fails, return failure. Let the plan handle retries.

        Args:
            plan_step: The plan step to execute
            user_query: Original user query
            accumulated_observations: Context from previous steps
            file_paths: Optional file paths
            session_id: Optional session ID
            all_plan_steps: Optional list of all plan steps for context
            step_results: Optional list of completed step results

        Returns:
            StepResult with execution details
        """
        logger.info(f"[PlanExecutor Step {plan_step.step_num}] Starting execution...")

        # Build context from previous steps
        context = "\n".join(accumulated_observations) if accumulated_observations else "This is the first step."

        # Use only the primary tool (no fallback)
        tool_name = plan_step.primary_tools[0] if plan_step.primary_tools else None

        if not tool_name:
            logger.error(f"[PlanExecutor Step {plan_step.step_num}] No primary tool specified")
            return StepResult(
                step_num=plan_step.step_num,
                goal=plan_step.goal,
                success=False,
                tool_used=None,
                attempts=0,
                observation="No tool specified for this step",
                error="No primary tool specified",
                metadata={}
            )

        logger.info(f"[PlanExecutor Step {plan_step.step_num}] Executing with primary tool '{tool_name}'")

        # Execute the tool once
        try:
            observation, success = await self._execute_tool_for_step(
                tool_name=tool_name,
                plan_step=plan_step,
                user_query=user_query,
                context=context,
                file_paths=file_paths,
                session_id=session_id,
                all_plan_steps=all_plan_steps,
                step_results=step_results
            )

            logger.info(f"[PlanExecutor Step {plan_step.step_num}] {'SUCCESS' if success else 'FAILED'} with tool '{tool_name}'")

            return StepResult(
                step_num=plan_step.step_num,
                goal=plan_step.goal,
                success=success,
                tool_used=tool_name,
                attempts=1,
                observation=observation,
                error=None if success else "Tool did not achieve goal",
                metadata={"tool_type": "primary"}
            )

        except Exception as e:
            logger.error(f"[PlanExecutor Step {plan_step.step_num}] Tool '{tool_name}' failed: {e}")
            return StepResult(
                step_num=plan_step.step_num,
                goal=plan_step.goal,
                success=False,
                tool_used=tool_name,
                attempts=1,
                observation=f"Error executing {tool_name}: {str(e)}",
                error=str(e),
                metadata={"tool_type": "primary"}
            )

    async def _execute_tool_for_step(
        self,
        tool_name: str,
        plan_step: PlanStep,
        user_query: str,
        context: str,
        file_paths: Optional[List[str]],
        session_id: Optional[str],
        all_plan_steps: Optional[List[PlanStep]] = None,
        step_results: Optional[List[StepResult]] = None
    ) -> Tuple[str, bool]:
        """
        Execute a specific tool to achieve the step goal.

        Args:
            tool_name: Name of tool to execute
            plan_step: Current plan step
            user_query: Original user query
            context: Context from previous steps
            file_paths: Optional file paths
            session_id: Optional session ID
            all_plan_steps: Optional list of all plan steps for context
            step_results: Optional list of completed step results

        Returns:
            Tuple of (observation, success)
        """
        # Use plan step's goal + context directly as action input (no LLM call needed!)
        action_input = self._build_action_input_from_plan(
            plan_step=plan_step,
            user_query=user_query,
            context=context
        )

        # Build plan_context for python_coder
        plan_context = None
        if tool_name in ["python_code", "python_coder"] and all_plan_steps:
            plan_context = self._build_plan_context(
                current_step=plan_step,
                all_plan_steps=all_plan_steps,
                step_results=step_results or []
            )

        # Map tool name to ToolName enum
        tool_mapping = {
            "web_search": ToolName.WEB_SEARCH,
            "rag_retrieval": ToolName.RAG_RETRIEVAL,
            "python_code": ToolName.PYTHON_CODER,
            "python_coder": ToolName.PYTHON_CODER,
        }

        tool_enum = tool_mapping.get(tool_name)
        if not tool_enum:
            return f"Unknown tool: {tool_name}", False

        # Execute the tool
        observation = await self.tool_executor.execute(
            action=tool_enum,
            action_input=action_input,
            file_paths=file_paths,
            session_id=session_id,
            steps=None,
            plan_context=plan_context
        )

        # Trust tool execution result - no separate verification needed
        # Success is determined by whether tool produced output without errors
        success = bool(observation and len(observation.strip()) > 0 and "error" not in observation.lower()[:100])

        return observation, success

    def _build_plan_context(
        self,
        current_step: PlanStep,
        all_plan_steps: List[PlanStep],
        step_results: List[StepResult]
    ) -> dict:
        """
        Build structured plan_context for python_coder prompt.

        Args:
            current_step: The current plan step being executed
            all_plan_steps: List of all plan steps
            step_results: List of completed step results

        Returns:
            Dict with plan context information
        """
        # Build plan structure with status
        plan = []
        for step in all_plan_steps:
            step_dict = {
                'step_number': step.step_num,
                'goal': step.goal,
                'success_criteria': step.success_criteria,
                'primary_tools': step.primary_tools
            }

            # Determine status
            if step.step_num < current_step.step_num:
                # Check if completed successfully
                result = next((r for r in step_results if r.step_num == step.step_num), None)
                if result:
                    step_dict['status'] = 'completed' if result.success else 'failed'
                else:
                    step_dict['status'] = 'completed'
            elif step.step_num == current_step.step_num:
                step_dict['status'] = 'current'
            else:
                step_dict['status'] = 'pending'

            plan.append(step_dict)

        # Build previous results summary
        previous_results = []
        for result in step_results:
            previous_results.append({
                'step_number': result.step_num,
                'summary': result.observation[:200] if result.observation else "",
                'success': result.success
            })

        return {
            'current_step': current_step.step_num,
            'total_steps': len(all_plan_steps),
            'plan': plan,
            'previous_results': previous_results
        }

    def _build_action_input_from_plan(
        self,
        plan_step: PlanStep,
        user_query: str,
        context: str
    ) -> str:
        """
        Build action input directly from plan step (no LLM call needed).

        Args:
            plan_step: Current plan step
            user_query: Original user query
            context: Context from previous steps

        Returns:
            Action input string for the tool
        """
        # Combine plan step goal + context + step-specific instructions
        parts = []

        # Add original user query for context
        parts.append(f"User Query: {user_query}")

        # Add step goal (this is the main instruction)
        parts.append(f"\nTask for this step: {plan_step.goal}")

        # Add step-specific context/instructions if available
        if plan_step.context:
            parts.append(f"\nInstructions: {plan_step.context}")

        # Add context from previous steps
        if context and context.strip() != "This is the first step.":
            parts.append(f"\nPrevious step results:\n{context}")

        return "\n".join(parts)

    async def _generate_final_answer_from_steps(
        self,
        user_query: str,
        step_results: List[StepResult],
        accumulated_observations: List[str]
    ) -> str:
        """
        Generate final answer based on all step results.

        Args:
            user_query: Original user query
            step_results: Results from all steps
            accumulated_observations: All observations

        Returns:
            Final answer string
        """
        # Build context from all steps
        steps_summary = []
        for result in step_results:
            status = "✓" if result.success else "✗"
            steps_summary.append(
                f"Step {result.step_num} {status}: {result.goal}\n"
                f"  Tool: {result.tool_used}\n"
                f"  Result: {result.observation[:300]}"
            )

        steps_text = "\n\n".join(steps_summary)
        observations_text = "\n".join(accumulated_observations)

        prompt = PromptRegistry.get('react_final_answer_from_steps',
                                     user_query=user_query,
                                     steps_text=steps_text,
                                     observations_text=observations_text)

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def select_tool_for_step(self, step: PlanStep) -> str:
        """
        Select the primary tool for a plan step.

        Args:
            step: PlanStep with tool preferences

        Returns:
            Tool name to use (from primary tools)
        """
        if step.primary_tools:
            return step.primary_tools[0]
        else:
            raise ValueError(f"No primary tool specified for step {step.step_num}")
