"""
ReAct Planning Module

Handles structured plan execution with dynamic adaptation.

Consolidated from:
- plan_executor.py
- plan_adapter.py
"""

from typing import List, Tuple, Optional
from langchain_core.messages import HumanMessage
import json

from .models import ToolName
from backend.config.prompts import PromptRegistry
from backend.config.settings import settings
from backend.models.schemas import PlanStep, StepResult
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PlanExecutor:
    """
    Executes structured plans step-by-step with dynamic adaptation.

    Features:
    - Step-by-step plan execution
    - Dynamic plan adaptation when steps fail
    - Context pruning for long executions
    """

    def __init__(
        self,
        tool_executor,
        context_formatter,
        verifier,
        answer_generator,
        llm
    ):
        """
        Initialize plan executor.

        Args:
            tool_executor: ToolExecutor instance
            context_formatter: ContextFormatter instance
            verifier: StepVerifier instance
            answer_generator: AnswerGenerator instance
            llm: Language model instance
        """
        self.tool_executor = tool_executor
        self.context_formatter = context_formatter
        self.verifier = verifier
        self.answer_generator = answer_generator
        self.llm = llm
        self.plan_adapter = PlanAdapter(llm)

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
            max_iterations_per_step: Max iterations per step

        Returns:
            Tuple of (final_answer, step_results)
        """
        logger.info("\n" + "=" * 100)
        logger.info("[PlanExecutor] EXECUTION STARTED")
        logger.info(f"Plan has {len(plan_steps)} steps")
        if file_paths:
            logger.info(f"Attached Files: {len(file_paths)} files")
        logger.info("=" * 100 + "\n")

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

                # Apply pruning if observations grow too large (keep last 5)
                if len(accumulated_observations) > 5:
                    pruned_observations = accumulated_observations[-5:]
                    logger.info(f"[PlanExecutor] Context pruned: keeping last 5 observations (removed {len(accumulated_observations) - 5})")
                    accumulated_observations = pruned_observations

            logger.info(f"\n[PlanExecutor] Step {plan_step.step_num} {'✓ SUCCESS' if step_result.success else '✗ FAILED'}")
            logger.info(f"Tool used: {step_result.tool_used}")
            logger.info(f"Attempts: {step_result.attempts}\n")

            # Check if re-planning is needed (skip if last step)
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

                    # Renumber adapted plan steps
                    next_step_num = plan_step.step_num + 1
                    for i, step in enumerate(adapted_plan):
                        step.step_num = next_step_num + i

                    # Update plan_steps
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
        Execute a single plan step with retry mechanism.

        Args:
            plan_step: The plan step to execute
            user_query: Original user query
            accumulated_observations: Context from previous steps
            file_paths: Optional file paths
            session_id: Optional session ID
            all_plan_steps: Optional list of all plan steps
            step_results: Optional list of completed step results

        Returns:
            StepResult with execution details
        """
        logger.info(f"[PlanExecutor Step {plan_step.step_num}] Starting execution...")

        # Build context from previous steps
        context = "\n".join(accumulated_observations) if accumulated_observations else "This is the first step."

        # Use only the primary tool
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

        max_retries = settings.react_step_max_retries
        logger.info(f"[PlanExecutor Step {plan_step.step_num}] Executing with primary tool '{tool_name}' (max retries: {max_retries})")

        # Retry loop
        last_observation = ""
        last_error = None
        attempts = 0

        for attempt in range(1, max_retries + 1):
            attempts = attempt
            logger.info(f"[PlanExecutor Step {plan_step.step_num}] Attempt {attempt}/{max_retries}")

            try:
                observation, success = await self._execute_tool_for_step(
                    tool_name=tool_name,
                    plan_step=plan_step,
                    user_query=user_query,
                    context=context,
                    file_paths=file_paths,
                    session_id=session_id,
                    all_plan_steps=all_plan_steps,
                    step_results=step_results,
                    previous_attempt_context=last_observation if attempt > 1 else None
                )

                last_observation = observation

                if success:
                    logger.info(f"[PlanExecutor Step {plan_step.step_num}] SUCCESS on attempt {attempt}/{max_retries}")
                    return StepResult(
                        step_num=plan_step.step_num,
                        goal=plan_step.goal,
                        success=True,
                        tool_used=tool_name,
                        attempts=attempt,
                        observation=observation,
                        error=None,
                        metadata={"tool_type": "primary", "succeeded_on_attempt": attempt}
                    )
                else:
                    logger.warning(f"[PlanExecutor Step {plan_step.step_num}] FAILED on attempt {attempt}/{max_retries}")
                    if attempt < max_retries:
                        logger.info(f"[PlanExecutor Step {plan_step.step_num}] Retrying...")

            except Exception as e:
                last_error = str(e)
                logger.error(f"[PlanExecutor Step {plan_step.step_num}] Exception on attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    logger.info(f"[PlanExecutor Step {plan_step.step_num}] Retrying after exception...")

        # All attempts exhausted
        logger.error(f"[PlanExecutor Step {plan_step.step_num}] FAILED after {attempts} attempts")
        return StepResult(
            step_num=plan_step.step_num,
            goal=plan_step.goal,
            success=False,
            tool_used=tool_name,
            attempts=attempts,
            observation=last_observation or f"Error executing {tool_name}: {last_error}",
            error=last_error or "Tool did not achieve goal after all attempts",
            metadata={"tool_type": "primary", "all_attempts_failed": True}
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
        step_results: Optional[List[StepResult]] = None,
        previous_attempt_context: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Execute a specific tool to achieve the step goal.

        Args:
            previous_attempt_context: Observation from previous failed attempt (for retry context)

        Returns:
            Tuple of (observation, success)
        """
        # Build action input from plan step
        action_input = self._build_action_input_from_plan(
            plan_step=plan_step,
            user_query=user_query,
            context=context,
            previous_attempt_context=previous_attempt_context
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

        # Determine success
        success = bool(observation and len(observation.strip()) > 0 and "error" not in observation.lower()[:100])

        return observation, success

    def _build_plan_context(
        self,
        current_step: PlanStep,
        all_plan_steps: List[PlanStep],
        step_results: List[StepResult]
    ) -> dict:
        """Build structured plan_context for python_coder prompt."""
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
                'summary': result.observation if result.observation else "",
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
        context: str,
        previous_attempt_context: Optional[str] = None
    ) -> str:
        """Build action input directly from plan step with structured guidance."""
        parts = []

        # Add original user query
        parts.append(f"User Query: {user_query}")

        # Add step goal
        parts.append(f"\nTask for this step: {plan_step.goal}")

        # Add retry context if this is a retry attempt
        if previous_attempt_context:
            parts.append("\n⚠️ RETRY ATTEMPT - Previous attempt failed:")
            parts.append(f"{previous_attempt_context}")
            parts.append("\nPlease adjust your approach to address the issues from the previous attempt.")

        # Add structured execution guidance
        if plan_step.context or plan_step.success_criteria:
            parts.append("\nExecution Guidance:")
            if plan_step.context:
                parts.append(f"  - Approach: {plan_step.context}")
            if plan_step.success_criteria:
                parts.append(f"  - Success Criteria: {plan_step.success_criteria}")

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
        """Generate final answer based on all step results."""
        # Build context from all steps
        steps_summary = []
        for result in step_results:
            status = "✓" if result.success else "✗"
            steps_summary.append(
                f"Step {result.step_num} {status}: {result.goal}\n"
                f"  Tool: {result.tool_used}\n"
                f"  Result: {result.observation}"
            )

        steps_text = "\n\n".join(steps_summary)
        observations_text = "\n".join(accumulated_observations)

        prompt = PromptRegistry.get('react_final_answer_from_steps',
                                     user_query=user_query,
                                     steps_text=steps_text,
                                     observations_text=observations_text)

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()


class PlanAdapter:
    """
    Handles dynamic plan adaptation during execution.

    Determines when re-planning is needed and generates adapted plans.
    """

    def __init__(self, llm):
        """
        Initialize PlanAdapter.

        Args:
            llm: LangChain LLM instance
        """
        self.llm = llm

    async def should_replan(
        self,
        current_step_result: StepResult,
        remaining_steps: List[PlanStep],
        original_query: str
    ) -> Tuple[bool, str]:
        """
        Determine if re-planning is needed.

        Uses multiple triggers:
        1. Step failed after max attempts
        2. Observation indicates additional requirements
        3. LLM-based assessment of plan viability

        Returns:
            Tuple of (needs_replan, reason)
        """
        # Trigger 1: Step failed after multiple attempts
        if not current_step_result.success and current_step_result.attempts >= 2:
            logger.warning(f"[PlanAdapter] Trigger 1: Step failed after {current_step_result.attempts} attempts")
            return True, f"Step '{current_step_result.goal}' failed after {current_step_result.attempts} attempts"

        # Trigger 3: LLM-based viability assessment
        if remaining_steps and len(remaining_steps) > 0:
            try:
                prompt = f"""Assess if the current plan should continue or needs re-planning.

Original Query: {original_query}

Current Step: {current_step_result.goal}
Success: {current_step_result.success}
Tool Used: {current_step_result.tool_used}
Attempts: {current_step_result.attempts}

Observation:
{current_step_result.observation}

Remaining Steps ({len(remaining_steps)}):
{self._format_remaining_steps(remaining_steps)}

Question: Based on the current step result, should we:
A) Continue with the current plan as-is
B) Re-plan to address issues or new requirements

Respond with JSON:
{{
  "continue_plan": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation"
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

                continue_plan = result.get("continue_plan", True)
                confidence = float(result.get("confidence", 0.5))
                reason = result.get("reason", "No reason provided")

                # Require high confidence to trigger re-planning
                if not continue_plan and confidence >= 0.7:
                    logger.warning(f"[PlanAdapter] Trigger 3: LLM recommends re-planning (confidence: {confidence:.2f})")
                    return True, f"LLM assessment: {reason}"

                logger.info(f"[PlanAdapter] LLM recommends continuing (confidence: {confidence:.2f})")

            except Exception as e:
                logger.warning(f"[PlanAdapter] LLM assessment failed: {e}, assuming continue")

        return False, "Plan execution on track"

    async def adapt_plan(
        self,
        original_plan: List[PlanStep],
        completed_steps: List[StepResult],
        current_step_result: StepResult,
        remaining_steps: List[PlanStep],
        original_query: str
    ) -> List[PlanStep]:
        """
        Generate adapted plan that preserves completed work.

        Returns:
            List of new PlanStep objects
        """
        # Build context from completed work
        completed_context = self._build_completed_context(completed_steps)

        # Build current issue description
        current_issue = self._describe_current_issue(current_step_result)

        prompt = f"""Generate an adapted execution plan that addresses current issues.

Original Query: {original_query}

COMPLETED WORK (preserve and build on this):
{completed_context}

CURRENT ISSUE:
{current_issue}

ORIGINAL REMAINING STEPS (for reference):
{self._format_plan_steps(remaining_steps)}

Task: Generate a NEW plan that:
1. BUILDS ON completed work (don't repeat successful steps)
2. ADDRESSES the current issue or blocker
3. ACHIEVES the original query goal
4. Has concrete, actionable steps with clear success criteria

Format: Respond with JSON array of plan steps:
[
  {{
    "step_num": 1,
    "goal": "specific goal description",
    "primary_tools": ["tool1", "tool2"],
    "success_criteria": "how to verify this step succeeded",
    "context": "relevant context for this step"
  }},
  ...
]

Available tools: web_search, rag_retrieval, python_coder
"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            # Parse JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            new_plan_data = json.loads(response_text.strip())

            # Convert to PlanStep objects
            adapted_plan = []
            for step_data in new_plan_data:
                plan_step = PlanStep(
                    step_num=step_data.get("step_num", len(adapted_plan) + 1),
                    goal=step_data.get("goal", ""),
                    primary_tools=step_data.get("primary_tools", []),
                    success_criteria=step_data.get("success_criteria", ""),
                    context=step_data.get("context")
                )
                adapted_plan.append(plan_step)

            logger.info(f"[PlanAdapter] Generated adapted plan with {len(adapted_plan)} steps:")
            for i, step in enumerate(adapted_plan, 1):
                logger.info(f"[PlanAdapter]   {i}. {step.goal} (tools: {step.primary_tools})")

            return adapted_plan

        except Exception as e:
            logger.error(f"[PlanAdapter] Failed to generate adapted plan: {e}")
            raise

    def _format_plan_steps(self, steps: List[PlanStep]) -> str:
        """Format plan steps for display."""
        if not steps:
            return "No remaining steps"

        lines = []
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step.goal}")
            lines.append(f"   Primary tools: {', '.join(step.primary_tools)}")
            lines.append(f"   Success criteria: {step.success_criteria}")
            lines.append("")

        return "\n".join(lines)

    def _format_remaining_steps(self, steps: List[PlanStep]) -> str:
        """Simplified format for remaining steps."""
        if not steps:
            return "No remaining steps"

        return "\n".join([f"  - {step.goal}" for step in steps])

    def _build_completed_context(self, completed_steps: List[StepResult]) -> str:
        """Build summary of completed work."""
        if not completed_steps:
            return "No steps completed yet"

        lines = []
        for step in completed_steps:
            status = "SUCCESS" if step.success else "FAILED"
            lines.append(f"Step {step.step_num}: {step.goal} - {status}")
            if step.tool_used:
                lines.append(f"  Tool used: {step.tool_used}")
            if step.observation:
                lines.append(f"  Result: {step.observation}")
            lines.append("")

        return "\n".join(lines)

    def _describe_current_issue(self, step_result: StepResult) -> str:
        """Describe the current issue/blocker."""
        lines = []
        lines.append(f"Step {step_result.step_num}: {step_result.goal}")
        lines.append(f"Status: {'SUCCESS' if step_result.success else 'FAILED'}")
        lines.append(f"Tool used: {step_result.tool_used}")
        lines.append(f"Attempts: {step_result.attempts}")

        if step_result.error:
            lines.append(f"Error: {step_result.error}")

        if step_result.observation:
            lines.append(f"Observation: {step_result.observation}")

        return "\n".join(lines)
