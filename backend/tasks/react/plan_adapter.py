"""
Plan Adapter Module for Dynamic Plan Adaptation

This module provides intelligent plan adaptation capabilities that allow
Plan-Execute workflows to adjust their execution strategy when steps fail
or when observations reveal new requirements.

Key Features:
- Re-planning trigger detection (when to adapt)
- Partial plan preservation (don't lose completed work)
- Context-aware plan generation (learns from failures)
- Multi-trigger strategy (step failures, observations, LLM assessment)

Architecture:
- PlanAdapter: Main class that orchestrates adaptation workflow
- Trigger detection: Multiple heuristics + LLM-based assessment
- Plan generation: Builds on completed work, addresses current issues
"""

from typing import List, Optional, Tuple
from langchain_core.messages import HumanMessage
import json

from backend.models.schemas import PlanStep, StepResult
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PlanAdapter:
    """
    Handles dynamic plan adaptation during Plan-Execute execution.

    This class determines when re-planning is needed and generates adapted
    plans that preserve completed work while addressing current issues.
    """

    def __init__(self, llm):
        """
        Initialize PlanAdapter.

        Args:
            llm: LangChain LLM instance for plan generation
        """
        self.llm = llm

    async def should_replan(
        self,
        current_step_result: StepResult,
        remaining_steps: List[PlanStep],
        original_query: str
    ) -> Tuple[bool, str]:
        """
        Determine if re-planning is needed based on current execution state.

        This method uses multiple triggers to detect when the current plan
        needs adaptation:
        1. Step failed after max attempts with all tools
        2. Observation indicates additional requirements
        3. LLM-based assessment of plan viability

        Args:
            current_step_result: Result from most recent step
            remaining_steps: Steps still to be executed
            original_query: User's original query

        Returns:
            Tuple of (needs_replan, reason)
        """
        # Trigger 1: Step failed after multiple attempts
        if not current_step_result.success and current_step_result.attempts >= 3:
            logger.warning(f"[PlanAdapter] Trigger 1: Step failed after {current_step_result.attempts} attempts")
            return True, f"Step '{current_step_result.goal}' failed after {current_step_result.attempts} attempts"

        # Trigger 2: Observation reveals additional work needed
        observation_lower = current_step_result.observation.lower() if current_step_result.observation else ""
        replan_keywords = [
            "additional analysis needed",
            "more information required",
            "cannot complete without",
            "missing prerequisite",
            "unexpected result",
            "requires further"
        ]

        for keyword in replan_keywords:
            if keyword in observation_lower:
                logger.warning(f"[PlanAdapter] Trigger 2: Observation indicates '{keyword}'")
                return True, f"Observation indicates additional work: '{keyword}'"

        # Trigger 3: LLM-based viability assessment (only if we have remaining steps)
        if remaining_steps and len(remaining_steps) > 0:
            try:
                prompt = f"""Assess if the current plan should continue or needs re-planning.

Original Query: {original_query}

Current Step: {current_step_result.goal}
Success: {current_step_result.success}
Tool Used: {current_step_result.tool_used}
Attempts: {current_step_result.attempts}

Observation:
{current_step_result.observation[:500]}

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

                # Parse JSON response
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

        # No re-planning triggers activated
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

        This method creates a new plan that:
        1. Acknowledges and builds on completed work
        2. Addresses the current issue or blocker
        3. Achieves the original query goal

        Args:
            original_plan: Original plan steps
            completed_steps: Steps that have been completed
            current_step_result: Most recent step result
            remaining_steps: Original remaining steps
            original_query: User's original query

        Returns:
            List of new PlanStep objects (adapted plan)
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

Available tools: web_search, rag_retrieval, python_coder, file_analyzer
"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            # Parse JSON response
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
        """Format plan steps for display in prompts."""
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
                obs_preview = step.observation[:200]
                lines.append(f"  Result: {obs_preview}...")
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
            lines.append(f"Observation: {step_result.observation[:300]}...")

        return "\n".join(lines)
