"""
ReAct Context Builder

This module provides context formatting functionality for the ReAct agent,
including context pruning optimization and plan-step context building.

Extracted from React.py to improve modularity and maintainability.
"""

from typing import List, Optional
from backend.tasks.react.models import ReActStep, ToolName
from backend.models.schemas import PlanStep


class ContextBuilder:
    """
    Handles context formatting for ReAct agent execution.

    Responsibilities:
    - Format step history with context pruning for efficiency
    - Build enhanced context for plan-execute mode
    - Manage code history integration for python_coder

    Key optimizations:
    - Context pruning: Summarizes early steps when count > 3
    - Code history integration: Shows previous code versions in context
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize context builder.

        Args:
            session_id: Optional session ID for code history retrieval
        """
        self.session_id = session_id

    def format_steps_context(self, steps: List[ReActStep]) -> str:
        """
        PERFORMANCE OPTIMIZATION: Format previous steps with context pruning

        - If ≤3 steps: send all steps in full detail
        - If >3 steps: send summary of early steps + last 2 steps in detail

        This reduces context size and speeds up LLM processing.

        Args:
            steps: List of ReActStep objects from execution history

        Returns:
            Formatted context string for LLM consumption
        """
        if not steps:
            return ""

        # If few steps, return all details
        if len(steps) <= 3:
            return self._format_all_steps(steps)

        # Context pruning: summary + recent steps
        context_parts = ["Previous Steps:\n"]

        # Summary of early steps
        early_steps = steps[:-2]
        tools_used = list(set([s.action for s in early_steps if s.action != ToolName.FINISH]))
        summary = f"Steps 1-{len(early_steps)} completed using: {', '.join(tools_used)}"
        context_parts.append(f"[Summary] {summary}\n")

        # Recent steps in full detail
        context_parts.append("\n[Recent Steps - Full Detail]")
        recent_steps = steps[-2:]
        for step in recent_steps:
            obs_display = step.observation[:500] if len(step.observation) > 500 else step.observation
            context_parts.append(f"""
Step {step.step_num}:
- Thought: {step.thought[:200]}...
- Action: {step.action}
- Action Input: {step.action_input[:200] if len(step.action_input) > 200 else step.action_input}
- Observation: {obs_display}
""")

        return "\n".join(context_parts)

    def _format_all_steps(self, steps: List[ReActStep]) -> str:
        """
        Format all steps in full detail (used when step count is low).

        Args:
            steps: List of ReActStep objects

        Returns:
            Formatted context string with all steps in full detail
        """
        context_parts = ["Previous Steps:"]
        for step in steps:
            obs_display = step.observation[:]
            context_parts.append(f"""
Step {step.step_num}:
- Thought: {step.thought}
- Action: {step.action}
- Action Input: {step.action_input}
- Observation: {obs_display}
""")
        return "\n".join(context_parts)

    def build_plan_step_context(
        self,
        plan_step: PlanStep,
        previous_steps_context: str
    ) -> str:
        """
        Build enhanced context for python_coder when executing in Plan-Execute mode.

        Integrates:
        - Previous code versions from earlier stages
        - Current step goal and success criteria
        - Results from previous plan steps

        Args:
            plan_step: Current plan step being executed
            previous_steps_context: Context from previous plan steps

        Returns:
            Formatted context string for python_coder with plan execution details
        """
        context_parts = []

        # Load previous code history if available
        code_history = self._get_code_history()
        if code_history:
            context_parts.append("=== Previous Code Versions from Earlier Stages ===\n")
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

            context_parts.append("You can reference and build upon these previous code versions from earlier stages.")
            context_parts.append("Use them as a starting point to avoid repeating work.\n")

        context_parts.append("=== Plan-Execute Mode Context ===\n")

        # Current step information
        context_parts.append(f"Current Step {plan_step.step_num}:")
        context_parts.append(f"  Goal: {plan_step.goal}")
        context_parts.append(f"  Success Criteria: {plan_step.success_criteria}")
        if plan_step.context:
            context_parts.append(f"  Additional Context: {plan_step.context}")
        context_parts.append("")

        # Previous steps results
        if previous_steps_context and previous_steps_context != "This is the first step.":
            context_parts.append("Previous Steps Results:")
            context_parts.append(previous_steps_context)
            context_parts.append("")

        context_parts.append("Use this context to:")
        context_parts.append("- Align code generation with the current step's goal")
        context_parts.append("- Build upon results from previous steps")
        context_parts.append("- Ensure success criteria are met")
        if code_history:
            context_parts.append("- Reference the previous code versions shown above from earlier stages")

        return "\n".join(context_parts)

    def _get_code_history(self, max_versions: int = 3) -> Optional[List[dict]]:
        """
        Retrieve previous code history from python_coder tool.

        This method is separated to allow for dependency injection and testing.

        Args:
            max_versions: Maximum number of code versions to retrieve

        Returns:
            List of code history entries, or None if unavailable
        """
        if not self.session_id:
            return None

        try:
            # Import here to avoid circular dependencies
            from backend.tools.python_coder import python_coder_tool
            return python_coder_tool.get_previous_code_history(self.session_id, max_versions=max_versions)
        except Exception:
            # If python_coder_tool is unavailable or errors, return None
            return None
