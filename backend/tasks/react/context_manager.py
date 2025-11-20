"""
ReAct Context Manager

This module provides context formatting and building functionality for the ReAct agent,
including context pruning optimization and plan-step context building.

Consolidated from context_builder.py and React.py for improved modularity.
"""

from typing import List, Optional
from .models import ReActStep, ToolName
from backend.models.schemas import PlanStep
from backend.utils.logging_utils import get_logger


class ContextManager:
    """
    Handles context formatting and building for ReAct agent execution.

    Responsibilities:
    - Format step history with context pruning for efficiency
    - Build enhanced context for plan-execute mode
    - Manage code history integration for python_coder
    - Build file context for file-related operations

    Key optimizations:
    - Context pruning: Summarizes early steps when count > 3
    - Code history integration: Shows previous code versions in context
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize context manager.

        Args:
            session_id: Optional session ID for code history retrieval
        """
        self.session_id = session_id
        self.notepad = None
        self.variable_metadata = None

    def set_notepad(self, notepad, variable_metadata: Optional[dict] = None):
        """
        Set notepad and variable metadata for context injection.

        Args:
            notepad: SessionNotepad instance
            variable_metadata: Optional variable metadata dictionary
        """
        self.notepad = notepad
        self.variable_metadata = variable_metadata

    def build_tool_context(self, steps: List[ReActStep]) -> str:
        """
        Build context string for tool execution (format with pruning).

        This is the main method for formatting step history for LLM consumption.
        Implements context pruning optimization:
        - If ≤3 steps: send all steps in full detail
        - If >3 steps: send summary of early steps + last 2 steps in detail

        Args:
            steps: List of ReActStep objects from execution history

        Returns:
            Formatted context string for LLM consumption
        """
        context_parts = []

        # Inject notepad context at the top
        notepad_context = self._build_notepad_context()
        if notepad_context:
            context_parts.append(notepad_context)
            context_parts.append("")  # Blank line separator

        # Build step history context
        if not steps:
            return "\n".join(context_parts) if context_parts else ""

        # If few steps, return all details
        if len(steps) <= 3:
            context_parts.append(self._format_all_steps(steps))
        else:
            # Context pruning: summary + recent steps
            context_parts.append(self._format_pruned_steps(steps))

        return "\n".join(context_parts)

    def prune_context(self, steps: List[ReActStep], keep_last_n: int = 3) -> List[ReActStep]:
        """
        Simplified context pruning: keep last N steps, discard older ones.

        No LLM summarization - just a sliding window approach.
        This method can be used when you need a pruned list of steps
        rather than a formatted string.

        Args:
            steps: Full list of ReActStep objects
            keep_last_n: Number of recent steps to keep (default: 3)

        Returns:
            Pruned list containing recent N steps only
        """
        if len(steps) <= keep_last_n:
            return steps

        # Keep only last N steps
        pruned_steps = steps[-keep_last_n:]

        # Log pruning action
        logger = get_logger(__name__)
        logger.info(f"Context pruned: kept last {keep_last_n} steps, discarded {len(steps) - keep_last_n} older steps")

        return pruned_steps

    def build_plan_context(
        self,
        step_goal: str,
        step_criteria: str,
        history: str
    ) -> str:
        """
        Build simple context for plan execution.

        Args:
            step_goal: Goal of current plan step
            step_criteria: Success criteria for the step
            history: Context from previous steps

        Returns:
            Formatted context string for plan execution
        """
        context_parts = []

        context_parts.append(f"=== Current Plan Step ===")
        context_parts.append(f"Goal: {step_goal}")
        context_parts.append(f"Success Criteria: {step_criteria}")
        context_parts.append("")

        if history and history != "This is the first step.":
            context_parts.append("=== Previous Steps ===")
            context_parts.append(history)
            context_parts.append("")

        return "\n".join(context_parts)

    def build_file_context(self, file_paths: List[str]) -> str:
        """
        Build context string describing attached files.

        Args:
            file_paths: List of file paths

        Returns:
            Formatted file context string
        """
        if not file_paths:
            return ""

        context_parts = [f"=== Attached Files ({len(file_paths)}) ==="]
        for idx, path in enumerate(file_paths, 1):
            context_parts.append(f"{idx}. {path}")
        context_parts.append("")

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

            context_parts.append("You can reference and build upon these previous code versions.")
            context_parts.append("")

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
            context_parts.append(f"""
Step {step.step_num}:
- Thought: {step.thought}
- Action: {step.action}
- Action Input: {step.action_input}
- Observation: {step.observation}
""")
        return "\n".join(context_parts)

    def _format_pruned_steps(self, steps: List[ReActStep]) -> str:
        """
        Format steps with pruning (summary of early + full detail of recent).

        Args:
            steps: List of ReActStep objects

        Returns:
            Formatted context string with pruned steps
        """
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

    def _get_code_history(self, max_versions: int = 3) -> Optional[List[dict]]:
        """
        Retrieve previous code history from python_coder tool.

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

    def _build_notepad_context(self) -> str:
        """
        Build notepad context string for injection into agent context.

        Returns:
            Formatted notepad context string with all entries and variables
        """
        if not self.notepad:
            return ""

        try:
            # Use the notepad's get_full_context method
            return self.notepad.get_full_context(self.variable_metadata)
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"[ContextManager] Failed to build notepad context: {e}")
            return ""

    def format_phase_handoff(
        self,
        phase_name: str,
        previous_findings: str,
        current_task: str,
        files_as_fallback: bool = True
    ) -> str:
        """
        Format context for multi-phase workflows with explicit handoffs.

        This method enables conversation context reuse by instructing the agent
        to prioritize information from previous phases over re-processing files.

        Args:
            phase_name: Name of the current phase (e.g., "Visualization Generation")
            previous_findings: Summary of findings from previous phase(s)
            current_task: Description of the current task
            files_as_fallback: Whether to include files-as-fallback instruction

        Returns:
            Formatted prompt for phase handoff

        Example:
            >>> context_mgr = ContextManager()
            >>> prompt = context_mgr.format_phase_handoff(
            ...     phase_name="Visualization",
            ...     previous_findings="Analyzed 100 files, found 10 outliers...",
            ...     current_task="Create charts based on the analysis"
            ... )
        """
        prompt_parts = [
            f"**PRIORITY: Use your {phase_name} findings first.**\n",
            "**Previous Analysis:**",
            previous_findings,
            "",
            "**Current Task:**",
            current_task,
            ""
        ]

        if files_as_fallback:
            prompt_parts.extend([
                "**IMPORTANT:** The attached files are ONLY for reference if you need to verify specific values.",
                "Your primary data source should be what you already calculated in previous phases.",
                "DO NOT re-analyze the raw data files from scratch.",
                ""
            ])

        return "\n".join(prompt_parts)
