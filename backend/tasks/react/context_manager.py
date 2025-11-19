"""
ReAct Context Manager

This module provides context formatting and building functionality for the ReAct agent,
including context pruning optimization and plan-step context building.

Consolidated from context_builder.py and React.py for improved modularity.

Enhanced Features (v2.0):
- Relevance-based notepad context injection
- Smart entry filtering and ranking
- Variable usage tracking
"""

from typing import List, Optional, Dict, Any
import re
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

    def build_tool_context(self, steps: List[ReActStep], user_query: Optional[str] = None) -> str:
        """
        Build context string for tool execution (format with pruning).

        This is the main method for formatting step history for LLM consumption.
        Implements context pruning optimization:
        - If ≤3 steps: send all steps in full detail
        - If >3 steps: send summary of early steps + last 2 steps in detail

        Args:
            steps: List of ReActStep objects from execution history
            user_query: Current user query (for relevance-based notepad filtering)

        Returns:
            Formatted context string for LLM consumption
        """
        context_parts = []

        # Inject notepad context at the top (with relevance filtering)
        notepad_context = self._build_notepad_context(user_query=user_query, max_entries=5)
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

    def _build_notepad_context(self, user_query: Optional[str] = None, max_entries: int = 5) -> str:
        """
        Build notepad context string with relevance-based filtering.

        Instead of including ALL entries, this method:
        1. Ranks entries by relevance to current query
        2. Includes only the top N most relevant entries
        3. Always includes variable metadata

        Args:
            user_query: Current user query (for relevance ranking)
            max_entries: Maximum number of entries to include

        Returns:
            Formatted notepad context string with relevant entries and variables
        """
        if not self.notepad:
            return ""

        try:
            logger_inst = get_logger(__name__)

            # Get relevant entries if query provided, otherwise get latest
            if user_query:
                relevant_entries = self._get_relevant_entries(user_query, max_entries)
            else:
                # No query - just get latest entries
                all_entries = [e for e in self.notepad.data["entries"] if not e.get("is_archived", False)]
                relevant_entries = all_entries[-max_entries:] if all_entries else []

            if not relevant_entries and not self.variable_metadata:
                return ""

            lines = []

            # Session memory header
            if relevant_entries:
                lines.append("=== Session Memory (Notepad) ===")
                lines.append(f"Showing {len(relevant_entries)} most relevant previous task(s):")
                lines.append("")

                # Add relevant entries
                for entry in relevant_entries:
                    entry_id = entry["entry_id"]
                    task = entry["task"]
                    description = entry["description"]
                    code_file = entry.get("code_file", "")
                    variables = entry.get("variables_saved", [])
                    outputs = entry.get("key_outputs", "")

                    # Entry header
                    code_ref = f" - {code_file}" if code_file else ""
                    lines.append(f"Entry {entry_id}: [{task}]{code_ref}")
                    lines.append(f"Description: {description}")

                    if variables:
                        lines.append(f"Variables saved: {', '.join(variables)}")

                    if outputs:
                        lines.append(f"Outputs: {outputs}")

                    lines.append("")  # Blank line between entries

            # Variable metadata section
            if self.variable_metadata:
                lines.append("=== Available Saved Variables ===")
                lines.append("You can load these variables from previous executions:")
                lines.append("")

                for idx, (var_name, meta) in enumerate(self.variable_metadata.items(), 1):
                    var_type = meta.get("type", "unknown")
                    load_code = meta.get("load_code", "")

                    lines.append(f"{idx}. {var_name} ({var_type})")

                    # Add type-specific details
                    if var_type == "pandas.DataFrame":
                        shape = meta.get("shape", [])
                        columns = meta.get("columns", [])
                        lines.append(f"   Shape: {shape}, Columns: {', '.join(columns[:5])}")
                    elif var_type == "numpy.ndarray":
                        shape = meta.get("shape", [])
                        dtype = meta.get("dtype", "")
                        lines.append(f"   Shape: {shape}, dtype: {dtype}")
                    elif var_type == "dict":
                        keys = meta.get("keys", [])
                        lines.append(f"   Keys: {', '.join(str(k) for k in keys)}")
                    elif var_type == "list":
                        length = meta.get("length", 0)
                        lines.append(f"   Length: {length}")

                    # Add load code
                    if load_code:
                        lines.append(f"   Load with: {load_code}")

                    lines.append("")  # Blank line between variables

            return "\n".join(lines)

        except Exception as e:
            logger_inst = get_logger(__name__)
            logger_inst.warning(f"[ContextManager] Failed to build notepad context: {e}")
            return ""

    def _get_relevant_entries(self, query: str, max_entries: int) -> List[Dict[str, Any]]:
        """
        Get most relevant notepad entries based on query.

        Uses keyword-based relevance scoring:
        - Matches in description, task, outputs
        - Considers variable overlap
        - Prefers recent entries when scores are equal

        Args:
            query: User's query
            max_entries: Maximum number of entries to return

        Returns:
            List of relevant entries, sorted by relevance
        """
        try:
            all_entries = [e for e in self.notepad.data["entries"] if not e.get("is_archived", False)]

            if not all_entries:
                return []

            query_lower = query.lower()
            query_words = set(re.findall(r'\b\w+\b', query_lower))

            # Score each entry
            scored_entries = []
            for entry in all_entries:
                score = self._score_entry_relevance(entry, query_lower, query_words)
                scored_entries.append((score, entry))

            # Sort by score (descending), then by recency
            scored_entries.sort(key=lambda x: (x[0], x[1]["entry_id"]), reverse=True)

            # Return top N
            return [entry for score, entry in scored_entries[:max_entries]]

        except Exception as e:
            logger_inst = get_logger(__name__)
            logger_inst.warning(f"[ContextManager] Failed to get relevant entries: {e}")
            return []

    def _score_entry_relevance(self, entry: Dict[str, Any], query_lower: str, query_words: set) -> float:
        """
        Score an entry's relevance to a query.

        Args:
            entry: Notepad entry
            query_lower: Lowercased query
            query_words: Set of words in query

        Returns:
            Relevance score (higher is better)
        """
        score = 0.0

        # Check description (weighted most heavily)
        description = entry.get("description", "").lower()
        if query_lower in description:
            score += 10.0

        # Check task category
        task = entry.get("task", "").lower()
        if query_lower in task:
            score += 5.0

        # Check outputs
        outputs = entry.get("key_outputs", "").lower()
        if query_lower in outputs:
            score += 3.0

        # Check word overlap
        entry_text = f"{description} {task} {outputs}"
        entry_words = set(re.findall(r'\b\w+\b', entry_text))
        word_overlap = len(query_words & entry_words)
        score += word_overlap * 0.5

        # Check variable relevance
        entry_vars = entry.get("variables_saved", [])
        for var in entry_vars:
            if var.lower() in query_lower:
                score += 3.0

        # Check tags
        tags = entry.get("tags", [])
        for tag in tags:
            if tag.lower() in query_lower:
                score += 2.0

        # Bonus for recent entries (recency bias)
        entry_id = entry.get("entry_id", 0)
        score += entry_id * 0.1

        return score
