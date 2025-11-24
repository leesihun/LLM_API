"""
Context Formatting Prompts
Prompts for formatting various types of context for LLM consumption.

These prompts help structure context from different sources (files, steps, notepad, etc.)
for optimal agent understanding and execution.
"""

from typing import List, Optional, Dict, Any


def get_file_context_summary_prompt(
    file_paths: List[str],
    file_metadata: List[Dict[str, Any]]
) -> str:
    """
    Format file context for agent consumption.

    Args:
        file_paths: List of file paths
        file_metadata: List of metadata dictionaries for each file

    Returns:
        Formatted file context string

    Example:
        >>> metadata = [{"filename": "data.csv", "rows": 1000, "columns": 5}]
        >>> prompt = get_file_context_summary_prompt(
        ...     file_paths=["/path/to/data.csv"],
        ...     file_metadata=metadata
        ... )
    """
    if not file_paths:
        return ""

    context_parts = [f"=== Attached Files ({len(file_paths)}) ==="]

    for idx, (path, metadata) in enumerate(zip(file_paths, file_metadata), 1):
        filename = metadata.get('filename', path.split('/')[-1])
        context_parts.append(f"\n{idx}. {filename}")

        # Add metadata details
        if 'rows' in metadata:
            context_parts.append(f"   - Rows: {metadata['rows']}")
        if 'columns' in metadata:
            columns = metadata['columns']
            if isinstance(columns, list):
                context_parts.append(f"   - Columns ({len(columns)}): {', '.join(columns[:5])}")
            else:
                context_parts.append(f"   - Columns: {columns}")
        if 'size' in metadata:
            context_parts.append(f"   - Size: {metadata['size']}")
        if 'type' in metadata:
            context_parts.append(f"   - Type: {metadata['type']}")

    context_parts.append("")
    return "\n".join(context_parts)


def get_step_history_summary_prompt(
    steps: List[Dict[str, Any]],
    include_full_observations: bool = False,
    max_observation_length: int = 500
) -> str:
    """
    Format step execution history for context.

    Args:
        steps: List of step dictionaries with thought, action, observation
        include_full_observations: Whether to include full observations
        max_observation_length: Maximum length for observations if truncated

    Returns:
        Formatted step history string

    Example:
        >>> steps = [{"step_num": 1, "thought": "...", "action": "web_search", ...}]
        >>> prompt = get_step_history_summary_prompt(steps)
    """
    if not steps:
        return "No previous steps."

    context_parts = ["=== Previous Steps ==="]

    for step in steps:
        step_num = step.get('step_num', '?')
        thought = step.get('thought', 'N/A')
        action = step.get('action', 'N/A')
        action_input = step.get('action_input', 'N/A')
        observation = step.get('observation', 'N/A')

        # Truncate long observations unless full detail requested
        if not include_full_observations and len(observation) > max_observation_length:
            observation = observation[:max_observation_length] + "..."

        context_parts.append(f"""
Step {step_num}:
- Thought: {thought}
- Action: {action}
- Action Input: {action_input}
- Observation: {observation}
""")

    return "\n".join(context_parts)


def get_notepad_context_summary_prompt(
    notepad_entries: List[Dict[str, Any]],
    variable_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format notepad entries for context injection.

    Args:
        notepad_entries: List of notepad entry dictionaries
        variable_metadata: Optional metadata about saved variables

    Returns:
        Formatted notepad context string

    Example:
        >>> entries = [{"task": "data_analysis", "description": "...", ...}]
        >>> prompt = get_notepad_context_summary_prompt(entries)
    """
    if not notepad_entries:
        return ""

    context_parts = ["=== Session Memory (Notepad) ==="]
    context_parts.append("Previous work in this session:\n")

    for idx, entry in enumerate(notepad_entries, 1):
        task = entry.get('task', 'unknown')
        description = entry.get('description', 'No description')
        key_outputs = entry.get('key_outputs', '')

        context_parts.append(f"{idx}. [{task}] {description}")
        if key_outputs:
            context_parts.append(f"   Outputs: {key_outputs}")

        # Add code summary if available
        code_summary = entry.get('code_summary', '')
        if code_summary:
            context_parts.append(f"   Code: {code_summary}")

    # Add variable metadata if available
    if variable_metadata:
        context_parts.append("\n=== Available Variables ===")
        for var_name, var_info in variable_metadata.items():
            var_type = var_info.get('type', 'unknown')
            context_parts.append(f"- {var_name}: {var_type}")

    context_parts.append("")
    return "\n".join(context_parts)


def get_pruned_context_prompt(
    early_steps_summary: str,
    recent_steps: List[Dict[str, Any]],
    keep_last_n: int = 2
) -> str:
    """
    Format context with pruning optimization (summary of early + full detail of recent).

    Args:
        early_steps_summary: Summary of early steps (e.g., "Steps 1-5: Used web_search, python_coder")
        recent_steps: List of recent step dictionaries to show in full detail
        keep_last_n: Number of recent steps included (for documentation)

    Returns:
        Formatted pruned context string

    Example:
        >>> summary = "Steps 1-3: Searched web, analyzed data"
        >>> recent = [{"step_num": 4, "thought": "...", ...}]
        >>> prompt = get_pruned_context_prompt(summary, recent)
    """
    context_parts = ["=== Previous Steps (Optimized) ==="]

    # Summary of early steps
    context_parts.append(f"\n[Summary] {early_steps_summary}\n")

    # Recent steps in full detail
    context_parts.append(f"[Recent Steps - Full Detail (Last {keep_last_n})]")

    for step in recent_steps:
        step_num = step.get('step_num', '?')
        thought = step.get('thought', 'N/A')
        action = step.get('action', 'N/A')
        observation = step.get('observation', 'N/A')

        # Truncate very long observations
        obs_display = observation[:500] if len(observation) > 500 else observation

        context_parts.append(f"""
Step {step_num}:
- Thought: {thought[:200]}{"..." if len(thought) > 200 else ""}
- Action: {action}
- Observation: {obs_display}
""")

    return "\n".join(context_parts)


def get_plan_step_context_prompt(
    current_step_goal: str,
    success_criteria: str,
    previous_steps_results: str,
    step_num: int
) -> str:
    """
    Format context for plan execution step.

    Args:
        current_step_goal: Goal of the current plan step
        success_criteria: Success criteria for the step
        previous_steps_results: Results from previous plan steps
        step_num: Current step number

    Returns:
        Formatted plan step context string

    Example:
        >>> prompt = get_plan_step_context_prompt(
        ...     current_step_goal="Calculate mean of numeric columns",
        ...     success_criteria="Mean values displayed for all columns",
        ...     previous_steps_results="Step 1: Loaded data...",
        ...     step_num=2
        ... )
    """
    context_parts = [
        f"=== Plan Execution - Step {step_num} ===\n",
        f"**Current Goal:** {current_step_goal}",
        f"**Success Criteria:** {success_criteria}",
        ""
    ]

    if previous_steps_results and previous_steps_results != "This is the first step.":
        context_parts.append("**Previous Steps Results:**")
        context_parts.append(previous_steps_results)
        context_parts.append("")

    return "\n".join(context_parts)


def get_code_history_context_prompt(
    code_history: List[Dict[str, Any]],
    max_versions: int = 3
) -> str:
    """
    Format code history from previous stages for context.

    Args:
        code_history: List of code history entries
        max_versions: Maximum number of code versions to show

    Returns:
        Formatted code history string

    Example:
        >>> history = [{"stage_name": "analysis", "code": "import pandas...", ...}]
        >>> prompt = get_code_history_context_prompt(history)
    """
    if not code_history:
        return ""

    context_parts = ["=== Previous Code Versions ===\n"]

    for idx, code_entry in enumerate(code_history[:max_versions], 1):
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

    return "\n".join(context_parts)
