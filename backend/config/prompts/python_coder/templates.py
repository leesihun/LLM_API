"""
Reusable Template Components for Python Code Generation
Provides composable sections for file context, rules, plans, and ReAct history.
"""

from typing import Optional, List, Dict, Any


def get_file_context_section(file_context: str) -> str:
    """
    Format file metadata section.

    Args:
        file_context: Pre-formatted file context string with metadata

    Returns:
        Formatted file context section
    """
    return f"""{"="*80}
META DATA (AVAILABLE FILES)
{"="*80}
{file_context}
"""


def get_rules_section(has_json_files: bool = False) -> str:
    """
    Core rules section - essential rules only.

    Args:
        has_json_files: Whether JSON files are present

    Returns:
        Formatted rules section
    """
    rules = f"""{"="*80}
RULES
{"="*80}

[RULE 1] EXACT FILENAMES
   - Copy EXACT filename from META DATA section above
   - [X] NO generic names: 'data.json', 'file.json', 'input.csv'
   - [OK] Example: filename = 'sales_report_Q4_2024.json'

[RULE 2] NO COMMAND-LINE ARGS / USER INPUT
   - Code runs via subprocess WITHOUT arguments
   - [X] NO sys.argv, NO input(), NO argparse
   - [OK] All filenames must be HARDCODED

[RULE 3] USE ACCESS PATTERNS
   - Copy access patterns from META DATA section
   - [X] DON'T guess keys or field names
   - [OK] Use .get() for safe dict access
"""

    if has_json_files:
        rules += """
[RULE 4] JSON SAFETY
   - Use .get() for dict access: data.get('key', default)
   - Check type: isinstance(data, dict) or isinstance(data, list)
   - Add error handling: try/except json.JSONDecodeError
"""

    return rules


def get_plan_section(plan_context: Dict[str, Any]) -> str:
    """
    Format plan-execute workflow context.

    Args:
        plan_context: Dict with 'current_step', 'total_steps', 'plan', 'previous_results'

    Returns:
        Formatted plan section
    """
    lines = [
        "="*80,
        "PLANS".center(80),
        "="*80,
        "",
        f"[Plan-Execute Workflow - Step {plan_context.get('current_step', '?')} of {plan_context.get('total_steps', '?')}]",
        ""
    ]

    if 'plan' in plan_context:
        lines.append("Full Plan:")
        for step in plan_context['plan']:
            step_num = step.get('step_number', '?')
            goal = step.get('goal', '')
            status = step.get('status', 'pending')

            status_marker = ""
            if status == 'completed':
                status_marker = " [OK] COMPLETED"
            elif status == 'current':
                status_marker = " <- CURRENT STEP"
            elif status == 'failed':
                status_marker = " [X] FAILED"

            lines.append(f"  Step {step_num}: {goal}{status_marker}")

            if 'success_criteria' in step:
                lines.append(f"    Success criteria: {step['success_criteria']}")
            if 'primary_tools' in step:
                lines.append(f"    Primary tools: {', '.join(step['primary_tools'])}")

        lines.append("")

    if 'previous_results' in plan_context and plan_context['previous_results']:
        lines.append("Previous Steps Results:")
        for result in plan_context['previous_results']:
            step_num = result.get('step_number', '?')
            summary = result.get('summary', '')
            lines.append(f"  Step {step_num} -> {summary}")
        lines.append("")

    lines.extend([
        "="*80,
        ""
    ])

    return "\n".join(lines)


def get_react_section(react_context: Dict[str, Any]) -> str:
    """
    Format ReAct iteration history with ALL code attempts (current + session-wide).

    NEW BEHAVIOR: Shows complete code execution history including:
    - All attempts from current ReAct execution (with multiple retries per step)
    - All code from previous executions in the same session

    Args:
        react_context: Dict with 'iteration', 'history', and 'session_code_history'

    Returns:
        Formatted ReAct section with complete code history
    """
    if not react_context:
        return ""

    has_current_history = react_context.get('history')
    has_session_history = react_context.get('session_code_history')

    if not has_current_history and not has_session_history:
        return ""

    lines = [
        "="*80,
        "REACTS (CODE EXECUTION HISTORY)".center(80),
        "="*80,
        ""
    ]

    # Section 1: Current ReAct Iteration History
    if has_current_history:
        lines.append("[CURRENT REACT EXECUTION]")
        lines.append("")

        current_iteration = react_context.get('iteration', 1)

        for idx, iteration in enumerate(react_context['history'], 1):
            lines.append(f"=== Iteration {idx} ===")

            # Thought
            if 'thought' in iteration:
                lines.append(f"Thought: {iteration['thought']}")

            # Action
            if 'action' in iteration:
                lines.append(f"Action: {iteration['action']}")

            # Tool Input
            if 'tool_input' in iteration:
                lines.append(f"Tool Input: {iteration['tool_input']}")

            # Generated Code - handle both single code and multiple attempts
            if 'code_attempts' in iteration:
                # Multiple attempts for this step
                lines.append("")
                lines.append(f"Code Attempts ({len(iteration['code_attempts'])} total):")
                for attempt in iteration['code_attempts']:
                    attempt_num = attempt.get('attempt_num', '?')
                    filename = attempt.get('filename', 'unknown')
                    lines.append(f"\n--- Attempt {attempt_num} ({filename}) ---")
                    lines.append("```python")
                    code_lines = attempt['code'].split('\n')
                    for line in code_lines:
                        lines.append(line)
                    lines.append("```")
            elif 'code' in iteration:
                # Single code version
                lines.append("")
                lines.append("Generated Code:")
                lines.append("```python")
                code_lines = iteration['code'].split('\n')
                for line in code_lines:
                    lines.append(line)
                lines.append("```")

            # Observation (error/result)
            if 'observation' in iteration:
                obs = iteration['observation']
                lines.append("")
                if iteration.get('status') == 'error':
                    lines.append(f"Observation: [ERROR] {obs}")

                    # Add error reason if available
                    if 'error_reason' in iteration:
                        lines.append(f"Error Reason: {iteration['error_reason']}")
                else:
                    lines.append(f"Observation: {obs}")

            lines.append("")

        # Current iteration marker
        lines.append(f"=== Iteration {current_iteration} (CURRENT) ===")
        lines.append("Awaiting code generation...")
        lines.append("")

    # Section 2: Session-Wide Code History
    if has_session_history:
        lines.append("")
        lines.append("[PREVIOUS SESSION CODE HISTORY]")
        lines.append(f"({len(react_context['session_code_history'])} code file(s) from this session)")
        lines.append("")

        for idx, code_entry in enumerate(react_context['session_code_history'], 1):
            filename = code_entry.get('filename', 'unknown')
            stage_name = code_entry.get('stage_name', 'unknown')

            lines.append(f"--- Code {idx}: {filename} (stage: {stage_name}) ---")
            lines.append("```python")
            code_lines = code_entry['code'].split('\n')
            for line in code_lines:
                lines.append(line)
            lines.append("```")
            lines.append("")

    lines.extend([
        "="*80,
        ""
    ])

    return "\n".join(lines)


def get_conversation_history_section(conversation_history: List[Dict]) -> str:
    """
    Format conversation history.

    Args:
        conversation_history: List of conversation turns with 'role', 'content', 'timestamp'

    Returns:
        Formatted conversation history section
    """
    if not conversation_history or len(conversation_history) == 0:
        return ""

    lines = [
        "="*80,
        "PAST HISTORIES".center(80),
        "="*80,
        ""
    ]

    for idx, turn in enumerate(conversation_history, 1):
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')
        timestamp = turn.get('timestamp', '')

        if role == 'user':
            lines.append(f"=== Turn {idx} (User) ===")
        elif role == 'assistant':
            lines.append(f"=== Turn {idx} (AI) ===")
        else:
            lines.append(f"=== Turn {idx} ({role}) ===")

        if timestamp:
            lines.append(f"Time: {timestamp}")

        # Show ALL content - no truncation
        lines.append(content)

        lines.append("")

    lines.extend([
        "="*80,
        ""
    ])

    return "\n".join(lines)
