"""
Reusable Template Components for Python Code Generation
Uses standardized structure from base.py.
"""

from typing import Optional, List, Dict, Any
from ..base import (
    section_border, MARKER_OK, MARKER_ERROR, MARKER_CRITICAL,
    FILENAME_RULES, NO_ARGS_RULES, JSON_SAFETY_RULES, ACCESS_PATTERN_RULES,
    OUTPUT_FILE_RULES, get_current_time_context
)


def get_file_context_section(file_context: str) -> str:
    """Format file metadata section."""
    return f"""{section_border("META DATA (AVAILABLE FILES)")}
{file_context}
"""


def get_rules_section(has_json_files: bool = False) -> str:
    """Core rules section using shared rule blocks."""
    rules = f"""{section_border("RULES")}
{FILENAME_RULES}
{NO_ARGS_RULES}
{ACCESS_PATTERN_RULES}
{OUTPUT_FILE_RULES}
"""
    if has_json_files:
        rules += JSON_SAFETY_RULES
    return rules


def get_plan_section(plan_context: Dict[str, Any]) -> str:
    """
    Format plan-execute workflow context with clear step relationships.
    Enhanced to show data flow between steps.
    """
    if not plan_context:
        return ""
    
    current = plan_context.get('current_step', '?')
    total = plan_context.get('total_steps', '?')
    
    lines = [
        section_border(f"PLAN CONTEXT - Step {current}/{total}"),
        ""
    ]
    
    # Show plan with status and data flow
    if 'plan' in plan_context:
        lines.append("Plan Overview:")
        for step in plan_context['plan']:
            num = step.get('step_number', '?')
            goal = step.get('goal', '')
            status = step.get('status', 'pending')
            
            # Status markers
            if status == 'completed':
                marker = f"{MARKER_OK} DONE"
            elif status == 'current':
                marker = "<- CURRENT"
            elif status == 'failed':
                marker = f"{MARKER_ERROR} FAILED"
            else:
                marker = ""
            
            lines.append(f"  Step {num}: {goal} {marker}")
            
            # Show success criteria for context
            if 'success_criteria' in step and status == 'current':
                lines.append(f"    Target: {step['success_criteria']}")
        
        lines.append("")
    
    # Show previous results with data summary
    if 'previous_results' in plan_context and plan_context['previous_results']:
        lines.append("Data from Previous Steps:")
        for result in plan_context['previous_results']:
            num = result.get('step_number', '?')
            summary = result.get('summary', '')
            # Truncate long summaries
            if len(summary) > 200:
                summary = summary[:200] + "..."
            lines.append(f"  Step {num} output: {summary}")
        lines.append("")
    
    lines.append("=" * 80)
    return "\n".join(lines)


def get_react_section(react_context: Dict[str, Any]) -> str:
    """
    Format ReAct iteration history.
    Shows current execution attempts for context.
    """
    if not react_context:
        return ""
    
    has_history = react_context.get('history')
    if not has_history:
        return ""
    
    lines = [
        section_border("PREVIOUS ATTEMPTS"),
        ""
    ]
    
    for idx, iteration in enumerate(react_context['history'], 1):
        lines.append(f"--- Attempt {idx} ---")
        
        if 'thought' in iteration:
            lines.append(f"Thought: {iteration['thought'][:100]}...")
        
        if 'code' in iteration:
            code_preview = iteration['code'][:300]
            if len(iteration['code']) > 300:
                code_preview += "\n... (truncated)"
            lines.append(f"Code:\n```python\n{code_preview}\n```")
        
        if 'observation' in iteration:
            obs = iteration['observation'][:200]
            status = iteration.get('status', '')
            if status == 'error':
                lines.append(f"Result: {MARKER_ERROR} {obs}")
            else:
                lines.append(f"Result: {obs}")
        
        lines.append("")
    
    lines.append("=" * 80)
    return "\n".join(lines)


def get_conversation_history_section(conversation_history: List[Dict]) -> str:
    """Format conversation history concisely."""
    if not conversation_history:
        return ""
    
    lines = [
        section_border("CONVERSATION HISTORY"),
        ""
    ]
    
    for idx, turn in enumerate(conversation_history[-5:], 1):  # Last 5 turns only
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')
        
        # Truncate long content
        if len(content) > 5000000:
            content = content[:5000000] + "..."
        
        role_label = "User" if role == 'user' else "AI"
        lines.append(f"[{role_label}]: {content}")
        lines.append("")
    
    lines.append("=" * 80)
    return "\n".join(lines)
