"""
Python Code Generation Prompts - Modular Composition
Provides composable, testable prompts for Python code generation, verification, and fixing.

This module composes prompts from specialized submodules:
- generation.py: Base generation prompts
- templates.py: Reusable template sections
- verification.py: Code verification prompts
- fixing.py: Error fixing prompts
"""

from typing import Optional, List, Dict, Any

# Import template functions
from .templates import (
    get_file_context_section,
    get_rules_section,
    get_plan_section,
    get_react_section,
    get_conversation_history_section
)

# Import generation functions
from .generation import (
    get_base_generation_prompt,
    get_task_guidance,
    get_prestep_generation_prompt,
    get_checklists_section
)

# Import verification functions
from .verification import (
    get_verification_prompt,
    get_self_verification_section,
    get_output_adequacy_prompt
)

# Import fixing functions
from .fixing import (
    get_modification_prompt,
    get_execution_fix_prompt
)


def get_python_code_generation_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    is_prestep: bool = False,
    has_json_files: bool = False,
    conversation_history: Optional[List[dict]] = None,
    plan_context: Optional[dict] = None,
    react_context: Optional[dict] = None
) -> str:
    """
    Main code generation prompt - composes sections based on parameters.

    This function builds a prompt dynamically by combining:
    - Conversation history (if provided)
    - Base generation prompt
    - File context (if provided)
    - Plan context (if provided)
    - ReAct context (if provided)
    - Task guidance
    - Rules
    - Checklists

    Args:
        query: User's task (the ACTUAL question user asked)
        context: Optional additional context
        file_context: File information
        is_prestep: Whether this is pre-step (fast analysis mode)
        has_json_files: Whether JSON files are present
        conversation_history: List of past conversation turns (dicts with 'role', 'content', 'timestamp')
        plan_context: Plan-Execute context (dict with 'current_step', 'total_steps', 'plan', 'previous_results')
        react_context: ReAct context (dict with 'iteration', 'history' containing failed attempts with code and errors)

    Returns:
        Composed generation prompt
    """
    # Pre-step mode: Use specialized fast prompt
    if is_prestep:
        return get_prestep_generation_prompt(query, file_context, has_json_files)

    # Normal mode: Compose from sections
    prompt_parts = []

    # 1. Conversation History (if provided)
    if conversation_history and len(conversation_history) > 0:
        prompt_parts.append(get_conversation_history_section(conversation_history))

    # 2. Base Generation Prompt (user's query)
    prompt_parts.append(get_base_generation_prompt(query, context))

    # 3. Plan Context (if from Plan-Execute workflow)
    if plan_context:
        prompt_parts.append(get_plan_section(plan_context))

    # 4. ReAct Context (if from ReAct iterations)
    if react_context:
        prompt_parts.append(get_react_section(react_context))

    # 5. Task Guidance (based on query type)
    prompt_parts.append(get_task_guidance(query))

    # 6. File Context (if files provided)
    if file_context:
        prompt_parts.append(get_file_context_section(file_context))

    # 7. Rules Section
    prompt_parts.append(get_rules_section(has_json_files))

    # 8. Checklists
    prompt_parts.append(get_checklists_section())

    return "\n".join(prompt_parts)


def get_python_code_verification_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    code: str,
    has_json_files: bool = False
) -> str:
    """
    Code verification prompt.

    Args:
        query: User's question
        context: Optional additional context
        file_context: File information
        code: Code to verify
        has_json_files: Whether JSON files are present

    Returns:
        Verification prompt
    """
    return get_verification_prompt(query, context, file_context, code, has_json_files)


def get_python_code_modification_prompt(
    query: str,
    context: Optional[str],
    code: str,
    issues: List[str]
) -> str:
    """
    Code modification prompt to fix issues.

    Args:
        query: Original user query
        context: Optional additional context
        code: Current code
        issues: List of issues to fix

    Returns:
        Modification prompt
    """
    return get_modification_prompt(query, context, code, issues)


def get_python_code_execution_fix_prompt(
    query: str,
    context: Optional[str],
    code: str,
    error_message: str
) -> str:
    """
    Prompt for fixing Python code after execution error.

    Args:
        query: Original user query
        context: Optional additional context
        code: Current code that failed
        error_message: Error from execution

    Returns:
        Execution fix prompt
    """
    return get_execution_fix_prompt(query, context, code, error_message)


def get_code_generation_with_self_verification_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    is_prestep: bool = False,
    has_json_files: bool = False,
    conversation_history: Optional[List[dict]] = None,
    plan_context: Optional[dict] = None,
    react_context: Optional[dict] = None
) -> str:
    """
    OPTIMIZED: Combined code generation + self-verification prompt.
    Generates code AND verifies it in a single LLM call.

    Args:
        query: User's task
        context: Optional additional context
        file_context: File information
        is_prestep: Whether this is pre-step (fast analysis mode)
        has_json_files: Whether JSON files are present
        conversation_history: List of past conversation turns
        plan_context: Plan-Execute context
        react_context: ReAct context with failed attempts

    Returns:
        Prompt that requests JSON response with code and self-check
    """
    # Build generation prompt (reuse existing logic)
    generation_prompt = get_python_code_generation_prompt(
        query=query,
        context=context,
        file_context=file_context,
        is_prestep=is_prestep,
        has_json_files=has_json_files,
        conversation_history=conversation_history,
        plan_context=plan_context,
        react_context=react_context
    )

    # Remove the last line "Generate ONLY the Python code, no explanations or markdown:"
    generation_lines = generation_prompt.split('\n')
    generation_prompt_clean = '\n'.join(generation_lines[:-2])  # Remove last 2 lines (empty + instruction)

    # Add self-verification instructions
    verification_instructions = get_self_verification_section(query, has_json_files)

    return generation_prompt_clean + verification_instructions


def get_output_adequacy_check_prompt(
    query: str,
    code: str,
    output: str,
    context: Optional[str] = None
) -> str:
    """
    Check if code execution output adequately answers the user's question.

    Args:
        query: Original user query
        code: The Python code that was executed
        output: The output from executing the code
        context: Optional additional context

    Returns:
        Output adequacy check prompt
    """
    return get_output_adequacy_prompt(query, code, output, context)


# Export all public functions
__all__ = [
    # Main composition function
    'get_python_code_generation_prompt',
    'get_python_code_verification_prompt',
    'get_python_code_modification_prompt',
    'get_python_code_execution_fix_prompt',
    'get_code_generation_with_self_verification_prompt',
    'get_output_adequacy_check_prompt',

    # Template functions (for direct use if needed)
    'get_file_context_section',
    'get_rules_section',
    'get_plan_section',
    'get_react_section',
    'get_conversation_history_section',

    # Generation functions
    'get_base_generation_prompt',
    'get_task_guidance',
    'get_prestep_generation_prompt',
    'get_checklists_section',

    # Verification functions
    'get_verification_prompt',
    'get_self_verification_section',
    'get_output_adequacy_prompt',

    # Fixing functions
    'get_modification_prompt',
    'get_execution_fix_prompt',
]
