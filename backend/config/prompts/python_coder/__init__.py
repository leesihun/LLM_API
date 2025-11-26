"""
Python Code Generation Prompts - Modular Composition
Provides composable prompts for Python code generation, verification, and fixing.

Uses standardized structure from base.py.
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
    get_execution_fix_prompt,
    get_retry_prompt_with_history
)


def get_python_code_generation_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    is_prestep: bool = False,
    has_json_files: bool = False,
    conversation_history: Optional[List[dict]] = None,
    plan_context: Optional[dict] = None,
    react_context: Optional[dict] = None,
    timezone: str = "UTC"
) -> str:
    """
    Main code generation prompt - composes sections based on parameters.
    """
    # Pre-step mode: Use specialized fast prompt
    if is_prestep:
        return get_prestep_generation_prompt(query, file_context, has_json_files, timezone)

    # Normal mode: Compose from sections
    prompt_parts = []

    # 1. Conversation History
    if conversation_history and len(conversation_history) > 0:
        prompt_parts.append(get_conversation_history_section(conversation_history))

    # 2. Base Generation Prompt
    prompt_parts.append(get_base_generation_prompt(query, context, timezone))

    # 3. Plan Context
    if plan_context:
        prompt_parts.append(get_plan_section(plan_context))

    # 4. ReAct Context
    if react_context:
        prompt_parts.append(get_react_section(react_context))

    # 5. Task Guidance
    prompt_parts.append(get_task_guidance(query))

    # 6. File Context
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
    """Code verification prompt."""
    return get_verification_prompt(query, context, file_context, code, has_json_files)


def get_python_code_modification_prompt(
    query: str,
    context: Optional[str],
    code: str,
    issues: List[str]
) -> str:
    """Code modification prompt to fix issues."""
    return get_modification_prompt(query, context, code, issues)


def get_python_code_execution_fix_prompt(
    query: str,
    context: Optional[str],
    code: str,
    error_message: str,
    error_namespace: Optional[dict] = None
) -> str:
    """Prompt for fixing Python code after execution error."""
    return get_execution_fix_prompt(query, context, code, error_message, error_namespace)


def get_code_generation_with_self_verification_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    is_prestep: bool = False,
    has_json_files: bool = False,
    conversation_history: Optional[List[dict]] = None,
    plan_context: Optional[dict] = None,
    react_context: Optional[dict] = None,
    timezone: str = "UTC"
) -> str:
    """Combined code generation + self-verification prompt."""
    # Build generation prompt
    generation_prompt = get_python_code_generation_prompt(
        query=query,
        context=context,
        file_context=file_context,
        is_prestep=is_prestep,
        has_json_files=has_json_files,
        conversation_history=conversation_history,
        plan_context=plan_context,
        react_context=react_context,
        timezone=timezone
    )

    # Remove last instruction line and add self-verification
    generation_lines = generation_prompt.split('\n')
    generation_prompt_clean = '\n'.join(generation_lines[:-2])

    verification_instructions = get_self_verification_section(query, has_json_files)

    return generation_prompt_clean + verification_instructions


def get_output_adequacy_check_prompt(
    query: str,
    code: str,
    output: str,
    context: Optional[str] = None
) -> str:
    """Check if code execution output adequately answers the user's question."""
    return get_output_adequacy_prompt(query, code, output, context)


__all__ = [
    # Main composition functions
    'get_python_code_generation_prompt',
    'get_python_code_verification_prompt',
    'get_python_code_modification_prompt',
    'get_python_code_execution_fix_prompt',
    'get_code_generation_with_self_verification_prompt',
    'get_output_adequacy_check_prompt',

    # Template functions
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
    'get_retry_prompt_with_history',
]
