"""
Centralized Prompt Management
All system prompts are organized by role/module for better maintainability.

This module provides a centralized PromptRegistry for accessing all prompts with:
- Caching for performance
- Easy access pattern: PromptRegistry.get('prompt_name', **kwargs)
- Enhanced registry with parameter introspection
"""

# Import the enhanced registry
from .registry import PromptRegistry, PromptRegistryMeta

# Import all prompt functions
from .task_classification import get_agent_type_classifier_prompt
from .react_agent import (
    get_react_thought_and_action_prompt,
    get_react_final_answer_prompt,
    get_react_step_verification_prompt,
    get_react_final_answer_from_steps_prompt,
    get_react_thought_prompt,
    get_react_action_selection_prompt
)
from .python_coder import (
    get_python_code_generation_prompt,
    get_python_code_verification_prompt,
    get_python_code_modification_prompt,
    get_python_code_execution_fix_prompt,
    get_code_generation_with_self_verification_prompt,
    get_output_adequacy_check_prompt
)
from .web_search import (
    get_search_query_refinement_prompt,
    get_search_answer_generation_system_prompt,
    get_search_answer_generation_user_prompt
)
from .plan_execute import (
    get_execution_plan_prompt
)
from .file_analyzer import get_deep_analysis_prompt


# Register all prompts in the enhanced registry
def _register_all_prompts():
    """Register all prompt functions in the PromptRegistry."""

    # Task Classification
    PromptRegistry.register('agent_type_classifier', get_agent_type_classifier_prompt)

    # ReAct Agent
    PromptRegistry.register('react_thought_and_action', get_react_thought_and_action_prompt)
    PromptRegistry.register('react_final_answer', get_react_final_answer_prompt)
    PromptRegistry.register('react_step_verification', get_react_step_verification_prompt)
    PromptRegistry.register('react_final_answer_from_steps', get_react_final_answer_from_steps_prompt)
    PromptRegistry.register('react_thought', get_react_thought_prompt)
    PromptRegistry.register('react_action_selection', get_react_action_selection_prompt)

    # Python Coder
    PromptRegistry.register('python_code_generation', get_python_code_generation_prompt)
    PromptRegistry.register('python_code_verification', get_python_code_verification_prompt)
    PromptRegistry.register('python_code_modification', get_python_code_modification_prompt)
    PromptRegistry.register('python_code_execution_fix', get_python_code_execution_fix_prompt)
    PromptRegistry.register('python_code_generation_with_self_verification', get_code_generation_with_self_verification_prompt)
    PromptRegistry.register('python_code_output_adequacy_check', get_output_adequacy_check_prompt)

    # Web Search
    PromptRegistry.register('search_query_refinement', get_search_query_refinement_prompt)
    PromptRegistry.register('search_answer_generation_system', get_search_answer_generation_system_prompt)
    PromptRegistry.register('search_answer_generation_user', get_search_answer_generation_user_prompt)

    # Plan-Execute
    PromptRegistry.register('execution_plan', get_execution_plan_prompt)

    # File Analyzer prompts
    PromptRegistry.register('file_analyzer_deep_analysis', get_deep_analysis_prompt)


# Auto-register all prompts on module import
_register_all_prompts()


__all__ = [
    # Main Registry
    'PromptRegistry',
    'PromptRegistryMeta',

    # Task Classification
    'get_agent_type_classifier_prompt',

    # ReAct Agent
    'get_react_thought_and_action_prompt',
    'get_react_final_answer_prompt',
    'get_react_step_verification_prompt',
    'get_react_final_answer_from_steps_prompt',
    'get_react_thought_prompt',
    'get_react_action_selection_prompt',

    # Python Coder
    'get_python_code_generation_prompt',
    'get_python_code_verification_prompt',
    'get_python_code_modification_prompt',
    'get_python_code_execution_fix_prompt',
    'get_code_generation_with_self_verification_prompt',
    'get_output_adequacy_check_prompt',

    # Web Search
    'get_search_query_refinement_prompt',
    'get_search_answer_generation_system_prompt',
    'get_search_answer_generation_user_prompt',

    # Plan-Execute
    'get_execution_plan_prompt',

    # File Analyzer
    'get_deep_analysis_prompt',
]
