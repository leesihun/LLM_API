"""
Centralized Prompt Management
All system prompts are organized by role/module for better maintainability.
"""

from .task_classification import get_agentic_classifier_prompt
from .react_agent import (
    get_react_thought_and_action_prompt,
    get_react_final_answer_prompt,
    get_react_final_answer_for_finish_step_prompt,
    get_react_action_input_for_step_prompt,
    get_react_step_verification_prompt,
    get_react_final_answer_from_steps_prompt
)
from .python_coder import (
    get_python_code_generation_prompt,
    get_python_code_verification_prompt,
    get_python_code_modification_prompt,
    get_python_code_execution_fix_prompt
)
from .web_search import (
    get_search_query_refinement_prompt,
    get_search_answer_generation_system_prompt,
    get_search_answer_generation_user_prompt
)

__all__ = [
    # Task Classification
    'get_agentic_classifier_prompt',

    # ReAct Agent
    'get_react_thought_and_action_prompt',
    'get_react_final_answer_prompt',
    'get_react_final_answer_for_finish_step_prompt',
    'get_react_action_input_for_step_prompt',
    'get_react_step_verification_prompt',
    'get_react_final_answer_from_steps_prompt',

    # Python Coder
    'get_python_code_generation_prompt',
    'get_python_code_verification_prompt',
    'get_python_code_modification_prompt',
    'get_python_code_execution_fix_prompt',

    # Web Search
    'get_search_query_refinement_prompt',
    'get_search_answer_generation_system_prompt',
    'get_search_answer_generation_user_prompt',
]
