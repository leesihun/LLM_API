"""
Centralized Prompt Management
All system prompts are organized by role/module for better maintainability.

This module provides a centralized PromptRegistry for accessing all prompts with:
- Caching for performance
- Validation for correctness
- Easy access pattern: PromptRegistry.get('prompt_name', **kwargs)
- Enhanced registry with parameter introspection
- Prompt validation utilities
"""

from typing import Dict, Any, Callable, Optional
import functools

# Import the enhanced registry
from .registry import PromptRegistry, PromptRegistryMeta

# Import all prompt functions
from .task_classification import get_agent_type_classifier_prompt
from .react_agent import (
    get_react_thought_and_action_prompt,
    get_react_final_answer_prompt,
    get_react_final_answer_for_finish_step_prompt,
    get_react_action_input_for_step_prompt,
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
from .agent_graph import (
    get_planning_prompt,
    get_reasoning_prompt,
    get_verification_prompt,
)
from .file_analyzer import get_deep_analysis_prompt
from .rag import (
    get_rag_query_enhancement_prompt,
    get_rag_answer_synthesis_prompt,
    get_rag_document_summary_prompt,
    get_rag_relevance_check_prompt,
    get_rag_multi_document_synthesis_prompt
)
from .phase_manager import (
    get_initial_phase_prompt,
    get_handoff_phase_prompt,
    get_phase_summary_prompt,
    get_workflow_completion_prompt
)
from .context_formatting import (
    get_file_context_summary_prompt,
    get_step_history_summary_prompt,
    get_pruned_context_prompt,
    get_plan_step_context_prompt,
    get_code_history_context_prompt
)

# Import validators
from .validators import (
    PromptValidator,
    PromptQualityChecker,
    validate_prompt_registry
)


# Register all prompts in the enhanced registry
def _register_all_prompts():
    """Register all prompt functions in the PromptRegistry."""

    # Task Classification
    PromptRegistry.register('agent_type_classifier', get_agent_type_classifier_prompt)

    # ReAct Agent
    PromptRegistry.register('react_thought_and_action', get_react_thought_and_action_prompt)
    PromptRegistry.register('react_final_answer', get_react_final_answer_prompt)
    PromptRegistry.register('react_final_answer_for_finish_step', get_react_final_answer_for_finish_step_prompt)
    PromptRegistry.register('react_action_input_for_step', get_react_action_input_for_step_prompt)
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

    # Agent Graph prompts
    PromptRegistry.register('agent_graph_planning', get_planning_prompt)
    PromptRegistry.register('agent_graph_reasoning', get_reasoning_prompt)
    PromptRegistry.register('agent_graph_verification', get_verification_prompt)

    # File Analyzer prompts
    PromptRegistry.register('file_analyzer_deep_analysis', get_deep_analysis_prompt)

    # RAG Retriever prompts
    PromptRegistry.register('rag_query_enhancement', get_rag_query_enhancement_prompt)
    PromptRegistry.register('rag_answer_synthesis', get_rag_answer_synthesis_prompt)
    PromptRegistry.register('rag_document_summary', get_rag_document_summary_prompt)
    PromptRegistry.register('rag_relevance_check', get_rag_relevance_check_prompt)
    PromptRegistry.register('rag_multi_document_synthesis', get_rag_multi_document_synthesis_prompt)

    # Phase Manager prompts
    PromptRegistry.register('phase_initial', get_initial_phase_prompt)
    PromptRegistry.register('phase_handoff', get_handoff_phase_prompt)
    PromptRegistry.register('phase_summary', get_phase_summary_prompt)
    PromptRegistry.register('workflow_completion', get_workflow_completion_prompt)

    # Context Formatting prompts
    PromptRegistry.register('context_file_summary', get_file_context_summary_prompt)
    PromptRegistry.register('context_step_history', get_step_history_summary_prompt)
    PromptRegistry.register('context_pruned', get_pruned_context_prompt)
    PromptRegistry.register('context_plan_step', get_plan_step_context_prompt)
    PromptRegistry.register('context_code_history', get_code_history_context_prompt)


# Auto-register all prompts on module import
_register_all_prompts()


__all__ = [
    # Main Registry
    'PromptRegistry',
    'PromptRegistryMeta',

    # Validators
    'PromptValidator',
    'PromptQualityChecker',
    'validate_prompt_registry',

    # Task Classification
    'get_agent_type_classifier_prompt',

    # ReAct Agent
    'get_react_thought_and_action_prompt',
    'get_react_final_answer_prompt',
    'get_react_final_answer_for_finish_step_prompt',
    'get_react_action_input_for_step_prompt',
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

    # Agent Graph
    'get_planning_prompt',
    'get_reasoning_prompt',
    'get_verification_prompt',

    # File Analyzer
    'get_deep_analysis_prompt',

    # RAG Retriever
    'get_rag_query_enhancement_prompt',
    'get_rag_answer_synthesis_prompt',
    'get_rag_document_summary_prompt',
    'get_rag_relevance_check_prompt',
    'get_rag_multi_document_synthesis_prompt',

    # Phase Manager
    'get_initial_phase_prompt',
    'get_handoff_phase_prompt',
    'get_phase_summary_prompt',
    'get_workflow_completion_prompt',

    # Context Formatting
    'get_file_context_summary_prompt',
    'get_step_history_summary_prompt',
    'get_pruned_context_prompt',
    'get_plan_step_context_prompt',
    'get_code_history_context_prompt',
]
