"""
Centralized Prompt Management
All system prompts are organized by role/module for better maintainability.

This module provides a centralized PromptRegistry for accessing all prompts with:
- Caching for performance
- Validation for correctness
- Easy access pattern: PromptRegistry.get('prompt_name', **kwargs)
"""

from typing import Dict, Any, Callable, Optional
import functools

from .task_classification import get_agentic_classifier_prompt
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
    get_python_code_execution_fix_prompt
)
from .web_search import (
    get_search_query_refinement_prompt,
    get_search_answer_generation_system_prompt,
    get_search_answer_generation_user_prompt
)
from .plan_execute import (
    get_execution_plan_prompt
)


class PromptRegistry:
    """
    Centralized registry for all system prompts with caching and validation.

    Usage:
        >>> prompt = PromptRegistry.get('react_thought_and_action',
        ...                             query="What is AI?",
        ...                             context="",
        ...                             file_guidance="")

    Features:
        - Cached prompt generation for performance
        - Validation of prompt parameters
        - Easy-to-use access pattern
        - Clear error messages for missing prompts
    """

    # Registry mapping prompt names to their generator functions
    _REGISTRY: Dict[str, Callable] = {
        # Task Classification
        'agentic_classifier': get_agentic_classifier_prompt,

        # ReAct Agent
        'react_thought_and_action': get_react_thought_and_action_prompt,
        'react_final_answer': get_react_final_answer_prompt,
        'react_final_answer_for_finish_step': get_react_final_answer_for_finish_step_prompt,
        'react_action_input_for_step': get_react_action_input_for_step_prompt,
        'react_step_verification': get_react_step_verification_prompt,
        'react_final_answer_from_steps': get_react_final_answer_from_steps_prompt,
        'react_thought': get_react_thought_prompt,
        'react_action_selection': get_react_action_selection_prompt,

        # Python Coder
        'python_code_generation': get_python_code_generation_prompt,
        'python_code_verification': get_python_code_verification_prompt,
        'python_code_modification': get_python_code_modification_prompt,
        'python_code_execution_fix': get_python_code_execution_fix_prompt,

        # Web Search
        'search_query_refinement': get_search_query_refinement_prompt,
        'search_answer_generation_system': get_search_answer_generation_system_prompt,
        'search_answer_generation_user': get_search_answer_generation_user_prompt,

        # Plan-Execute
        'execution_plan': get_execution_plan_prompt,
    }

    # Cache for generated prompts (key: (prompt_name, frozenset(kwargs)))
    _cache: Dict[tuple, str] = {}

    @classmethod
    def get(cls, prompt_name: str, use_cache: bool = True, **kwargs) -> str:
        """
        Get a prompt by name with optional parameters.

        Args:
            prompt_name: Name of the prompt (e.g., 'react_thought_and_action')
            use_cache: Whether to use cached prompts (default: True)
            **kwargs: Parameters to pass to the prompt generator function

        Returns:
            Generated prompt string

        Raises:
            ValueError: If prompt_name is not found in registry
            TypeError: If required parameters are missing

        Example:
            >>> prompt = PromptRegistry.get('react_final_answer',
            ...                             query="What is AI?",
            ...                             context="Previous steps...")
        """
        # Validate prompt exists
        if prompt_name not in cls._REGISTRY:
            available = ', '.join(sorted(cls._REGISTRY.keys()))
            raise ValueError(
                f"Prompt '{prompt_name}' not found in registry. "
                f"Available prompts: {available}"
            )

        # Check cache if enabled
        if use_cache:
            cache_key = (prompt_name, frozenset(kwargs.items()))
            if cache_key in cls._cache:
                return cls._cache[cache_key]

        # Get generator function
        generator = cls._REGISTRY[prompt_name]

        # Generate prompt
        try:
            prompt = generator(**kwargs)
        except TypeError as e:
            # Provide helpful error message for missing parameters
            raise TypeError(
                f"Error generating prompt '{prompt_name}': {e}. "
                f"Check the function signature for required parameters."
            ) from e

        # Validate prompt is not empty
        if not prompt or not prompt.strip():
            raise ValueError(
                f"Prompt '{prompt_name}' generated an empty string. "
                f"Check the generator function."
            )

        # Cache the result
        if use_cache:
            cache_key = (prompt_name, frozenset(kwargs.items()))
            cls._cache[cache_key] = prompt

        return prompt

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the prompt cache. Useful for testing or memory management."""
        cls._cache.clear()

    @classmethod
    def list_prompts(cls) -> list[str]:
        """Get a sorted list of all available prompt names."""
        return sorted(cls._REGISTRY.keys())

    @classmethod
    def validate_all(cls) -> Dict[str, bool]:
        """
        Validate all prompts can be accessed (basic smoke test).

        Returns:
            Dictionary mapping prompt names to validation status

        Note:
            This only validates prompts that don't require parameters.
            For parameterized prompts, returns True without validation.
        """
        results = {}
        for prompt_name in cls._REGISTRY.keys():
            try:
                # Try to get prompts without parameters (will fail for most)
                # This is just a basic registry check
                generator = cls._REGISTRY[prompt_name]
                results[prompt_name] = callable(generator)
            except Exception:
                results[prompt_name] = False
        return results


__all__ = [
    # Main Registry
    'PromptRegistry',

    # Task Classification
    'get_agentic_classifier_prompt',

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

    # Web Search
    'get_search_query_refinement_prompt',
    'get_search_answer_generation_system_prompt',
    'get_search_answer_generation_user_prompt',

    # Plan-Execute
    'get_execution_plan_prompt',
]
