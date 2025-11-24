"""
Enhanced Prompt Registry
Provides centralized prompt management with caching, validation, and parameter introspection.

Features:
- Auto-registration of prompt functions
- LRU caching for performance
- Parameter validation
- Prompt introspection
- Batch validation for testing
"""

from typing import Dict, Any, Callable, Optional, List, Set
import functools
import inspect
from functools import lru_cache

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PromptRegistryMeta:
    """
    Enhanced registry for all system prompts with caching and validation.

    Usage:
        >>> prompt = PromptRegistryMeta.get('react_thought_and_action',
        ...                                  query="What is AI?",
        ...                                  context="",
        ...                                  file_guidance="")

    Features:
        - Cached prompt generation for performance
        - Validation of prompt parameters
        - Easy-to-use access pattern
        - Clear error messages for missing prompts
        - Parameter introspection
    """

    # Registry mapping prompt names to their generator functions
    _REGISTRY: Dict[str, Callable] = {}

    # Cache for generated prompts (key: (prompt_name, frozenset(kwargs)))
    _cache: Dict[tuple, str] = {}

    # Maximum cache size
    _max_cache_size: int = 256

    @classmethod
    def register(cls, name: str, func: Optional[Callable] = None) -> Callable:
        """
        Register a prompt function in the registry.

        Can be used as a decorator or called directly:

        >>> @PromptRegistryMeta.register('my_prompt')
        ... def get_my_prompt(query: str) -> str:
        ...     return f"Query: {query}"

        >>> PromptRegistryMeta.register('my_prompt', get_my_prompt)

        Args:
            name: Name to register the prompt under
            func: Optional function to register (if not using as decorator)

        Returns:
            The original function (for decorator usage)
        """
        def decorator(f: Callable) -> Callable:
            if name in cls._REGISTRY:
                logger.warning(f"[PromptRegistry] Overwriting existing prompt: {name}")

            cls._REGISTRY[name] = f
            logger.debug(f"[PromptRegistry] Registered prompt: {name}")
            return f

        # If called with function directly
        if func is not None:
            return decorator(func)

        # If used as decorator
        return decorator

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
            >>> prompt = PromptRegistryMeta.get('react_final_answer',
            ...                                  query="What is AI?",
            ...                                  context="Previous steps...")
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
                logger.debug(f"[PromptRegistry] Cache hit for: {prompt_name}")
                return cls._cache[cache_key]

        # Get generator function
        generator = cls._REGISTRY[prompt_name]

        # Validate parameters before calling
        try:
            cls._validate_parameters(prompt_name, generator, kwargs)
        except TypeError as e:
            raise TypeError(
                f"Error validating parameters for prompt '{prompt_name}': {e}"
            ) from e

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

        # Cache the result (with size limit)
        if use_cache:
            cache_key = (prompt_name, frozenset(kwargs.items()))
            cls._cache[cache_key] = prompt

            # Prune cache if too large
            if len(cls._cache) > cls._max_cache_size:
                # Remove oldest 25% of entries
                items_to_remove = len(cls._cache) // 4
                for key in list(cls._cache.keys())[:items_to_remove]:
                    del cls._cache[key]
                logger.debug(f"[PromptRegistry] Pruned cache: removed {items_to_remove} entries")

        return prompt

    @classmethod
    def _validate_parameters(cls, prompt_name: str, func: Callable, kwargs: Dict[str, Any]) -> None:
        """
        Validate that all required parameters are provided.

        Args:
            prompt_name: Name of the prompt (for error messages)
            func: Prompt generator function
            kwargs: Parameters provided by caller

        Raises:
            TypeError: If required parameters are missing
        """
        sig = inspect.signature(func)
        required_params = [
            name for name, param in sig.parameters.items()
            if param.default == inspect.Parameter.empty and param.kind != inspect.Parameter.VAR_KEYWORD
        ]

        missing = set(required_params) - set(kwargs.keys())
        if missing:
            raise TypeError(
                f"Missing required parameters for prompt '{prompt_name}': {', '.join(sorted(missing))}"
            )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the prompt cache. Useful for testing or memory management."""
        cls._cache.clear()
        logger.debug("[PromptRegistry] Cache cleared")

    @classmethod
    def list_prompts(cls) -> List[str]:
        """Get a sorted list of all available prompt names."""
        return sorted(cls._REGISTRY.keys())

    @classmethod
    def list_all(cls) -> List[str]:
        """Alias for list_prompts() for consistency with other APIs."""
        return cls.list_prompts()

    @classmethod
    def get_params(cls, prompt_name: str) -> List[str]:
        """
        Get the parameter names for a specific prompt.

        Args:
            prompt_name: Name of the prompt

        Returns:
            List of parameter names

        Raises:
            ValueError: If prompt not found

        Example:
            >>> PromptRegistryMeta.get_params('react_final_answer')
            ['query', 'context']
        """
        if prompt_name not in cls._REGISTRY:
            raise ValueError(f"Prompt '{prompt_name}' not found in registry")

        func = cls._REGISTRY[prompt_name]
        sig = inspect.signature(func)
        return list(sig.parameters.keys())

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

    @classmethod
    def get_info(cls, prompt_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a prompt.

        Args:
            prompt_name: Name of the prompt

        Returns:
            Dictionary with prompt metadata

        Raises:
            ValueError: If prompt not found
        """
        if prompt_name not in cls._REGISTRY:
            raise ValueError(f"Prompt '{prompt_name}' not found in registry")

        func = cls._REGISTRY[prompt_name]
        sig = inspect.signature(func)

        params = {}
        for name, param in sig.parameters.items():
            params[name] = {
                'required': param.default == inspect.Parameter.empty,
                'default': None if param.default == inspect.Parameter.empty else param.default,
                'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None
            }

        return {
            'name': prompt_name,
            'function': func.__name__,
            'module': func.__module__,
            'docstring': func.__doc__,
            'parameters': params,
            'num_parameters': len(params)
        }

    @classmethod
    def unregister(cls, prompt_name: str) -> None:
        """
        Unregister a prompt (mainly for testing).

        Args:
            prompt_name: Name of the prompt to unregister
        """
        if prompt_name in cls._REGISTRY:
            del cls._REGISTRY[prompt_name]
            logger.debug(f"[PromptRegistry] Unregistered prompt: {prompt_name}")

        # Clear cache entries for this prompt
        keys_to_remove = [k for k in cls._cache.keys() if k[0] == prompt_name]
        for key in keys_to_remove:
            del cls._cache[key]


# Convenience alias
PromptRegistry = PromptRegistryMeta


__all__ = ['PromptRegistry', 'PromptRegistryMeta']
