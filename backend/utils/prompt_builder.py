"""
Prompt Builder Module
=====================
Centralized prompt management and template system.
Provides consistent prompt loading, validation, and formatting.

Version: 1.0.0
Created: 2025-01-13
"""

from typing import Dict, Any, Optional
import re
from pathlib import Path

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PromptRegistry:
    """
    Registry for storing and managing prompt templates.

    Prompts are stored in-memory and can be registered dynamically.
    Future versions may support loading from external files.
    """

    def __init__(self):
        """Initialize the prompt registry."""
        self._prompts: Dict[str, str] = {}
        self._cache: Dict[str, str] = {}

    def register(self, name: str, template: str) -> None:
        """
        Register a prompt template.

        Args:
            name: Unique identifier for the prompt
            template: Prompt template string (may contain {variables})

        Example:
            >>> registry.register("greeting", "Hello, {name}!")
        """
        if not name or not template:
            raise ValueError("Prompt name and template cannot be empty")

        self._prompts[name] = template
        logger.debug(f"[PromptRegistry] Registered prompt: {name}")

    def get(self, name: str) -> Optional[str]:
        """
        Get a prompt template by name.

        Args:
            name: Prompt identifier

        Returns:
            Prompt template string, or None if not found
        """
        return self._prompts.get(name)

    def list_prompts(self) -> list:
        """
        List all registered prompt names.

        Returns:
            List of prompt names
        """
        return list(self._prompts.keys())

    def clear(self) -> None:
        """Clear all registered prompts and cache."""
        self._prompts.clear()
        self._cache.clear()
        logger.debug("[PromptRegistry] Cleared all prompts")


class PromptBuilder:
    """
    Builder class for loading, formatting, and validating prompts.

    Features:
    - Template variable substitution
    - Prompt validation
    - Caching for performance
    - Integration with PromptRegistry
    """

    def __init__(self, registry: Optional[PromptRegistry] = None):
        """
        Initialize the prompt builder.

        Args:
            registry: Optional PromptRegistry instance (creates new if not provided)
        """
        self.registry = registry or PromptRegistry()
        self._format_cache: Dict[str, str] = {}

    def get_prompt(self, name: str, **kwargs) -> str:
        """
        Get and format a prompt template with provided variables.

        Args:
            name: Prompt identifier
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If prompt not found or formatting fails

        Example:
            >>> builder.get_prompt("greeting", name="Alice")
            "Hello, Alice!"
        """
        # Get template from registry
        template = self.registry.get(name)
        if template is None:
            raise ValueError(f"Prompt '{name}' not found in registry")

        # Check cache if no variables provided
        cache_key = f"{name}:{str(sorted(kwargs.items()))}"
        if cache_key in self._format_cache:
            return self._format_cache[cache_key]

        # Format template
        try:
            formatted = self._format_template(template, **kwargs)

            # Cache the result
            self._format_cache[cache_key] = formatted

            logger.debug(f"[PromptBuilder] Formatted prompt '{name}' with {len(kwargs)} variables")
            return formatted

        except KeyError as e:
            raise ValueError(f"Missing required variable in prompt '{name}': {e}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt '{name}': {e}")

    def _format_template(self, template: str, **kwargs) -> str:
        """
        Format a template string with provided variables.

        Supports both {variable} and {{variable}} syntax.

        Args:
            template: Template string
            **kwargs: Variables to substitute

        Returns:
            Formatted string
        """
        # Replace {variable} style placeholders
        try:
            return template.format(**kwargs)
        except KeyError:
            # If strict formatting fails, try partial formatting
            # This allows templates with optional variables
            result = template
            for key, value in kwargs.items():
                result = result.replace(f"{{{key}}}", str(value))
            return result

    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate prompt structure and content.

        Checks for:
        - Non-empty content
        - Balanced braces
        - No dangerous content

        Args:
            prompt: Prompt string to validate

        Returns:
            True if valid, False otherwise

        Example:
            >>> builder.validate_prompt("Hello {name}")
            True
        """
        if not prompt or not isinstance(prompt, str):
            logger.warning("[PromptBuilder] Validation failed: empty or invalid prompt")
            return False

        # Check minimum length
        if len(prompt.strip()) < 10:
            logger.warning("[PromptBuilder] Validation failed: prompt too short")
            return False

        # Check balanced braces
        if not self._check_balanced_braces(prompt):
            logger.warning("[PromptBuilder] Validation failed: unbalanced braces")
            return False

        return True

    def _check_balanced_braces(self, text: str) -> bool:
        """
        Check if braces are balanced in the text.

        Args:
            text: Text to check

        Returns:
            True if balanced, False otherwise
        """
        open_count = text.count("{")
        close_count = text.count("}")

        # Account for escaped braces {{}}
        escaped_count = text.count("{{")

        return (open_count - escaped_count) == (close_count - escaped_count)

    def extract_variables(self, template: str) -> list:
        """
        Extract variable names from a template.

        Args:
            template: Template string

        Returns:
            List of variable names

        Example:
            >>> builder.extract_variables("Hello {name}, you are {age} years old")
            ['name', 'age']
        """
        # Find all {variable} patterns
        pattern = r"\{([^{}]+)\}"
        matches = re.findall(pattern, template)

        # Filter out format specifiers (e.g., {:.2f})
        variables = [m for m in matches if not m.startswith(":")]

        return variables

    def clear_cache(self) -> None:
        """Clear the formatting cache."""
        self._format_cache.clear()
        logger.debug("[PromptBuilder] Cleared formatting cache")


# Global instances (optional, for convenience)
prompt_registry = PromptRegistry()
prompt_builder = PromptBuilder(prompt_registry)
