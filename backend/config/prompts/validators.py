"""
Prompt Validators
Validation utilities for ensuring prompt quality and correctness.

Features:
- Length validation
- Content validation
- Structure validation
- Parameter validation
"""

from typing import List, Dict, Any, Optional
import re

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PromptValidator:
    """
    Validates prompt content and structure.

    Usage:
        >>> validator = PromptValidator()
        >>> issues = validator.validate_prompt(prompt_text)
        >>> if issues:
        ...     print(f"Validation issues: {issues}")
    """

    # Maximum recommended prompt length (chars)
    MAX_PROMPT_LENGTH = 10000  # ~2500 tokens

    # Minimum prompt length (chars)
    MIN_PROMPT_LENGTH = 10

    # Required elements for different prompt types
    REQUIRED_ELEMENTS = {
        'system': ['You are', 'Task:', 'Your role'],
        'instruction': ['IMPORTANT:', 'Example:', 'Format:'],
        'generation': ['Generate', 'Create', 'Produce']
    }

    def __init__(self, max_length: int = MAX_PROMPT_LENGTH, min_length: int = MIN_PROMPT_LENGTH):
        """
        Initialize the validator.

        Args:
            max_length: Maximum allowed prompt length
            min_length: Minimum required prompt length
        """
        self.max_length = max_length
        self.min_length = min_length

    def validate_prompt(
        self,
        prompt: str,
        prompt_name: Optional[str] = None,
        check_elements: bool = True
    ) -> List[str]:
        """
        Validate a prompt and return list of issues.

        Args:
            prompt: Prompt text to validate
            prompt_name: Optional name for better error messages
            check_elements: Whether to check for required elements

        Returns:
            List of validation issues (empty if valid)

        Example:
            >>> validator = PromptValidator()
            >>> issues = validator.validate_prompt("Short")
            >>> print(issues)
            ['Prompt too short: 5 chars (minimum: 10)']
        """
        issues = []
        prefix = f"{prompt_name}: " if prompt_name else ""

        # 1. Check if prompt is empty or not string
        if not prompt or not isinstance(prompt, str):
            issues.append(f"{prefix}Empty or invalid prompt")
            return issues

        # 2. Length validation
        length_issues = self._validate_length(prompt)
        if length_issues:
            issues.extend([f"{prefix}{issue}" for issue in length_issues])

        # 3. Structure validation
        structure_issues = self._validate_structure(prompt)
        if structure_issues:
            issues.extend([f"{prefix}{issue}" for issue in structure_issues])

        # 4. Content validation
        if check_elements:
            content_issues = self._validate_content(prompt)
            if content_issues:
                issues.extend([f"{prefix}{issue}" for issue in content_issues])

        return issues

    def _validate_length(self, prompt: str) -> List[str]:
        """
        Validate prompt length.

        Args:
            prompt: Prompt text

        Returns:
            List of length-related issues
        """
        issues = []
        prompt_length = len(prompt)

        if prompt_length < self.min_length:
            issues.append(f"Prompt too short: {prompt_length} chars (minimum: {self.min_length})")

        if prompt_length > self.max_length:
            issues.append(f"Prompt too long: {prompt_length} chars (maximum: {self.max_length})")

        return issues

    def _validate_structure(self, prompt: str) -> List[str]:
        """
        Validate prompt structure (balanced braces, etc.).

        Args:
            prompt: Prompt text

        Returns:
            List of structure-related issues
        """
        issues = []

        # Check balanced braces
        if not self._check_balanced_braces(prompt):
            issues.append("Unbalanced braces in prompt")

        # Check for common formatting issues
        if prompt.count('"""') % 2 != 0:
            issues.append("Unbalanced triple quotes")

        if prompt.count("'''") % 2 != 0:
            issues.append("Unbalanced triple single quotes")

        return issues

    def _validate_content(self, prompt: str) -> List[str]:
        """
        Validate prompt content (required elements, clarity, etc.).

        Args:
            prompt: Prompt text

        Returns:
            List of content-related issues
        """
        issues = []

        # Check for role definition
        has_role = any(phrase in prompt for phrase in ['You are', 'Your role', 'Task:'])
        if not has_role:
            issues.append("Missing role/task definition")

        # Check for excessive repetition (same phrase repeated >5 times)
        words = prompt.split()
        if len(words) > 50:  # Only check for longer prompts
            word_counts = {}
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                word_counts[phrase] = word_counts.get(phrase, 0) + 1

            for phrase, count in word_counts.items():
                if count > 5:
                    issues.append(f"Excessive repetition: '{phrase}' appears {count} times")

        return issues

    def _check_balanced_braces(self, text: str) -> bool:
        """
        Check if braces are balanced in the text.

        Args:
            text: Text to check

        Returns:
            True if balanced, False otherwise
        """
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}

        for char in text:
            if char in pairs.keys():
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                last_open = stack.pop()
                if pairs[last_open] != char:
                    return False

        return len(stack) == 0

    def validate_parameters(
        self,
        provided_params: Dict[str, Any],
        required_params: List[str],
        optional_params: Optional[List[str]] = None
    ) -> List[str]:
        """
        Validate prompt parameters.

        Args:
            provided_params: Parameters provided by caller
            required_params: Required parameter names
            optional_params: Optional parameter names

        Returns:
            List of parameter validation issues

        Example:
            >>> validator = PromptValidator()
            >>> issues = validator.validate_parameters(
            ...     {'query': 'test'},
            ...     ['query', 'context'],
            ...     ['file_guidance']
            ... )
            >>> print(issues)
            ["Missing required parameter: 'context'"]
        """
        issues = []
        optional_params = optional_params or []

        # Check for missing required parameters
        missing = set(required_params) - set(provided_params.keys())
        if missing:
            issues.append(f"Missing required parameters: {', '.join(sorted(missing))}")

        # Check for unknown parameters
        all_known = set(required_params) | set(optional_params)
        unknown = set(provided_params.keys()) - all_known
        if unknown:
            issues.append(f"Unknown parameters: {', '.join(sorted(unknown))}")

        # Check for empty string values in required parameters
        for param in required_params:
            if param in provided_params:
                value = provided_params[param]
                if isinstance(value, str) and not value.strip():
                    issues.append(f"Required parameter '{param}' is empty")

        return issues


class PromptQualityChecker:
    """
    Advanced prompt quality checking.

    Checks for:
    - Token efficiency
    - Clarity of instructions
    - Completeness
    """

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count (rough approximation).

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters for English
        return len(text) // 4

    @staticmethod
    def check_clarity(prompt: str) -> Dict[str, Any]:
        """
        Check prompt clarity.

        Args:
            prompt: Prompt text

        Returns:
            Dictionary with clarity metrics
        """
        # Count instruction markers
        instruction_count = len(re.findall(r'\d+\.\s+', prompt))

        # Count example markers
        example_count = prompt.count('Example:') + prompt.count('e.g.')

        # Count clarification phrases
        clarification_count = sum([
            prompt.count('IMPORTANT:'),
            prompt.count('Note:'),
            prompt.count('Remember:'),
            prompt.count('WARNING:')
        ])

        return {
            'instructions': instruction_count,
            'examples': example_count,
            'clarifications': clarification_count,
            'has_format_spec': 'Format:' in prompt or 'format:' in prompt,
            'estimated_tokens': PromptQualityChecker.estimate_tokens(prompt)
        }

    @staticmethod
    def suggest_improvements(prompt: str) -> List[str]:
        """
        Suggest improvements for a prompt.

        Args:
            prompt: Prompt text

        Returns:
            List of improvement suggestions
        """
        suggestions = []
        clarity = PromptQualityChecker.check_clarity(prompt)

        # Check token efficiency
        if clarity['estimated_tokens'] > 2500:
            suggestions.append(
                f"Prompt is very long ({clarity['estimated_tokens']} tokens). "
                "Consider splitting into multiple prompts or using more concise language."
            )

        # Check for examples
        if clarity['examples'] == 0 and len(prompt) > 500:
            suggestions.append("Consider adding examples to clarify expected output format")

        # Check for format specification
        if not clarity['has_format_spec'] and 'generate' in prompt.lower():
            suggestions.append("Consider specifying the expected output format")

        # Check for excessive clarifications
        if clarity['clarifications'] > 5:
            suggestions.append(
                f"Prompt has many clarifications ({clarity['clarifications']}). "
                "Consider consolidating or simplifying instructions."
            )

        return suggestions


def validate_prompt_registry(registry_dict: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate all prompts in a registry.

    Args:
        registry_dict: Dictionary mapping prompt names to generator functions

    Returns:
        Dictionary mapping prompt names to lists of issues

    Example:
        >>> from backend.config.prompts import PromptRegistry
        >>> issues = validate_prompt_registry(PromptRegistry._REGISTRY)
        >>> for name, problems in issues.items():
        ...     if problems:
        ...         print(f"{name}: {problems}")
    """
    validator = PromptValidator()
    all_issues = {}

    for prompt_name, generator in registry_dict.items():
        try:
            # Generate dummy parameters
            import inspect
            sig = inspect.signature(generator)
            dummy_params = {}

            for param_name, param in sig.parameters.items():
                if param.default == inspect.Parameter.empty:
                    # Required parameter - provide dummy value
                    if param.annotation == int:
                        dummy_params[param_name] = 0
                    elif param.annotation == bool:
                        dummy_params[param_name] = False
                    elif param.annotation == list:
                        dummy_params[param_name] = []
                    else:
                        dummy_params[param_name] = f"test_{param_name}"

            # Generate prompt with dummy parameters
            prompt = generator(**dummy_params)

            # Validate
            issues = validator.validate_prompt(prompt, prompt_name=prompt_name)
            all_issues[prompt_name] = issues

        except Exception as e:
            all_issues[prompt_name] = [f"Failed to generate: {e}"]

    return all_issues


__all__ = [
    'PromptValidator',
    'PromptQualityChecker',
    'validate_prompt_registry'
]
