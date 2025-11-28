"""
Qwen Thinking Mode Helpers
Utilities for adding severity-based /think and /no_think prefixes to prompts

This module provides simple helper functions to control Qwen's thinking behavior
by adding appropriate prefixes to prompts. Each thinking effort level (none, low, mid, high)
includes severity-specific instructions that guide the depth of reasoning.

Examples:
    >>> from backend.utils.thinking_helpers import add_thinking_prefix
    >>>
    >>> # Fast classification (no thinking)
    >>> prompt = add_thinking_prefix("Classify this query", effort='none')
    >>> # Result: '/no_think\\n\\nClassify this query'
    >>>
    >>> # Deep code generation
    >>> prompt = add_thinking_prefix("Generate Python code", effort='high')
    >>> # Result: '/think Deep reasoning: analyze thoroughly with 4+ steps...\\n\\nGenerate Python code'
"""

from backend.config.settings import settings


def add_thinking_prefix(prompt: str, effort: str = None) -> str:
    """
    Add Qwen thinking mode prefix with severity guidance to prompt.

    The prefix includes both the /think or /no_think control and
    severity-specific instructions that guide the depth of reasoning.

    Severity Levels:
    - none: /no_think - Instant response, no reasoning
    - low: /think + brief guidance - Quick 1-2 step reasoning
    - mid: /think + moderate guidance - Balanced 2-4 step reasoning
    - high: /think + deep guidance - Thorough 4+ step analysis

    Args:
        prompt: Original prompt text
        effort: Thinking effort level ('none', 'low', 'mid', 'high')
                If None, uses settings.thinking_effort_default

    Returns:
        Prompt with thinking prefix and severity guidance

    Examples:
        >>> add_thinking_prefix("Solve this problem", effort='high')
        '/think Deep reasoning: analyze thoroughly with 4+ steps. Break down the problem, consider alternatives, verify your approach, then provide a comprehensive answer.\\n\\nSolve this problem'

        >>> add_thinking_prefix("Quick question", effort='none')
        '/no_think\\n\\nQuick question'

        >>> add_thinking_prefix("Analyze data", effort='low')
        '/think Brief reasoning: consider 1-2 key steps, then respond concisely.\\n\\nAnalyze data'
    """
    if effort is None:
        effort = settings.thinking_effort_default

    prefix = settings.get_thinking_prompt_prefix(effort)

    # Add newlines for readability
    return f"{prefix}\n\n{prompt}"


def get_thinking_prefix(effort: str = None) -> str:
    """
    Get thinking prefix with severity guidance without adding to prompt.

    Args:
        effort: Thinking effort level ('none', 'low', 'mid', 'high')
                If None, uses settings.thinking_effort_default

    Returns:
        Thinking prefix string with severity guidance

    Examples:
        >>> get_thinking_prefix('low')
        '/think Brief reasoning: consider 1-2 key steps, then respond concisely.'

        >>> get_thinking_prefix('none')
        '/no_think'

        >>> get_thinking_prefix()  # Uses default
        '/think Moderate reasoning: think through 2-4 steps. Analyze the problem, plan your approach, then respond.'
    """
    if effort is None:
        effort = settings.thinking_effort_default

    return settings.get_thinking_prompt_prefix(effort)


def wrap_system_message(system_msg: str, effort: str = None) -> str:
    """
    Wrap system message with thinking control prefix.

    Useful for adding thinking control to system-level instructions
    in multi-turn conversations or agent configurations.

    Args:
        system_msg: System message content
        effort: Thinking effort level ('none', 'low', 'mid', 'high')
                If None, uses settings.thinking_effort_default

    Returns:
        System message with thinking prefix

    Examples:
        >>> wrap_system_message("You are a helpful assistant", effort='mid')
        '/think Moderate reasoning: think through 2-4 steps. Analyze the problem, plan your approach, then respond.\\n\\nYou are a helpful assistant'

        >>> wrap_system_message("You are a fast classifier", effort='none')
        '/no_think\\n\\nYou are a fast classifier'
    """
    if effort is None:
        effort = settings.thinking_effort_default

    prefix = settings.get_thinking_prompt_prefix(effort)
    return f"{prefix}\n\n{system_msg}"
