"""
Prompt registry for agents and classifiers.

All prompt strings live under backend/config/prompts to make them easy to
modify without touching agent logic.
"""

from backend.config.prompts.agent_prompts import (
    REACT_SYSTEM_PROMPT,
    REACT_STEP_PROMPT,
    REACT_NATIVE_TOOL_PROMPT,
    PLAN_CREATE_PROMPT,
    PLAN_EXECUTION_SYSTEM_PROMPT,
    AGENT_CLASSIFIER_PROMPT,
)
from backend.config.prompts.tool_prompts import (
    PYTHON_CODER_PROMPT,
    WEB_SEARCH_ANSWER_PROMPT,
)

__all__ = [
    "REACT_SYSTEM_PROMPT",
    "REACT_STEP_PROMPT",
    "REACT_NATIVE_TOOL_PROMPT",
    "PLAN_CREATE_PROMPT",
    "PLAN_EXECUTION_SYSTEM_PROMPT",
    "AGENT_CLASSIFIER_PROMPT",
    "PYTHON_CODER_PROMPT",
    "WEB_SEARCH_ANSWER_PROMPT",
]

