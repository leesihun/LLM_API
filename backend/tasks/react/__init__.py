"""
ReAct Agent Module

Simplified, modular ReAct agent implementation.

Consolidated structure (6 files):
- agent.py: Main ReActAgent orchestration
- models.py: Data structures (ToolName, ReActStep, ReActResult)
- generators.py: Thought/action/answer generation + context formatting
- execution.py: Tool execution + verification
- planning.py: Plan execution + dynamic adaptation
"""

# Main agent class and factory
from .agent import ReActAgent, ReActAgentFactory, react_agent

# Data models
from .models import ToolName, ReActStep, ReActResult

# Specialized modules (for advanced usage)
from .generators import ThoughtActionGenerator, AnswerGenerator, ContextFormatter
from .execution import ToolExecutor, StepVerifier
from .planning import PlanExecutor, PlanAdapter

__all__ = [
    # Main exports (public API)
    "ReActAgent",
    "ReActAgentFactory",
    "react_agent",

    # Data models
    "ToolName",
    "ReActStep",
    "ReActResult",

    # Specialized modules (for advanced usage/testing)
    "ThoughtActionGenerator",
    "AnswerGenerator",
    "ContextFormatter",
    "ToolExecutor",
    "StepVerifier",
    "PlanExecutor",
    "PlanAdapter",
]
