"""
ReAct Agent Module

Reasoning + Acting pattern implementation for agentic workflows.

This module provides a clean, modular ReAct agent implementation with:
- Main orchestration: ReActAgent
- Specialized modules: ThoughtActionGenerator, ToolExecutor, AnswerGenerator, etc.
- Data models: ToolName, ReActStep, ReActResult
- Global instance: react_agent

Architecture:
- agent.py: Main ReActAgent class (orchestration)
- models.py: Data structures (ToolName, ReActStep, ReActResult)
- thought_action_generator.py: Thought and action generation
- tool_executor.py: Tool execution and routing
- answer_generator.py: Final answer synthesis
- context_manager.py: Context formatting and management
- verification.py: Step verification and auto-finish detection
- plan_executor.py: Plan-based execution
"""

# Main agent class
from .agent import ReActAgent, react_agent

# Data models
from .models import ToolName, ReActStep, ReActResult

# Specialized modules (for advanced usage)
from .thought_action_generator import ThoughtActionGenerator
from .tool_executor import ToolExecutor
from .answer_generator import AnswerGenerator
from .context_manager import ContextManager
from .verification import StepVerifier
from .plan_executor import PlanExecutor

__all__ = [
    # Main exports (public API)
    "ReActAgent",
    "react_agent",

    # Data models
    "ToolName",
    "ReActStep",
    "ReActResult",

    # Specialized modules (for advanced usage/testing)
    "ThoughtActionGenerator",
    "ToolExecutor",
    "AnswerGenerator",
    "ContextManager",
    "StepVerifier",
    "PlanExecutor",
]
