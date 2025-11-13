"""
ReAct Agent Data Models

This module contains the core data structures used by the ReAct agent:
- ToolName: Enumeration of available tools
- ReActStep: Represents a single Thought-Action-Observation cycle
"""

from typing import Dict, Any
from enum import Enum


class ToolName(str, Enum):
    """
    Available tools for ReAct agent execution.

    Each tool represents a capability the agent can invoke during execution:
    - WEB_SEARCH: Search the web using Tavily API
    - RAG_RETRIEVAL: Retrieve documents from vector store (FAISS)
    - PYTHON_CODER: Generate and execute Python code
    - FILE_ANALYZER: Analyze file metadata and structure
    - FINISH: Complete execution and return final answer
    """
    WEB_SEARCH = "web_search"
    RAG_RETRIEVAL = "rag_retrieval"
    PYTHON_CODER = "python_coder"
    FILE_ANALYZER = "file_analyzer"
    FINISH = "finish"


class ReActStep:
    """
    Represents a single Thought-Action-Observation cycle in the ReAct pattern.

    The ReAct pattern follows this loop:
    1. Thought: Reason about what to do next based on current state
    2. Action: Select a tool to execute
    3. Action Input: Provide input parameters for the tool
    4. Observation: Observe the result from tool execution

    Attributes:
        step_num: Sequential step number in the execution (1-indexed)
        thought: Reasoning about the current situation and next action
        action: The tool selected for execution (from ToolName enum)
        action_input: Input/parameters provided to the tool
        observation: Result/output from executing the action

    Example:
        step = ReActStep(1)
        step.thought = "I need to search for recent news about AI"
        step.action = ToolName.WEB_SEARCH
        step.action_input = "latest AI developments 2025"
        step.observation = "Found 5 articles about recent AI advances..."
    """

    def __init__(self, step_num: int):
        """
        Initialize a new ReAct step.

        Args:
            step_num: Sequential step number (typically 1-indexed)
        """
        self.step_num = step_num
        self.thought: str = ""
        self.action: str = ""
        self.action_input: str = ""
        self.observation: str = ""

    def __str__(self) -> str:
        """
        Format step as human-readable string for logging and debugging.

        Returns:
            Formatted multi-line string representation of the step
        """
        return f"""
Step {self.step_num}:
Thought: {self.thought}
Action: {self.action}
Action Input: {self.action_input}
Observation: {self.observation}
"""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert step to dictionary for serialization and storage.

        Useful for:
        - JSON serialization
        - Database storage
        - API responses
        - Conversation history persistence

        Returns:
            Dictionary containing all step attributes
        """
        return {
            "step_num": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation
        }

    def is_complete(self) -> bool:
        """
        Check if step has all required fields populated.

        A complete step must have:
        - Non-empty thought
        - Valid action (any string, will be validated elsewhere)
        - Non-empty action_input
        - Non-empty observation

        Note: action_input and observation may be legitimately short
        (e.g., "No results found"), so we only check for empty strings.

        Returns:
            True if step has all required fields, False otherwise
        """
        return (
            bool(self.thought.strip()) and
            bool(self.action) and
            bool(self.action_input.strip()) and
            bool(self.observation.strip())
        )
