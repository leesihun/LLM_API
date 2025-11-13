"""
ReAct Agent Data Models

This module contains the core data structures used by the ReAct agent:
- ToolName: Enumeration of available tools
- ReActStep: Represents a single Thought-Action-Observation cycle
- ReActResult: Represents the result of a complete ReAct execution
"""

from typing import Dict, Any, List
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


class ReActResult:
    """
    Represents the result of a ReAct agent execution.

    This model encapsulates the final answer and all metadata from
    a complete ReAct execution, providing a structured return value
    for the agent's execute() method.

    Attributes:
        final_answer: The synthesized final answer to the user's query
        metadata: Dictionary containing execution details (steps, tools, etc.)
        steps: List of ReActStep objects representing the execution history
        success: Whether the execution completed successfully

    Example:
        result = ReActResult(
            final_answer="Paris is the capital of France",
            metadata={"total_iterations": 3, "tools_used": ["web_search"]},
            steps=[step1, step2, step3],
            success=True
        )
    """

    def __init__(
        self,
        final_answer: str,
        metadata: Dict[str, Any],
        steps: List[ReActStep],
        success: bool = True
    ):
        """
        Initialize ReAct execution result.

        Args:
            final_answer: The final answer generated by the agent
            metadata: Execution metadata dictionary
            steps: List of ReActStep objects from execution
            success: Whether execution completed successfully
        """
        self.final_answer = final_answer
        self.metadata = metadata
        self.steps = steps
        self.success = success

    def to_tuple(self) -> tuple[str, Dict[str, Any]]:
        """
        Convert to tuple format for backward compatibility.

        Returns:
            Tuple of (final_answer, metadata)
        """
        return self.final_answer, self.metadata
