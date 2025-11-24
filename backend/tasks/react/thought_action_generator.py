"""
ReAct Thought and Action Generator

Handles generation and parsing of thoughts and actions for the ReAct agent.
Uses PromptRegistry for all prompts and llm_factory for LLM instances.

Key Features:
- Combined thought-action generation (1 LLM call instead of 2)
- Robust parsing with multiple fallback strategies
"""

import re
from typing import Tuple, List, Optional
from langchain_core.messages import HumanMessage

from .models import ToolName, ReActStep
from backend.config import prompts
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ThoughtActionGenerator:
    """
    Handles generation and parsing of thoughts and actions for ReAct agent.

    Architecture:
    - Main method: generate() - Single call for thought + action
    - Parser methods: _parse_thought_and_action()
    """

    VALID_ACTIONS = {tool.value for tool in ToolName}

    def __init__(self, llm, file_paths: Optional[List[str]] = None):
        """
        Initialize thought-action generator.

        Args:
            llm: LangChain LLM instance (ChatOllama or compatible)
            file_paths: Optional list of file paths (used for file-specific guidance)
        """
        self.llm = llm
        self.file_paths = file_paths

    async def generate(
        self,
        user_query: str,
        steps: List[ReActStep],
        context: str
    ) -> Tuple[str, str, str]:
        """
        Generate thought and action in single LLM call (performance optimization).

        Args:
            user_query: User's original query
            steps: Previous ReAct steps (for context)
            context: Formatted context string from steps

        Returns:
            Tuple of (thought, action, action_input)
        """
        # Build file guidance if files are attached
        file_guidance = self._build_file_guidance()

        # Use centralized prompt from PromptRegistry
        prompt = prompts.get_react_thought_and_action_prompt(
            query=user_query,
            context=context,
            file_guidance=file_guidance
        )

        # Single LLM call for combined thought-action
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()

        # Parse the combined response
        thought, action, action_input = self._parse_thought_and_action(response_text)

        logger.info("Combined thought-action generation completed")

        return thought, action, action_input

    def _parse_thought_and_action(self, response: str) -> Tuple[str, str, str]:
        """
        Parse combined thought-action response from LLM.

        Parsing strategy:
        1. Try structured format: THOUGHT: ... ACTION: ... ACTION INPUT: ...
        2. Fallback to old format: Action: ... Action Input: ...
        3. Provide defaults for missing fields

        Args:
            response: LLM response containing thought and action

        Returns:
            Tuple of (thought, action, action_input)
        """
        thought = ""
        action = ""
        action_input = ""

        # Strategy 1: Extract structured format
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|$)', response, re.IGNORECASE | re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(r'ACTION:\s*(\w+)', response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip().lower()

        input_match = re.search(r'ACTION\s+INPUT:\s*(.+?)(?=\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if input_match:
            action_input = input_match.group(1).strip()

        # Strategy 2: Fallback parsing if structured format not found
        if not thought:
            # Use first paragraph as thought
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            if paragraphs:
                thought = paragraphs[0]

        if not action:
            # Try old format
            action_match = re.search(r'action\s*:\s*(\w+)', response, re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip().lower()

        if not action_input:
            # Try old format
            input_match = re.search(r'action\s+input\s*:\s*(.+?)(?=\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
            if input_match:
                action_input = input_match.group(1).strip()

        # Strategy 3: Provide defaults
        if not thought:
            thought = "Proceeding with action execution."
        if not action or action not in self.VALID_ACTIONS:
            action = ToolName.FINISH
        if not action_input:
            action_input = "No specific input provided."

        return thought, action, action_input

    def _build_file_guidance(self) -> str:
        """
        Build guidance text for file-related actions.

        Returns:
            Guidance string (empty if no files attached)
        """
        if not self.file_paths:
            return ""

        return (
            "\nGuidelines:\n"
            "- If any files are available, first attempt local analysis using python_coder.\n"
            "- Only use rag_retrieval or web_search if local analysis failed or is insufficient.\n"
            "- You may call different tools across iterations to complete the task."
        )

    # ============================================================================
    # LEGACY METHODS (for backward compatibility)
    # ============================================================================

    async def generate_thought_only(
        self,
        user_query: str,
        steps: List[ReActStep],
        context: str
    ) -> str:
        """
        LEGACY: Generate reasoning about what to do next.

        Note: The primary method now uses combined thought-action generation.
        """
        from backend.config.settings import settings

        prompt = prompts.get_react_thought_prompt(
            query=user_query,
            context=context,
            available_tools=settings.available_tools
        )

        logger.info("Thought generation requested (legacy method)")

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    async def select_action(
        self,
        user_query: str,
        thought: str,
        steps: List[ReActStep],
        context: str
    ) -> Tuple[str, str]:
        """
        LEGACY: Select which tool to use and what input to provide.

        Note: The primary method now uses combined thought-action generation.
        """
        file_guidance = self._build_file_guidance()

        prompt = prompts.get_react_action_selection_prompt(
            query=user_query,
            context=context,
            thought=thought,
            file_guidance=file_guidance
        )

        logger.info("Action selection requested (legacy method)")

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        # Parse response (simplified)
        action, action_input = self._parse_action_simple(response.content)

        return action, action_input

    def _parse_action_simple(self, response: str) -> Tuple[str, str]:
        """Simplified action parsing for legacy method."""
        action = ""
        action_input = ""

        # Try to extract action and input
        lines = response.strip().split('\n')
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower().startswith("action:"):
                action = line_stripped.split(":", 1)[1].strip().lower()
            elif line_stripped.lower().startswith("action input:"):
                action_input = line_stripped.split(":", 1)[1].strip()

        # Regex fallback
        if not action:
            action_match = re.search(r'action\s*:\s*(\w+)', response, re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip().lower()

        if not action_input:
            input_match = re.search(
                r'action\s+input\s*:\s*(.+?)(?=\n\n|\naction\s*:|\Z)',
                response,
                re.IGNORECASE | re.DOTALL
            )
            if input_match:
                action_input = input_match.group(1).strip()

        # Validate action
        if not action or action not in self.VALID_ACTIONS:
            action = ToolName.FINISH

        # Handle FINISH with insufficient input
        if action == ToolName.FINISH and (not action_input or len(action_input.strip()) < 1):
            action_input = response.strip() if response.strip() else "No information available."

        return action, action_input
