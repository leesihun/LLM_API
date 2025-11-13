"""
ReAct Thought and Action Generator

This module handles the generation and parsing of thoughts and actions for the ReAct agent.
Extracted from React.py to improve modularity and maintainability.

Key Features:
- Combined thought-action generation (1 LLM call instead of 2)
- Fuzzy matching for action names to handle LLM variations
- Legacy methods for backward compatibility
- Robust parsing with multiple fallback strategies
"""

import re
import logging
from typing import Tuple, List, Optional
from langchain_core.messages import HumanMessage

from backend.tasks.react.models import ToolName, ReActStep
from backend.config import prompts
from backend.config.settings import settings
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ThoughtActionGenerator:
    """
    Handles generation and parsing of thoughts and actions for ReAct agent.

    This class encapsulates the logic for:
    1. Generating combined thought-action responses (performance optimization)
    2. Parsing LLM responses into structured thought, action, and action_input
    3. Applying fuzzy matching to handle action name variations
    4. Providing legacy methods for backward compatibility

    Architecture:
    - Main method: generate() - Single call for thought + action
    - Legacy methods: generate_thought_only(), select_action() - Two-step approach
    - Parser methods: _parse_thought_and_action(), _parse_action_response()
    - Fuzzy matching: _apply_fuzzy_action_matching()
    """

    # Fuzzy action mapping for common LLM variations
    FUZZY_ACTION_MAPPING = {
        "web": ToolName.WEB_SEARCH,
        "search": ToolName.WEB_SEARCH,
        "rag": ToolName.RAG_RETRIEVAL,
        "retrieval": ToolName.RAG_RETRIEVAL,
        "retrieve": ToolName.RAG_RETRIEVAL,
        "document": ToolName.RAG_RETRIEVAL,
        "python": ToolName.PYTHON_CODER,
        "code": ToolName.PYTHON_CODER,
        "coder": ToolName.PYTHON_CODER,
        "generate": ToolName.PYTHON_CODER,
        "generate_code": ToolName.PYTHON_CODER,
        "done": ToolName.FINISH,
        "answer": ToolName.FINISH,
        "complete": ToolName.FINISH,
    }

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
        PERFORMANCE OPTIMIZATION: Generate thought and action in single LLM call.

        This is the primary method for thought-action generation. It combines
        reasoning and action selection into a single LLM call, reducing latency
        by 50% compared to the two-step approach.

        Args:
            user_query: User's original query
            steps: Previous ReAct steps (for context)
            context: Formatted context string from steps

        Returns:
            Tuple of (thought, action, action_input)

        Example:
            thought = "I need to search for current information"
            action = "web_search"
            action_input = "latest AI developments 2025"
        """
        # Build file guidance if files are attached
        file_guidance = self._build_file_guidance()

        # Use centralized prompt from backend.config.prompts
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

        logger.info("")
        logger.info("Combined thought-action generation completed")
        logger.info("")

        return thought, action, action_input

    def _parse_thought_and_action(self, response: str) -> Tuple[str, str, str]:
        """
        Parse combined thought-action response from LLM.

        Parsing strategy:
        1. Try structured format: THOUGHT: ... ACTION: ... ACTION INPUT: ...
        2. Fallback to old format: Action: ... Action Input: ...
        3. Apply fuzzy matching for action names
        4. Provide defaults for missing fields

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
            # Try to find action using old parsing method
            action_match = re.search(r'action\s*:\s*(\w+)', response, re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip().lower()

        if not action_input:
            # Try old format
            input_match = re.search(r'action\s+input\s*:\s*(.+?)(?=\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
            if input_match:
                action_input = input_match.group(1).strip()

        # Strategy 3: Validate and apply fuzzy matching for action
        action = self._apply_fuzzy_action_matching(action)

        # Strategy 4: Provide defaults
        if not thought:
            thought = "Proceeding with action execution."
        if not action_input:
            action_input = "No specific input provided."

        return thought, action, action_input

    def _apply_fuzzy_action_matching(self, action: str) -> str:
        """
        Apply fuzzy matching to handle action name variations.

        LLMs may output action names that don't exactly match ToolName values.
        This method maps common variations to valid tool names.

        Args:
            action: Raw action string from LLM

        Returns:
            Valid ToolName value (or FINISH as fallback)

        Examples:
            "search" -> "web_search"
            "code" -> "python_coder"
            "done" -> "finish"
        """
        if not action:
            return ToolName.FINISH

        # Check if action is already valid
        valid_actions = [e.value for e in ToolName]
        if action in valid_actions:
            return action

        # Apply fuzzy matching
        matched_action = self.FUZZY_ACTION_MAPPING.get(action)
        if matched_action:
            return matched_action

        # Default to finish if no match found
        return ToolName.FINISH

    def _build_file_guidance(self) -> str:
        """
        Build guidance text for file-related actions.

        When files are attached, provides specific guidance to prioritize
        local analysis (python_coder) over external tools (web_search, rag).

        Returns:
            Guidance string (empty if no files attached)
        """
        if not self.file_paths:
            return ""

        return (
            "\nGuidelines:\n"
            "- If any files are available, first attempt local analysis using python_coder or python_code.\n"
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
        LEGACY METHOD: Generate reasoning about what to do next.

        This is the old two-step approach where thought and action are generated
        separately. Kept for backward compatibility with existing code.

        Note: The primary method now uses combined thought-action generation
        via the generate() method.

        Args:
            user_query: User's original query
            steps: Previous ReAct steps
            context: Formatted context string

        Returns:
            Thought string (reasoning about next action)
        """
        # Use centralized prompt
        prompt = prompts.get_react_thought_prompt(
            query=user_query,
            context=context,
            available_tools=settings.available_tools
        )

        logger.info("")
        logger.info("Thought generation requested (legacy method)")
        logger.info("")

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
        LEGACY METHOD: Select which tool to use and what input to provide.

        This is the second step of the old two-step approach. The thought is
        already generated, and this method selects the action based on it.

        Note: The primary method now uses combined thought-action generation.

        Args:
            user_query: User's original query
            thought: Previously generated thought/reasoning
            steps: Previous ReAct steps
            context: Formatted context string

        Returns:
            Tuple of (action_name, action_input)
        """
        file_guidance = self._build_file_guidance()

        # Use centralized prompt
        prompt = prompts.get_react_action_selection_prompt(
            query=user_query,
            context=context,
            thought=thought,
            file_guidance=file_guidance
        )

        logger.info("")
        logger.info("Action selection requested (legacy method)")
        logger.info("")

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        # Parse response
        action, action_input = self._parse_action_response(response.content)

        logger.info("")
        logger.info(f"Action: {action}")
        logger.info(f"Action Input: {action_input}")
        logger.info("")

        return action, action_input

    def _parse_action_response(self, response: str) -> Tuple[str, str]:
        """
        Parse LLM response to extract action and input.

        Enhanced parsing with better error handling and fuzzy matching.
        This is used by the legacy select_action() method.

        Args:
            response: LLM response containing action selection

        Returns:
            Tuple of (action_name, action_input)
        """
        lines = response.strip().split('\n')
        action = ""
        action_input = ""

        # Try strict parsing first
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower().startswith("action:"):
                action = line_stripped.split(":", 1)[1].strip().lower()
            elif line_stripped.lower().startswith("action input:"):
                action_input = line_stripped.split(":", 1)[1].strip()

        # If strict parsing failed, try regex-based extraction
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

        # Log warnings if parsing failed
        if not action:
            logger.warning("\n" + "!" * 100)
            logger.warning("[ThoughtActionGenerator] PARSING ERROR - Failed to parse action from response")
            logger.warning("!" * 100)
            logger.warning(f"Raw Response:\n{response}")
            logger.warning("!" * 100 + "\n")

        if not action_input:
            logger.warning("\n" + "!" * 100)
            logger.warning("[ThoughtActionGenerator] PARSING ERROR - Failed to parse action input from response")
            logger.warning("!" * 100)
            logger.warning(f"Raw Response:\n{response}")
            logger.warning("!" * 100 + "\n")

        # Validate and apply fuzzy matching
        action = self._apply_fuzzy_action_matching(action)

        # Handle FINISH action with insufficient input
        if action == ToolName.FINISH and (not action_input or len(action_input.strip()) < 1):
            logger.warning("\n" + "!" * 100)
            logger.warning("[ThoughtActionGenerator] FINISH action with insufficient input")
            logger.warning("!" * 100)
            logger.warning("Attempting to extract answer from response...")
            logger.warning("!" * 100 + "\n")

            extracted = self._extract_answer_from_response(response)
            if extracted and len(extracted.strip()) >= 1:
                action_input = extracted
                logger.info(f"Extracted answer: {action_input[:200]}...")
            elif response.strip():
                action_input = response.strip()
                logger.info(f"Using full response as answer: {action_input[:200]}...")
            else:
                action_input = "I don't have enough information to answer this question."
                logger.warning("No extractable content found, using default message")

        return action, action_input

    def _extract_answer_from_response(self, response: str) -> str:
        """
        Extract answer-like content from LLM response when format parsing fails.

        Looks for declarative sentences, conclusions, or final statements that
        could serve as the final answer.

        Args:
            response: LLM response text

        Returns:
            Extracted answer string (empty if nothing suitable found)
        """
        text = response.strip()

        # Try to find answer patterns
        patterns = [
            r'(?:the answer is|answer:|final answer:|conclusion:|result is?)\s*:?\s*(.+?)(?:\n\n|\Z)',
            r'(?:therefore|thus|so|hence),?\s+(.+?)(?:\n\n|\Z)',
            r'(?:in summary|to conclude|in conclusion),?\s+(.+?)(?:\n\n|\Z)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                if len(answer) >= 10:
                    logger.info("[ThoughtActionGenerator] Extracted answer from response pattern")
                    return answer

        # If no patterns match, try to get the last substantial paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            last_para = paragraphs[-1]
            # Avoid action format lines
            if not re.match(r'action\s*:', last_para, re.IGNORECASE):
                if len(last_para) >= 20:
                    logger.info("[ThoughtActionGenerator] Using last paragraph as answer")
                    return last_para

        # Fallback: return full response if it's reasonable length
        if len(text) >= 20:
            return text

        return ""
