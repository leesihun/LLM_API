"""
ReAct Generators Module

Handles all LLM-based generation for ReAct agent:
- Thought and action generation
- Final answer synthesis
- Context formatting and pruning

Consolidated from:
- thought_action_generator.py
- answer_generator.py
- context_manager.py
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
    Generates thoughts and actions for ReAct agent.

    Uses combined thought-action generation (1 LLM call instead of 2).
    """

    VALID_ACTIONS = {tool.value for tool in ToolName}

    def __init__(self, llm, file_paths: Optional[List[str]] = None):
        """
        Initialize thought-action generator.

        Args:
            llm: LangChain LLM instance
            file_paths: Optional list of file paths (for file-specific guidance)
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
        Generate thought and action in single LLM call.

        Args:
            user_query: User's original query
            steps: Previous ReAct steps
            context: Formatted context string

        Returns:
            Tuple of (thought, action, action_input)
        """
        file_guidance = self._build_file_guidance()

        prompt = prompts.get_react_thought_and_action_prompt(
            query=user_query,
            context=context,
            file_guidance=file_guidance
        )

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()

        thought, action, action_input = self._parse_thought_and_action(response_text)

        return thought, action, action_input

    def _parse_thought_and_action(self, response: str) -> Tuple[str, str, str]:
        """
        Parse combined thought-action response from LLM.

        Strategy:
        1. Try structured format: THOUGHT: ... ACTION: ... ACTION INPUT: ...
        2. Fallback to old format: Action: ... Action Input: ...
        3. Provide defaults for missing fields
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

        # Strategy 2: Fallback parsing
        if not thought:
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            if paragraphs:
                thought = paragraphs[0]

        if not action:
            action_match = re.search(r'action\s*:\s*(\w+)', response, re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip().lower()

        if not action_input:
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
        """Build guidance text for file-related actions."""
        if not self.file_paths:
            return ""

        return (
            "\nGuidelines:\n"
            "- If any files are available, first attempt local analysis using python_coder.\n"
            "- Only use rag_retrieval or web_search if local analysis failed or is insufficient.\n"
            "- You may call different tools across iterations to complete the task."
        )


class AnswerGenerator:
    """
    Generates final answers for ReAct agent.

    Synthesizes all accumulated observations into a coherent final answer.
    """

    def __init__(self, llm):
        """
        Initialize answer generator.

        Args:
            llm: LangChain LLM instance
        """
        self.llm = llm

    async def generate_final_answer(self, user_query: str, steps: List[ReActStep]) -> str:
        """
        Generate final answer by synthesizing all observations.

        Uses context formatting with pruning optimization.

        Args:
            user_query: Original user question
            steps: List of ReActStep objects

        Returns:
            Final answer string
        """
        context = format_steps_context(steps)

        prompt = prompts.get_react_final_answer_prompt(query=user_query, context=context)

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        logger.info("Final answer generation completed")

        return response.content.strip()

    def extract_from_steps(self, user_query: str, steps: List[ReActStep]) -> str:
        """
        Extract answer directly from observations (fallback strategy).

        Multi-tier fallback:
        1. Recent observation extraction
        2. Attempted tools summary
        3. Generic fallback

        Args:
            user_query: Original user question
            steps: List of ReActStep objects

        Returns:
            Extracted answer string
        """
        if not steps:
            return "I apologize, but I was unable to generate a proper response. Please try rephrasing your question."

        # Find most informative observation from recent steps
        for step in reversed(steps):
            obs = step.observation.strip()
            if obs and len(obs) >= 20 and not obs.startswith("Error") and not obs.startswith("No "):
                logger.info(f"Extracted answer from step {step.step_num} observation")
                return f"Based on my research: {obs}"

        # Summarize what was attempted
        actions_taken = [step.action for step in steps if step.action != ToolName.FINISH]
        if actions_taken:
            return f"I attempted to answer your question using {', '.join(set(actions_taken))}, but was unable to find sufficient information. Please try rephrasing your question or providing more context."

        return "I apologize, but I was unable to generate a proper response. Please try rephrasing your question."


class ContextFormatter:
    """
    Formats context for ReAct execution with pruning optimization.

    Pruning strategy:
    - ≤3 steps: show all steps in full
    - >3 steps: summary of early steps + last 2 steps in full
    """

    def __init__(self, session_id: Optional[str] = None):
        """Initialize context formatter."""
        self.session_id = session_id

    def build_tool_context(self, steps: List[ReActStep]) -> str:
        """
        Build context string with automatic pruning.

        Args:
            steps: List of ReActStep objects

        Returns:
            Formatted context string
        """
        if not steps:
            return ""

        if len(steps) <= 3:
            return format_all_steps(steps)
        else:
            return format_pruned_steps(steps)


# Module-level utility functions for context formatting
def format_steps_context(steps: List[ReActStep]) -> str:
    """
    Format steps with intelligent context pruning.

    Public utility function for formatting step context.
    """
    if not steps:
        return ""

    if len(steps) <= 3:
        return format_all_steps(steps)
    else:
        return format_pruned_steps(steps)


def format_all_steps(steps: List[ReActStep]) -> str:
    """Format all steps in full detail (≤3 steps)."""
    context_parts = ["Previous Steps:"]
    for step in steps:
        context_parts.append(f"""
Step {step.step_num}:
- Thought: {step.thought}
- Action: {step.action}
- Action Input: {step.action_input}
- Observation: {step.observation}
""")
    return "\n".join(context_parts)


def format_pruned_steps(steps: List[ReActStep]) -> str:
    """Format steps with pruning (summary of early + full detail of recent)."""
    context_parts = ["Previous Steps:\n"]

    # Summary of early steps
    early_steps = steps[:-2]
    tools_used = list(set([s.action for s in early_steps if s.action != ToolName.FINISH]))
    summary = f"Steps 1-{len(early_steps)} completed using: {', '.join(tools_used)}"
    context_parts.append(f"[Summary] {summary}\n")

    # Recent steps in full detail
    context_parts.append("\n[Recent Steps - Full Detail]")
    recent_steps = steps[-2:]
    for step in recent_steps:
        obs_display = step.observation[:500] if len(step.observation) > 500 else step.observation
        context_parts.append(f"""
Step {step.step_num}:
- Thought: {step.thought[:200]}...
- Action: {step.action}
- Action Input: {step.action_input[:200] if len(step.action_input) > 200 else step.action_input}
- Observation: {obs_display}
""")

    return "\n".join(context_parts)
