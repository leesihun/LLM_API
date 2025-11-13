"""
ReAct Step Verification Module

This module handles step verification, auto-finish detection, and completion checking
for ReAct agent execution. Extracted from React.py for improved modularity.

Key Features:
- Auto-finish detection: Triggers early exit when observation contains complete answer
- Step verification: Validates if a plan step achieved its goal
- Completion checking: Determines if multi-step execution is complete
"""

from typing import List, Optional
from langchain_core.messages import HumanMessage

from .models import ReActStep
from backend.models.schemas import PlanStep
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class StepVerifier:
    """
    Handles verification of ReAct execution steps and completion detection.

    This class provides methods to:
    1. Detect when an observation contains a complete answer (early exit)
    2. Verify if a plan step achieved its goal
    3. Check if overall execution is complete

    Attributes:
        llm: Language model instance for verification checks
    """

    def __init__(self, llm):
        """
        Initialize step verifier.

        Args:
            llm: LangChain LLM instance for verification tasks
        """
        self.llm = llm

    def should_auto_finish(self, observation: str, step_num: int) -> bool:
        """
        Detect if observation contains complete answer (early exit optimization).

        Simplified auto-finish with 3 core heuristics for efficiency:
        1. Length check: Observation > 200 characters (substantial content)
        2. Error check: No error keywords present
        3. Keyword check: Contains answer keywords OR very long observation (> 500 chars)

        Args:
            observation: Latest observation from tool execution
            step_num: Current step number

        Returns:
            True if should auto-finish, False to continue iterations
        """
        # Need at least 2 steps before considering early exit
        if step_num < 2:
            return False

        # Heuristic 1: Check minimum length
        if not observation or len(observation) < 200:
            return False

        observation_lower = observation.lower()

        # Heuristic 2: Has error - DON'T finish
        error_keywords = ['error', 'failed', 'exception', 'traceback']
        if any(keyword in observation_lower for keyword in error_keywords):
            return False

        # Heuristic 3: Has answer keywords - DO finish
        answer_keywords = ['answer is', 'result is', 'found that', 'shows that', 'indicates']
        if any(keyword in observation_lower for keyword in answer_keywords):
            logger.info("⚡ EARLY EXIT: Observation contains answer keywords")
            return True

        # Heuristic 3b: Very long observation (> 500 chars) with data - probably complete
        if len(observation) > 500:
            logger.info("⚡ EARLY EXIT: Observation is substantial (> 500 chars)")
            return True

        return False

    async def verify_step_success(
        self,
        plan_step: PlanStep,
        observation: str,
        tool_used: str
    ) -> bool:
        """
        Verify if a plan step achieved its goal based on observation.

        This method uses a combination of heuristics and LLM verification
        to determine if a step successfully achieved its goal according
        to the success criteria.

        Args:
            plan_step: Current plan step with goal and success criteria
            observation: Observation from tool execution
            tool_used: Which tool was used

        Returns:
            True if step goal achieved, False otherwise
        """
        # Check for obvious failures
        if not observation or len(observation.strip()) < 5:
            return False

        if "error" in observation.lower() or "failed" in observation.lower():
            # But allow if there's also success indicators
            if "success" not in observation.lower():
                return False

        # For simple cases, use heuristics
        if "success" in observation.lower() and len(observation) > 50:
            return True

        # Use LLM to verify goal achievement
        verification_prompt = f"""Verify if the step goal was achieved.

Step Goal: {plan_step.goal}
Success Criteria: {plan_step.success_criteria}

Tool Used: {tool_used}
Observation: {observation[:1000]}

Based on the observation, was the step goal achieved according to the success criteria?
Answer with "YES" or "NO" and brief reasoning:"""

        response = await self.llm.ainvoke([HumanMessage(content=verification_prompt)])
        result = response.content.strip().lower()

        is_success = result.startswith("yes")
        logger.info(f"[Step Verification] {'SUCCESS' if is_success else 'FAILED'} - {result[:100]}")

        return is_success

    def check_completion(self, steps: List[ReActStep], query: str) -> bool:
        """
        Check if execution is complete based on step history.

        This is a simple heuristic check that can be used to determine
        if execution has reached a natural conclusion without needing
        explicit FINISH action.

        Args:
            steps: List of ReAct steps executed
            query: Original user query

        Returns:
            True if execution appears complete, False otherwise
        """
        if not steps:
            return False

        # Check last step
        last_step = steps[-1]

        # If last observation is substantial and contains conclusive language
        if len(last_step.observation) > 200:
            obs_lower = last_step.observation.lower()
            conclusive_phrases = [
                "the answer",
                "in conclusion",
                "final result",
                "therefore",
            ]
            if any(phrase in obs_lower for phrase in conclusive_phrases):
                return True

        return False
