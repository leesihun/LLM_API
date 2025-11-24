"""
ReAct Step Verification Module

This module handles step verification, auto-finish detection, and completion checking
for ReAct agent execution. Extracted from React.py for improved modularity.

Key Features:
- Auto-finish detection: Triggers early exit when observation contains complete answer
- Enhanced auto-finish with LLM-based adequacy checking (confidence scoring)
- Step verification: Validates if a plan step achieved its goal
- Completion checking: Determines if multi-step execution is complete
"""

from typing import List, Optional, Tuple
from langchain_core.messages import HumanMessage
import json

from .models import ReActStep
from backend.models.schemas import PlanStep
from backend.config.prompts import PromptRegistry
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
            logger.info("[EARLY EXIT] Observation contains answer keywords")
            return True

        # Heuristic 3b: Very long observation (> 500 chars) with data - probably complete
        if len(observation) > 500:
            logger.info("[EARLY EXIT] Observation is substantial (> 500 chars)")
            return True

        return False

    async def should_auto_finish_enhanced(
        self,
        observation: str,
        user_query: str,
        iteration: int,
        steps_context: str
    ) -> Tuple[bool, float, str]:
        """
        Enhanced auto-finish with LLM-based adequacy check and confidence scoring.

        This method improves upon the heuristic-based should_auto_finish() by:
        1. Quick heuristic checks first (avoid LLM call if obvious)
        2. LLM adequacy assessment for borderline cases
        3. Confidence scoring to prevent premature exits
        4. Clear reasoning for auto-finish decisions

        Args:
            observation: Latest observation from tool execution
            user_query: Original user query
            iteration: Current iteration number
            steps_context: Formatted context from previous steps

        Returns:
            Tuple of (should_finish, confidence_score, reason)
        """
        # Quick heuristic checks first (avoid LLM call)
        if len(observation) < 50:
            return False, 0.0, "Observation too short (< 50 chars)"

        if iteration < 2:
            return False, 0.0, "Too early (need at least 2 iterations)"

        observation_lower = observation.lower()

        # Check for obvious errors
        error_keywords = ['error', 'failed', 'exception', 'traceback']
        if any(keyword in observation_lower for keyword in error_keywords):
            # But allow if there's also success indicators
            if 'success' not in observation_lower:
                return False, 0.0, "Observation contains error keywords"

        # Strong heuristics can skip LLM check
        strong_answer_keywords = [
            'the answer is',
            'therefore the result is',
            'in conclusion',
            'final result is'
        ]
        if any(keyword in observation_lower for keyword in strong_answer_keywords):
            return True, 0.95, "Strong answer keywords detected"

        # LLM adequacy check for borderline cases
        try:
            prompt = f"""Assess if this observation adequately answers the user's query.

User Query:
{user_query}

Latest Observation:
{observation[:1000]}

Previous Steps Context:
{steps_context[-500:] if len(steps_context) > 500 else steps_context}

Question: Does the observation contain enough information to FULLY and ACCURATELY answer the user's query?

Respond with JSON only:
{{
  "adequate": true or false,
  "confidence": 0.0 to 1.0 (how confident are you in this assessment),
  "reason": "brief explanation",
  "missing_info": "what's still needed if inadequate, otherwise empty string"
}}
"""

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            result = json.loads(response_text.strip())

            adequate = result.get("adequate", False)
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "No reason provided")
            missing_info = result.get("missing_info", "")

            # Require high confidence (>= 0.8) to auto-finish
            should_finish = adequate and confidence >= 0.8

            if should_finish:
                logger.info(f"[AUTO-FINISH ENHANCED] Confidence: {confidence:.2f} - {reason}")
            else:
                logger.info(f"[CONTINUE] Confidence: {confidence:.2f} - {reason}")
                if missing_info:
                    logger.info(f"[CONTINUE] Missing: {missing_info}")

            return should_finish, confidence, reason

        except Exception as e:
            logger.error(f"[AUTO-FINISH ENHANCED] LLM check failed: {e}")
            raise

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
        verification_prompt = PromptRegistry.get(
            'react_step_verification',
            plan_step_goal=plan_step.goal,
            success_criteria=plan_step.success_criteria,
            tool_used=tool_used,
            observation=observation[:1000]
        )

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
