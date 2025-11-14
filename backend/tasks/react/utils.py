"""
ReAct Agent Utility Functions
Extracted utility methods for execution optimization and metadata management
"""

from typing import Dict, Any, List
from backend.models.schemas import StepResult
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


def should_auto_finish(observation: str, step_num: int) -> bool:
    """
    PERFORMANCE OPTIMIZATION: Detect if observation contains complete answer

    Early exit optimization - automatically triggers finish if observation
    appears to contain a complete answer, saving unnecessary iterations.

    Args:
        observation: Latest observation from tool execution
        step_num: Current step number

    Returns:
        True if should auto-finish, False to continue iterations
    """
    # Need at least 2 steps before considering early exit
    if step_num < 2:
        return False

    # Check minimum length
    if len(observation) < 200:
        return False

    # Skip if observation contains errors
    if "error" in observation.lower() or "failed" in observation.lower():
        # But allow if also has success indicators
        if "success" not in observation.lower():
            return False

    # Check for answer indicators
    answer_phrases = [
        "the answer",
        "result is",
        "therefore",
        "in conclusion",
        "based on",
        "to summarize",
        "in summary",
        "concluded",
        "final result",
        "outcome is"
    ]

    observation_lower = observation.lower()
    has_answer_phrase = any(phrase in observation_lower for phrase in answer_phrases)

    # Check for substantive content (numbers, facts, concrete results)
    has_numbers = any(char.isdigit() for char in observation)
    is_substantial = len(observation) > 300

    # Auto-finish if observation looks like a complete answer
    should_finish = has_answer_phrase or (has_numbers and is_substantial)

    if should_finish:
        logger.info("")
        logger.info("⚡ EARLY EXIT: Observation contains complete answer")
        logger.info("")

    return should_finish


def is_final_answer_unnecessary(step_results: List[StepResult], user_query: str) -> bool:
    """
    PERFORMANCE OPTIMIZATION: Check if final answer generation can be skipped

    Skips final LLM call if the last step already contains a comprehensive answer.

    Args:
        step_results: Results from all executed steps
        user_query: Original user query

    Returns:
        True if final answer generation can be skipped, False otherwise
    """
    if not step_results:
        return False

    last_step = step_results[-1]

    # Only skip if last step was successful
    if not last_step.success:
        return False

    observation = last_step.observation

    # Check if observation is substantial
    if len(observation) < 150:
        return False

    # Check if observation appears to be a complete answer (not just raw data)
    observation_lower = observation.lower()

    # If it's just code output or raw data, we need synthesis
    raw_data_indicators = [
        "dtype:",
        "columns:",
        "shape:",
        "<class",
        "array(",
        "dataframe",
    ]
    if any(indicator in observation_lower for indicator in raw_data_indicators):
        return False

    # If observation contains conclusion/summary phrases, it's likely complete
    complete_indicators = [
        "the answer",
        "in conclusion",
        "to summarize",
        "based on the",
        "the result",
        "therefore",
        "this shows",
        "analysis reveals"
    ]
    has_conclusion = any(phrase in observation_lower for phrase in complete_indicators)

    # If only one step executed and it has a conclusive answer, we can skip
    if len(step_results) == 1 and has_conclusion:
        return True

    # For multi-step executions, only skip if last step is explicitly marked as final/summary
    last_goal_lower = last_step.goal.lower()
    is_final_step = any(word in last_goal_lower for word in ["final", "answer", "summary", "synthesize", "combine"])

    # Skip if it's the final step and has substantial output
    should_skip = is_final_step and len(observation) > 200

    if should_skip:
        logger.info(f"Final answer unnecessary: Last step '{last_step.goal}' contains complete answer ({len(observation)} chars)")

    return should_skip


def build_execution_metadata(steps: List[Any], max_iterations: int) -> Dict[str, Any]:
    """
    Build metadata dictionary with execution details

    Args:
        steps: List of ReActStep objects from execution
        max_iterations: Maximum iterations configured

    Returns:
        Dictionary with execution metadata
    """
    from backend.tasks.react.models import ToolName

    # Collect unique tools used
    tools_used = list(set([
        step.action for step in steps
        if step.action != ToolName.FINISH
    ]))

    # Build execution steps
    execution_steps = [step.to_dict() for step in steps]

    return {
        "agent_type": "react",
        "total_iterations": len(steps),
        "max_iterations": max_iterations,
        "tools_used": tools_used,
        "execution_steps": execution_steps,
        "execution_trace": get_execution_trace(steps)
    }


def get_execution_trace(steps: List[Any]) -> str:
    """
    Get full trace of ReAct execution for debugging

    Args:
        steps: List of ReActStep objects from execution

    Returns:
        Formatted trace string showing all execution steps
    """
    if not steps:
        return "No steps executed."

    trace = ["=== ReAct Execution Trace ===\n"]
    for step in steps:
        trace.append(str(step))

    return "\n".join(trace)
