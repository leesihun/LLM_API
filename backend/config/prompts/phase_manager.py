"""
Phase Manager Prompts
Prompts for multi-phase workflow management with context handoffs.

These prompts enable conversation memory reuse across phases, reducing redundant file parsing.
"""

from typing import List, Optional


def get_initial_phase_prompt(
    phase_name: str,
    task: str,
    expected_outputs: Optional[List[str]] = None,
    files_as_fallback: bool = True
) -> str:
    """
    Create prompt for the first phase in a multi-phase workflow.

    Args:
        phase_name: Name of the phase (e.g., "Data Analysis")
        task: Detailed task description
        expected_outputs: List of expected outputs from this phase
        files_as_fallback: Whether to include files-as-fallback instruction

    Returns:
        Formatted prompt string for initial phase

    Example:
        >>> prompt = get_initial_phase_prompt(
        ...     phase_name="Data Analysis",
        ...     task="Analyze sales_data.csv and identify outliers",
        ...     expected_outputs=["Key statistics", "Outlier count"]
        ... )
    """
    prompt_parts = [
        f"**PHASE 1: {phase_name.upper()}**\n",
        f"TASK: {task}\n"
    ]

    if expected_outputs:
        prompt_parts.append("\n**Required Outputs:**")
        for i, output in enumerate(expected_outputs, 1):
            prompt_parts.append(f"{i}. {output}")
        prompt_parts.append("")

    if files_as_fallback:
        prompt_parts.append(
            "\n**Note:** If files are attached, use them for your analysis. "
            "Store your findings in conversation memory for use in subsequent phases."
        )

    return "\n".join(prompt_parts)


def get_handoff_phase_prompt(
    phase_name: str,
    task: str,
    previous_phases_summary: str,
    expected_outputs: Optional[List[str]] = None,
    files_as_fallback: bool = True,
    phase_num: int = 2
) -> str:
    """
    Create prompt for a subsequent phase with context handoff from previous phases.

    This prompt instructs the agent to prioritize conversation memory over re-processing files,
    enabling the "files as fallback" pattern for multi-phase workflows.

    Args:
        phase_name: Name of the current phase
        task: Detailed task description
        previous_phases_summary: Summary of findings from previous phases
        expected_outputs: List of expected outputs
        files_as_fallback: Whether to include files-as-fallback instruction
        phase_num: Phase number (for labeling)

    Returns:
        Formatted prompt with context from previous phases

    Example:
        >>> prompt = get_handoff_phase_prompt(
        ...     phase_name="Visualization",
        ...     task="Create charts based on Phase 1 analysis",
        ...     previous_phases_summary="Phase 1: Analyzed data, found 10 outliers...",
        ...     phase_num=2
        ... )
    """
    prompt_parts = [
        f"**PHASE {phase_num}: {phase_name.upper()}**\n",
        f"**PRIORITY: Use your previous phase findings first.**\n"
    ]

    # Summarize previous phases
    prompt_parts.append("**Previous Work Completed:**")
    prompt_parts.append(previous_phases_summary)

    prompt_parts.append(f"\n**Current Task:**\n{task}\n")

    if expected_outputs:
        prompt_parts.append("**Required Outputs:**")
        for i, output in enumerate(expected_outputs, 1):
            prompt_parts.append(f"{i}. {output}")
        prompt_parts.append("")

    if files_as_fallback:
        prompt_parts.append(
            "\n**IMPORTANT:** The attached files are ONLY for reference if you need to verify specific values. "
            "Your primary data source should be what you already calculated in previous phases. "
            "DO NOT re-analyze the raw data files from scratch."
        )

    return "\n".join(prompt_parts)


def get_phase_summary_prompt(
    phase_name: str,
    execution_result: str,
    max_summary_length: int = 200
) -> str:
    """
    Generate a concise summary of a completed phase for use in subsequent handoffs.

    Args:
        phase_name: Name of the completed phase
        execution_result: Full result/output from the phase execution
        max_summary_length: Maximum length for the summary

    Returns:
        Formatted prompt asking LLM to summarize the phase result

    Example:
        >>> prompt = get_phase_summary_prompt(
        ...     phase_name="Data Analysis",
        ...     execution_result="Analyzed 1000 rows, found mean=50.5...",
        ...     max_summary_length=200
        ... )
    """
    return f"""Summarize the key findings from the "{phase_name}" phase in {max_summary_length} characters or less.

Phase Output:
{execution_result[:1000]}

Focus on:
1. Key metrics or findings
2. Important patterns or outliers
3. Critical data for subsequent phases

Provide a concise summary:"""


def get_workflow_completion_prompt(
    original_query: str,
    all_phases_summary: str
) -> str:
    """
    Generate final answer prompt after all phases are complete.

    Args:
        original_query: The user's original query
        all_phases_summary: Summary of all completed phases

    Returns:
        Formatted prompt for generating comprehensive final answer

    Example:
        >>> prompt = get_workflow_completion_prompt(
        ...     original_query="Analyze data and create visualizations",
        ...     all_phases_summary="Phase 1: Analysis...\nPhase 2: Visualization..."
        ... )
    """
    return f"""Generate a comprehensive final answer based on the multi-phase workflow execution.

Original User Query: {original_query}

All Phases Summary:
{all_phases_summary}

Your final answer should:
1. Directly address the user's original query
2. Synthesize findings from all phases
3. Highlight key results and deliverables
4. Mention any artifacts generated (charts, files, etc.)

Provide a clear, complete final answer:"""
