"""
Plan-Execute Agent Prompts
Prompts for planning and executing complex multi-step tasks.
"""

from typing import Optional


def get_execution_plan_prompt(
    query: str,
    conversation_history: str,
    available_tools: str,
    has_files: bool = False
) -> str:
    """
    Prompt for creating a structured execution plan for complex tasks.
    Used in: backend/tasks/Plan_execute.py (_create_execution_plan)

    Args:
        query: User's current query
        conversation_history: Previous conversation context
        available_tools: List of available tools
        has_files: Whether files are attached to the query
    """
    file_first_note = (
        "\nIMPORTANT: There are attached files for this task. "
        "Perform local file analysis FIRST. "
    ) if has_files else ""

    return f"""You are an AI planning expert. Analyze this user query and create a DETAILED, focused execution plan.

{file_first_note}

Conversation History:
{conversation_history}

Current User Query: {query}

Available Tools:
{available_tools}

Create a MINIMAL step-by-step execution plan with ONLY the work steps needed to answer the query. For EACH step, provide:
1. Goal: What this step aims to accomplish (be specific and detailed)
2. Primary Tools: Main tools to use
3. Success Criteria: How to verify the step succeeded (be concrete and measurable)
4. Context: Additional instructions or notes for executing this step

CRITICAL: You MUST respond with a JSON array of steps. Each step must have this exact structure:
{{
  "step_num": 1,
  "goal": "Clear, detailed description of what to accomplish",
  "primary_tools": ["tool_name"],
  "success_criteria": "How to verify success",
  "context": "Additional context or specific instructions for this step"
}}

Valid tool names ONLY: web_search, rag_retrieval, python_coder

IMPORTANT RULES:
- Only include ACTUAL WORK steps (file analysis, data processing, searches, etc.)
- Do NOT include a separate "synthesize results" or "generate final answer" step - this happens automatically
- Each step should produce concrete output/data
- The final work step should naturally lead to a complete answer

SUCCESS CRITERIA EXAMPLES:

GOOD Success Criteria (Specific, Measurable):
✓ "Successfully loaded data with column names, types, and shape displayed"
✓ "Mean and median calculated for all numeric columns with values printed"
✓ "Search returned at least 3 relevant articles from 2025"
✓ "Chart saved to temp_charts/sales_trend.png with proper labels"
✓ "Outliers identified and count displayed (expect 5-10 outliers)"

BAD Success Criteria (Vague, Unmeasurable):
✗ "Data processed successfully" (what does "processed" mean?)
✗ "Analysis complete" (what analysis? what outputs?)
✗ "Information retrieved" (how much? what quality?)
✗ "Code runs without errors" (but what should it output?)
✗ "Step finished" (not helpful for verification)

Example response format:
[
  {{
    "step_num": 1,
    "goal": "Load and explore the uploaded JSON file structure, extract column names, data types, and preview first few rows",
    "primary_tools": ["python_coder"],
    "success_criteria": "Successfully loaded data with structure information (columns, types, shape) and preview displayed",
    "context": "Use json.load() to read JSON. Show df.head(), df.info(), df.describe(). Handle nested JSON structures if present."
  }},
  {{
    "step_num": 2,
    "goal": "Calculate mean and median of all numeric columns. If user-requested column names don't match, find semantically similar columns.",
    "primary_tools": ["python_coder"],
    "success_criteria": "Mean and median values calculated and displayed for all numeric columns with clear labels",
    "context": "Use numpy statistical functions"
  }}
]

Now create a structured plan for the user's query. Respond with ONLY the JSON array, no additional text:"""
