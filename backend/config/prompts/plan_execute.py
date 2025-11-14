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
        "Prefer local file analysis FIRST using python_coder or python_code. "
        "Only fall back to web_search if local analysis fails or is insufficient."
    ) if has_files else ""

    return f"""You are an AI planning expert. Analyze this user query and create a detailed, structured execution plan.

{file_first_note}

Conversation History:
{conversation_history}

Current User Query: {query}

Available Tools:
{available_tools}

Create a step-by-step execution plan. For EACH step, provide:
1. Goal: What this step aims to accomplish
2. Primary Tools: Main tools to try (in order of preference)
3. Fallback Tools: Alternative tools if primary fails
4. Success Criteria: How to know the step succeeded

CRITICAL: You MUST respond with a JSON array of steps. Each step must have this exact structure:
{{
  "step_num": 1,
  "goal": "Clear description of what to accomplish",
  "primary_tools": ["tool_name1", "tool_name2"],
  "fallback_tools": ["backup_tool1"],
  "success_criteria": "How to verify success",
  "context": "Additional context or notes"
}}

Valid tool names ONLY: web_search, rag_retrieval, python_code, python_coder

Example response format:
[
  {{
    "step_num": 1,
    "goal": "Load and analyze the uploaded JSON file, save its reading code to a file ",
    "primary_tools": ["python_coder"],
    "fallback_tools": ["python_code"],
    "success_criteria": "Data successfully loaded with basic statistics displayed",
    "context": "Use pandas to read JSON and show head, shape, describe how to access it"
  }},
  {{
    "step_num": 2,
    "goal": "Calculate mean and median of numeric columns, if the user asked file header doesn't match the given data, think and if there are any headers that are semantically similar, use the header that is most similar to the user's query",
    "primary_tools": ["python_code"],
    "fallback_tools": ["python_coder"],
    "success_criteria": "Mean and median values displayed for all numeric columns",
    "context": "Use numpy or pandas statistical functions"
  }},
  {{
    "step_num": 3,
    "goal": "Generate final summary answer",
    "primary_tools": ["finish"],
    "fallback_tools": [],
    "success_criteria": "Complete answer provided to user",
    "context": "Synthesize all results into coherent answer"
  }}
]

Now create a structured plan for the user's query. Respond with ONLY the JSON array, no additional text:"""
