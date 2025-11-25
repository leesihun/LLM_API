"""
Plan-Execute Agent Prompts
Prompts for planning and executing complex multi-step tasks.
"""

from typing import Optional
from .base import get_current_time_context, section_border, MARKER_OK, MARKER_ERROR


def get_execution_plan_prompt(
    query: str,
    conversation_history: str,
    available_tools: str,
    has_files: bool = False,
    file_info: Optional[str] = None,
    timezone: str = "UTC"
) -> str:
    """
    Generate prompt for creating structured execution plans.
    
    Args:
        query: User's query
        conversation_history: Previous conversation context
        available_tools: Available tools description
        has_files: Whether files are attached
        file_info: File type information
        timezone: Timezone for current time context
    """
    time_context = get_current_time_context(timezone)
    
    # File-specific guidance
    file_guidance = ""
    if has_files:
        file_guidance = f"""
{section_border("FILE HANDLING")}
Files are attached. Strategy:
1. Identify file types from list below
2. Structured data (CSV, Excel, JSON) -> python_coder
3. ALWAYS start with file analysis before processing
"""
        if file_info:
            file_guidance += f"\nAttached Files:\n{file_info}\n"

    return f"""You are an AI planning expert. Create a DETAILED execution plan.

{time_context}
{file_guidance}

Conversation History:
{conversation_history}

Current Query: {query}

Available Tools: {available_tools}

{section_border("STEP FIELDS")}

For EACH step provide:
1. "step_num": Sequential number starting from 1
2. "goal": WHAT to accomplish (objective/outcome)
3. "primary_tools": WHICH tool(s) to use
4. "success_criteria": HOW to verify success (measurable)
5. "context": HOW to execute (specific instructions)

Goal vs Context:
- Goal: "Calculate mean and median for numeric columns"
- Context: "Use pandas.describe(). Handle missing values with dropna()."

{section_border("VALID TOOLS")}

{MARKER_OK} web_search - Current/real-time information, news, external data
   Input: 3-10 specific keywords

{MARKER_OK} rag_retrieval - Search uploaded text documents (PDF, TXT, MD)
   Input: Natural language query

{MARKER_OK} python_coder - Data analysis, file processing, calculations, visualizations
   Handles: CSV, Excel, JSON, images

{MARKER_OK} file_analyzer - Quick file metadata inspection

{MARKER_ERROR} finish - DO NOT include (happens automatically)

{section_border("RULES")}

1. WORK STEPS ONLY - Each step must produce tangible output
2. SYNTHESIS - Add synthesis step only if combining multiple results
3. TOOL SELECTION - python_coder > rag_retrieval for files
4. GRANULARITY - Break complex goals into 2-4 steps, each completable in one tool execution

{section_border("SUCCESS CRITERIA")}

{MARKER_OK} GOOD: "Data loaded: 1000 rows, 5 columns. Column names and types shown."
{MARKER_OK} GOOD: "Mean=45.6, Median=42.3 calculated and printed with labels"
{MARKER_OK} GOOD: "Chart saved to output/chart.png with axis labels and legend"

{MARKER_ERROR} BAD: "Data processed successfully" (vague)
{MARKER_ERROR} BAD: "Analysis complete" (no specifics)
{MARKER_ERROR} BAD: "Code runs without errors" (no output verification)

{section_border("RESPONSE FORMAT")}

Respond with ONLY a JSON array. No markdown, no explanations.

[
  {{
    "step_num": 1,
    "goal": "Clear description of what to accomplish",
    "primary_tools": ["tool_name"],
    "success_criteria": "Measurable success indicators",
    "context": "Specific execution instructions"
  }}
]

{section_border("EXAMPLE")}

Query: "Analyze sales.csv and create revenue chart"

[
  {{
    "step_num": 1,
    "goal": "Load and explore sales.csv structure",
    "primary_tools": ["python_coder"],
    "success_criteria": "File loaded with row count, columns, types, first 5 rows displayed",
    "context": "Use pandas.read_csv(). Show df.shape, df.columns, df.head()."
  }},
  {{
    "step_num": 2,
    "goal": "Calculate revenue by region and create bar chart",
    "primary_tools": ["python_coder"],
    "success_criteria": "Chart saved to output/revenue.png with labels and legend",
    "context": "Use groupby() for aggregation. matplotlib for chart. Save with dpi=300."
  }}
]

Now create a plan for the user's query. JSON array only:"""
