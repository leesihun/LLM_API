"""
Base Code Generation Prompts
Provides concise, focused prompts for Python code generation.
"""

from typing import Optional


def get_base_generation_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Minimal base generation prompt - essential guidance only.

    Args:
        query: User's task/question
        context: Optional additional context

    Returns:
        Base generation prompt
    """
    prompt = f"""{"="*80}
MY ORIGINAL INPUT PROMPT
{"="*80}

{query}
"""

    if context:
        prompt += f"\n[ADDITIONAL CONTEXT] {context}\n"

    prompt += f"""
{"="*80}
"""

    return prompt


def get_task_guidance(query: str) -> str:
    """
    Provide task-specific guidance based on query type.

    Args:
        query: User's query

    Returns:
        Task-specific guidance section
    """
    query_lower = query.lower()
    is_aggregation = any(word in query_lower for word in ['sum', 'total', 'count', 'average', 'mean', 'median', 'calculate'])
    is_visualization = any(word in query_lower for word in ['plot', 'graph', 'chart', 'visualize', 'draw'])
    is_analysis = any(word in query_lower for word in ['analyze', 'report', 'summary', 'summarize', 'insights'])

    lines = [
        "="*80,
        "FINAL TASK FOR LLM AT THIS STAGE".center(80),
        "="*80,
        ""
    ]

    if is_visualization:
        lines.extend([
            "[TASK TYPE] Visualization/Plotting",
            "",
            "Workflow:",
            "  1. Import matplotlib: import matplotlib.pyplot as plt",
            "  2. Load file data (use access patterns from METADATA below)",
            "  3. Extract x, y values",
            "  4. Create plot",
            "  5. Save: plt.savefig('output.png')",
            ""
        ])
    elif is_aggregation:
        lines.extend([
            "[TASK TYPE] Calculation/Aggregation",
            "",
            "Workflow:",
            "  1. Load file data (use access patterns from METADATA below)",
            "  2. Extract relevant field",
            "  3. Calculate result",
            "  4. Print result with label",
            ""
        ])
    elif is_analysis:
        lines.extend([
            "[TASK TYPE] Analysis/Reporting",
            "",
            "Workflow:",
            "  1. Load file data once",
            "  2. Calculate multiple metrics",
            "  3. Print results clearly",
            ""
        ])
    else:
        lines.extend([
            "[TASK TYPE] General",
            "",
            "Generate Python code to complete the task above.",
            ""
        ])

    lines.extend([
        "="*80,
        ""
    ])

    return "\n".join(lines)


def get_prestep_generation_prompt(
    query: str,
    file_context: str,
    has_json_files: bool = False
) -> str:
    """
    Fast pre-analysis mode prompt - direct and focused.

    Args:
        query: User's task
        file_context: File information
        has_json_files: Whether JSON files are present

    Returns:
        Pre-step generation prompt
    """
    prompt_parts = [
        "You are a Python code generator in FAST PRE-ANALYSIS MODE.",
        "Your goal is to quickly analyze the attached files and provide an immediate answer to the user's question.",
        "",
        "IMPORTANT: Do NOT use Unicode emojis in your response. Use ASCII-safe markers like [OK], [X], [WARNING], [!!!] instead.",
        "",
        f"Task: {query}",
        "",
        file_context,
        "",
        "PRE-STEP MODE INSTRUCTIONS:",
        "- This is the FIRST attempt to answer the question using ONLY the provided files",
        "- Generate DIRECT, FOCUSED code that answers the specific question",
        "- Prioritize SPEED and CLARITY over comprehensive analysis"
    ]

    if file_context:
        prompt_parts.extend([
            "[!!!] CRITICAL: Use the EXACT filenames shown in the file list above",
            "[!!!] DO NOT use generic names like 'file.json', 'data.csv', 'input.json', etc.",
            "[!!!] COPY the actual filename from the list - character by character",
            "- NEVER makeup data, ALWAYS use the real files provided"
        ])

    prompt_parts.extend([
        "",
        "[!!!] EXECUTION ENVIRONMENT (CRITICAL - READ CAREFULLY):",
        "- Code will be executed via subprocess WITHOUT command-line arguments",
        "- DO NOT use sys.argv - it will be empty (only script name)",
        "- DO NOT use input() - this is non-interactive execution",
        "- ALL filenames MUST be HARDCODED directly in the code",
        "- Files are in the current working directory - use filenames directly",
        "- If you create functions, call them with HARDCODED filenames in main code",
        "",
        "[X] FORBIDDEN PATTERNS:",
        "  if __name__ == '__main__':",
        "      import sys",
        "      if len(sys.argv) > 1:",
        "          main(sys.argv[1])  # [X] WRONG - no arguments available!",
        "",
        "[OK] CORRECT PATTERN:",
        "  if __name__ == '__main__':",
        "      main('complex_json.json')  # [OK] CORRECT - hardcoded filename",
        "",
        "- Output results using print() statements with clear labels",
        "- Include basic error handling (try/except)",
        "- Focus on the MOST RELEVANT data columns/fields for the question",
        "",
        "CODE STYLE:",
        "- Keep it simple and direct",
        "- Use pandas/numpy for data files",
        "- Print intermediate steps for transparency",
        "- Always use real data from files, NO fake data, NO placeholders"
    ])

    if has_json_files:
        prompt_parts.extend([
            "",
            "JSON FILE HANDLING (CRITICAL - READ CAREFULLY):",
            "1. ALWAYS use: with open('EXACT_FILENAME_FROM_LIST.json', 'r', encoding='utf-8') as f: data = json.load(f)",
            "   [!!!] Replace 'EXACT_FILENAME_FROM_LIST.json' with the ACTUAL filename from the file list above!",
            "2. Wrap in try/except json.JSONDecodeError for error handling",
            "3. Check structure type FIRST: isinstance(data, dict) or isinstance(data, list)",
            "4. Use .get() method for dict access: data.get('key', default) NEVER data['key']",
            "5. ONLY use keys from \"Access Patterns\" section - DO NOT make up or guess keys",
            "6. For nested access, validate each level: data.get('parent', {}).get('child', default)",
            "7. For arrays, check length first: if len(data) > 0: item = data[0]",
            "8. COPY the \"Access Patterns\" shown above - they are structure-validated",
            "9. Handle None/null values: if value is not None: process(value)",
            "10. Add debug prints: print(\"Data type:\", type(data), \"Keys:\", list(data.keys()) if isinstance(data, dict) else 'N/A')"
        ])

    prompt_parts.append("\nGenerate ONLY the Python code, no explanations or markdown:")
    return "\n".join(prompt_parts)


def get_checklists_section() -> str:
    """
    Get checklist section for code validation.

    Returns:
        Formatted checklist section
    """
    return f"""{"="*80}
CHECKLISTS
{"="*80}

[1] Task Completion
    ? Does code answer the original prompt?
    ? Does code produce the expected output?

[2] Filename Validation
    ? Are ALL filenames from META DATA section (exact match)?
    ? NO generic names (data.json, file.csv, input.xlsx)?
    ? NO sys.argv, input(), or argparse?

[3] Safety & Error Handling
    ? try/except for file operations?
    ? .get() for dict access (JSON)?
    ? Type checks with isinstance()?

[4] Access Patterns
    ? Using access patterns from META DATA section?
    ? NOT guessing keys or field names?

{"="*80}

Generate ONLY executable Python code (no markdown, no explanations):
"""
