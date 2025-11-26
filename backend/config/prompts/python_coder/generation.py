"""
Base Code Generation Prompts
Provides concise, focused prompts for Python code generation.
"""

from typing import Optional
from ..base import (
    section_border, MARKER_OK, MARKER_ERROR, MARKER_CRITICAL,
    get_current_time_context
)


def get_base_generation_prompt(query: str, context: Optional[str] = None, timezone: str = "UTC") -> str:
    """
    Minimal base generation prompt with time context.
    """
    time_context = get_current_time_context(timezone)
    
    prompt = f"""{section_border("TASK")}

{time_context}

{query}
"""
    if context:
        prompt += f"\n[Additional Context] {context}\n"
    
    return prompt


def get_task_guidance(query: str) -> str:
    """Provide task-specific workflow guidance."""
    return f"""{section_border("WORKFLOW GUIDANCE")}

[Visualization] Import matplotlib -> Load data -> Extract values -> Create plot -> Save: plt.savefig('output.png')

[Calculation] Load data -> Extract values -> Calculate -> Save to result.txt -> Print confirmation

[Analysis] Load data -> Calculate metrics -> Save to result.csv/result.txt -> Print confirmation

[General] Generate Python code to complete the task above.

{section_border("OUTPUT REQUIREMENTS")}

{MARKER_CRITICAL} ALWAYS save final results to file (DO NOT just print large data):
- Tabular data (DataFrames, tables): df.to_csv('result.csv', index=False)
- Text results/summaries: write to 'result.txt'
- After saving, print a brief confirmation: print("Results saved to result.csv")

{MARKER_OK} CORRECT:
  df.to_csv('result.csv', index=False)
  print(f"Saved {{len(df)}} rows to result.csv")

{MARKER_ERROR} WRONG:
  print(df)  # Large DataFrame will be truncated!
  print(df.to_string())  # Too long for console!
"""


def get_prestep_generation_prompt(
    query: str,
    file_context: str,
    has_json_files: bool = False,
    timezone: str = "UTC"
) -> str:
    """Fast pre-analysis mode prompt - direct and focused."""
    time_context = get_current_time_context(timezone)
    
    prompt_parts = [
        "You are a Python code generator in FAST PRE-ANALYSIS MODE.",
        "",
        time_context,
        "",
        f"Task: {query}",
        "",
        file_context,
        "",
        "PRE-STEP INSTRUCTIONS:",
        "- FIRST attempt using ONLY provided files",
        "- Generate DIRECT, FOCUSED code",
        "- Prioritize SPEED and CLARITY",
        "",
        f"{MARKER_CRITICAL} CRITICAL RULES:",
        "- Use EXACT filenames from file list above",
        f"- {MARKER_ERROR} NO generic names (file.json, data.csv)",
        f"- {MARKER_ERROR} NO sys.argv, input(), argparse",
        "- ALL filenames HARDCODED in code",
        "- Output results with print() and clear labels",
        "",
        f"{MARKER_OK} CORRECT: main('actual_filename.json')",
        f"{MARKER_ERROR} WRONG: main(sys.argv[1])"
    ]
    
    if has_json_files:
        prompt_parts.extend([
            "",
            "JSON HANDLING:",
            "- with open('EXACT_NAME.json', 'r', encoding='utf-8') as f: data = json.load(f)",
            "- Check type: isinstance(data, dict) or isinstance(data, list)",
            "- Safe access: data.get('key', default)",
            "- ONLY use keys from Access Patterns section"
        ])
    
    prompt_parts.extend([
        "",
        f"{MARKER_CRITICAL} OUTPUT REQUIREMENT:",
        "- Save results to file: df.to_csv('result.csv', index=False) or write to 'result.txt'",
        "- Print ONLY a brief confirmation, NOT the full data",
        f"- {MARKER_ERROR} DO NOT print(df) - large data will be truncated!"
    ])
    
    prompt_parts.append("\nGenerate ONLY Python code, no markdown:")
    return "\n".join(prompt_parts)


def get_checklists_section() -> str:
    """Code validation checklist."""
    return f"""{section_border("CHECKLIST")}

[1] Task: Does code answer the prompt? Expected output produced?
[2] Filenames: Exact match from META DATA? No generic names? No sys.argv?
[3] Safety: try/except for files? .get() for dicts? isinstance() checks?
[4] Patterns: Using access patterns from META DATA?

Generate ONLY executable Python code (no markdown):
"""
