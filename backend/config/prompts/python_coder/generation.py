"""
Base Code Generation Prompts
Provides concise, focused prompts for Python code generation.

Version: 2.0.0 - Modernized for Anthropic/Claude Code style
Changes: Removed ASCII markers, markdown structure, cleaner formatting
"""

from typing import Optional
from ..base import get_current_time_context


def get_base_generation_prompt(query: str, context: Optional[str] = None, timezone: str = "UTC") -> str:
    """
    Minimal base generation prompt with time context.
    Clean, focused structure for straightforward tasks.

    Args:
        query: User's task/question
        context: Optional additional context
        timezone: Timezone for time context

    Returns:
        Base generation prompt
    """
    time_context = get_current_time_context(timezone)
    context_section = f"\n## Additional Context\n{context}\n" if context else ""

    return f"""## Task

{time_context}

{query}
{context_section}"""


def get_prestep_generation_prompt(
    query: str,
    file_context: str,
    has_json_files: bool = False,
    timezone: str = "UTC"
) -> str:
    """
    Fast pre-analysis mode for file-based queries.
    Optimized for speed and directness.

    Args:
        query: User's task/question
        file_context: File metadata and context
        has_json_files: Whether JSON files are present
        timezone: Timezone for time context

    Returns:
        Pre-step generation prompt
    """
    time_context = get_current_time_context(timezone)

    json_section = ""
    if has_json_files:
        json_section = """
### JSON Handling
- Load: `with open('EXACT_NAME.json', 'r', encoding='utf-8') as f: data = json.load(f)`
- Check type: `isinstance(data, dict)` or `isinstance(data, list)`
- Safe access: `data.get('key', default)`
- Use keys from Access Patterns section only
"""

    return f"""You are a Python code generation specialist operating in fast pre-analysis mode.

{time_context}

## Task
{query}

{file_context}

## Pre-Step Strategy
- First attempt using ONLY provided files
- Generate direct, focused code
- Prioritize speed and clarity

## Critical Requirements
- Use exact filenames from file list above
- Avoid generic names: `file.json`, `data.csv`
- No command-line arguments: no `sys.argv`, `input()`, `argparse`
- All filenames hardcoded in code
- Output results with `print()` and clear labels

**Example:**
- Good: `main('actual_filename.json')`
- Bad: `main(sys.argv[1])`
{json_section}
## Output Requirements
- Print results directly: `print(df)` or `print(result)`
- Pandas display pre-configured to show ALL rows/columns
- `print(df)` shows complete data, no truncation
- ONLY save files when necessary: images, PPTX, Excel reports

Generate ONLY Python code, no markdown:"""


def get_task_guidance(query: str) -> str:
    """Provide task-specific workflow guidance."""
    return """## Workflow Guidance

**Visualization:** Import matplotlib → Load data → Extract values → Create plot → Save: `plt.savefig('output.png')`

**Calculation:** Load data → Extract values → Calculate → Print results directly

**Analysis:** Load data → Calculate metrics → Print results directly

**General:** Generate Python code to complete the task above.

## Output Requirements

**Good:** Print results directly (pandas display options are pre-configured)
- DataFrames: `print(df)` will show ALL rows and columns (no truncation)
- Calculation results: `print(result)` for numeric/text results
- Summaries: `print(summary_text)` for analysis results

**Critical:** ONLY save to files when necessary
- Images/plots: `plt.savefig('output.png')`
- PowerPoint: `prs.save('output.pptx')`
- Excel reports: `df.to_excel('report.xlsx')` (when explicitly requested)
- PDF documents: Save as PDF when explicitly requested

**Good:**
```python
print(df)  # Shows complete DataFrame, no truncation
print(f"Total: {result}")
```

**Bad:**
```python
df.to_csv('result.csv', index=False)  # Don't save unless necessary
with open('result.txt', 'w') as f: f.write(text)  # Don't save text files
```
"""


def get_checklists_section() -> str:
    """Code validation checklist."""
    return """## Checklist

1. **Task:** Does code answer the prompt? Expected output produced?
2. **Filenames:** Exact match from metadata? No generic names? No sys.argv?
3. **Safety:** try/except for files? .get() for dicts? isinstance() checks?
4. **Patterns:** Using access patterns from metadata?

Generate ONLY executable Python code (no markdown):
"""
