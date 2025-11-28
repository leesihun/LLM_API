"""
Base Prompt Utilities
Provides standardized structure, markers, and temporal context for all prompts.

Version: 2.0.0 - Modernized for Anthropic/Claude Code style
Changes: Added new markdown-based utilities, deprecated ASCII markers
"""

from datetime import datetime
from typing import Optional, List
import pytz


# ============================================================================
# DEPRECATED ASCII MARKERS (Kept for backward compatibility)
# ============================================================================
# DEPRECATED: Use markdown emphasis instead
# Migration: [OK] → **Good:**, [X] → **Bad:**, [!!!] → **Critical:**

MARKER_OK = "**Good:**"  # Previously "[OK]"
MARKER_ERROR = "**Bad:**"  # Previously "[X]"
MARKER_CRITICAL = "**Critical:**"  # Previously "[!!!]"
MARKER_WARNING = "**Warning:**"  # Previously "[WARNING]"
MARKER_CHECK = "**Check:**"  # Previously "[CHECK]"
MARKER_RULE = ""  # DEPRECATED: Use numbered lists instead
MARKER_TIP = "**Tip:**"  # Previously "[TIP]"


# ============================================================================
# MODERN MARKDOWN UTILITIES
# ============================================================================

def section_header(title: str, level: int = 2) -> str:
    """
    Create a markdown section header.

    Args:
        title: Section title
        level: Markdown header level (1-6, default 2 for ##)

    Returns:
        Markdown header string

    Example:
        section_header("Task", 2) -> "## Task"
        section_header("Requirements", 3) -> "### Requirements"
    """
    prefix = "#" * min(max(level, 1), 6)
    return f"{prefix} {title}"


def role_definition(
    title: str,
    expertise: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """
    Create a specific role definition following Anthropic best practices.

    Args:
        title: Professional role (e.g., "Python code generation expert")
        expertise: Optional specialization (e.g., "specializing in data analysis")
        context: Optional additional context

    Returns:
        Complete role definition string

    Example:
        role_definition(
            "Python code generation expert",
            "specializing in data analysis",
            "You work with pandas, numpy, and matplotlib."
        )
        → "You are a Python code generation expert specializing in data analysis.
           You work with pandas, numpy, and matplotlib."
    """
    parts = [f"You are a {title}"]

    if expertise:
        parts[0] += f" {expertise}"

    parts[0] += "."

    if context:
        parts.append(context)

    return " ".join(parts)


def format_code_block(code: str, language: str = "python") -> str:
    """
    Format code with proper markdown fences.

    Args:
        code: Code content
        language: Syntax highlighting language

    Returns:
        Markdown code block
    """
    return f"```{language}\n{code}\n```"


# ============================================================================
# DEPRECATED SEPARATORS (Kept for backward compatibility)
# ============================================================================

def section_border(title: str = "", width: int = 80) -> str:
    """
    DEPRECATED: Use section_header() instead.

    Create a standard section border with optional centered title.
    Kept for backward compatibility during migration.
    """
    if title:
        return section_header(title, level=2)
    return ""


def subsection_border(width: int = 80) -> str:
    """
    DEPRECATED: Use section_header() with level=3 instead.

    Create a subsection separator.
    Kept for backward compatibility during migration.
    """
    return ""


# ============================================================================
# TEMPORAL CONTEXT
# ============================================================================

def get_current_time_context(timezone: str = "UTC") -> str:
    """
    Generate current time context string for prompts.
    
    Args:
        timezone: Timezone string (e.g., "UTC", "Asia/Seoul", "America/New_York")
    
    Returns:
        Formatted time context string
    
    Example output:
        Current Time: 2025-01-15 (Wednesday) 14:30:45 UTC
    """
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
    except Exception:
        now = datetime.utcnow()
        timezone = "UTC"
    
    date_str = now.strftime("%Y-%m-%d")
    day_of_week = now.strftime("%A")
    time_str = now.strftime("%H:%M:%S")
    
    return f"Current Time: {date_str} ({day_of_week}) {time_str} {timezone}"


def get_time_context_dict(timezone: str = "UTC") -> dict:
    """
    Get time context as a dictionary for flexible formatting.
    
    Args:
        timezone: Timezone string
    
    Returns:
        Dictionary with date, day_of_week, time, timezone, month, year
    """
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
    except Exception:
        now = datetime.utcnow()
        timezone = "UTC"
    
    return {
        "date": now.strftime("%Y-%m-%d"),
        "day_of_week": now.strftime("%A"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": timezone,
        "month": now.strftime("%B"),
        "year": now.strftime("%Y"),
        "full_datetime": now.strftime("%Y-%m-%d %H:%M:%S")
    }


# ============================================================================
# STANDARD PROMPT SECTIONS
# ============================================================================

def format_context_section(context: str) -> str:
    """Format a CONTEXT section with standard border."""
    return f"""{section_border("CONTEXT")}

{context}
"""


def format_task_section(task: str) -> str:
    """Format a TASK section with standard border."""
    return f"""{section_border("TASK")}

{task}
"""


def format_rules_section(rules: list) -> str:
    """
    Format a RULES section with numbered rules.
    
    Args:
        rules: List of rule strings
    
    Returns:
        Formatted rules section
    """
    rules_text = "\n".join(f"{MARKER_RULE} {i+1}. {rule}" for i, rule in enumerate(rules))
    return f"""{section_border("RULES")}

{rules_text}
"""


def format_format_section(format_instructions: str) -> str:
    """Format a FORMAT section with standard border."""
    return f"""{section_border("RESPONSE FORMAT")}

{format_instructions}
"""


def format_examples_section(examples: list) -> str:
    """
    Format an EXAMPLES section.
    
    Args:
        examples: List of example strings or dicts with 'input' and 'output'
    
    Returns:
        Formatted examples section
    """
    examples_text = []
    for i, example in enumerate(examples, 1):
        if isinstance(example, dict):
            examples_text.append(f"Example {i}:")
            examples_text.append(f"  Input: {example.get('input', '')}")
            examples_text.append(f"  Output: {example.get('output', '')}")
        else:
            examples_text.append(f"Example {i}: {example}")
        examples_text.append("")
    
    return f"""{section_border("EXAMPLES")}

{chr(10).join(examples_text)}
"""


# ============================================================================
# COMMON RULE BLOCKS (Modernized with markdown)
# ============================================================================

FILENAME_RULES = """
### Filename Requirements
- Use exact filenames from metadata
- Avoid generic names: `data.json`, `file.json`, `input.csv`
- Example: `filename = 'sales_report_Q4_2024.json'`
"""

NO_ARGS_RULES = """
### No Command-Line Arguments
- Code runs via subprocess without arguments
- No `sys.argv`, `input()`, or `argparse`
- All filenames must be hardcoded
"""

JSON_SAFETY_RULES = """
### JSON Safety
- Use `.get()` for dict access: `data.get('key', default)`
- Check type: `isinstance(data, dict)` or `isinstance(data, list)`
- Add error handling: `try/except json.JSONDecodeError`
- Validate nested access: `data.get('parent', {}).get('child', default)`
"""

ACCESS_PATTERN_RULES = """
### Access Patterns
- Copy access patterns from metadata section
- Don't guess keys or field names
- Use `.get()` for safe dict access
"""

OUTPUT_FILE_RULES = """
### Output Requirements
- Use `print()` to display results directly in stdout
- DataFrames will show ALL rows/columns (pandas display options pre-configured)
- ONLY save to files when necessary:
  - Images/plots: `plt.savefig('output.png')`
  - PowerPoint: `prs.save('output.pptx')`
  - Excel reports: `df.to_excel('report.xlsx')`
  - PDF documents: Save as PDF when explicitly requested
- Do not save ordinary results (CSV, TXT) to files - just print them
- `print(df)` will show complete data, no truncation
"""


__all__ = [
    # Deprecated markers (backward compatibility)
    'MARKER_OK',
    'MARKER_ERROR',
    'MARKER_CRITICAL',
    'MARKER_WARNING',
    'MARKER_CHECK',
    'MARKER_RULE',
    'MARKER_TIP',

    # Modern utilities (NEW)
    'section_header',
    'role_definition',
    'format_code_block',

    # Deprecated separators (backward compatibility)
    'section_border',
    'subsection_border',

    # Time context
    'get_current_time_context',
    'get_time_context_dict',

    # Section formatters
    'format_context_section',
    'format_task_section',
    'format_rules_section',
    'format_format_section',
    'format_examples_section',

    # Reusable rule blocks (modernized)
    'FILENAME_RULES',
    'NO_ARGS_RULES',
    'JSON_SAFETY_RULES',
    'ACCESS_PATTERN_RULES',
    'OUTPUT_FILE_RULES',
]

