"""
Base Prompt Utilities
Provides standardized structure, markers, and temporal context for all prompts.
"""

from datetime import datetime
from typing import Optional
import pytz


# ============================================================================
# STANDARD ASCII MARKERS
# ============================================================================
# Use these consistently across all prompts instead of Unicode symbols

MARKER_OK = "[OK]"
MARKER_ERROR = "[X]"
MARKER_CRITICAL = "[!!!]"
MARKER_WARNING = "[WARNING]"
MARKER_CHECK = "[CHECK]"
MARKER_RULE = "[RULE]"
MARKER_TIP = "[TIP]"


# ============================================================================
# STANDARD SEPARATORS
# ============================================================================

def section_border(title: str = "", width: int = 80) -> str:
    """Create a standard section border with optional centered title."""
    border = "=" * width
    if title:
        return f"{border}\n{title.center(width)}\n{border}"
    return border


def subsection_border(width: int = 80) -> str:
    """Create a subsection separator."""
    return "-" * width


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
# COMMON RULE BLOCKS (Reusable across prompts)
# ============================================================================

FILENAME_RULES = f"""
{MARKER_RULE} EXACT FILENAMES
   - Copy EXACT filename from META DATA section
   - {MARKER_ERROR} NO generic names: 'data.json', 'file.json', 'input.csv'
   - {MARKER_OK} Example: filename = 'sales_report_Q4_2024.json'
"""

NO_ARGS_RULES = f"""
{MARKER_RULE} NO COMMAND-LINE ARGS / USER INPUT
   - Code runs via subprocess WITHOUT arguments
   - {MARKER_ERROR} NO sys.argv, NO input(), NO argparse
   - {MARKER_OK} All filenames must be HARDCODED
"""

JSON_SAFETY_RULES = f"""
{MARKER_RULE} JSON SAFETY
   - Use .get() for dict access: data.get('key', default)
   - Check type: isinstance(data, dict) or isinstance(data, list)
   - Add error handling: try/except json.JSONDecodeError
   - Validate nested access: data.get('parent', {{}}).get('child', default)
"""

ACCESS_PATTERN_RULES = f"""
{MARKER_RULE} USE ACCESS PATTERNS
   - Copy access patterns from META DATA section
   - {MARKER_ERROR} DON'T guess keys or field names
   - {MARKER_OK} Use .get() for safe dict access
"""


__all__ = [
    # Markers
    'MARKER_OK',
    'MARKER_ERROR', 
    'MARKER_CRITICAL',
    'MARKER_WARNING',
    'MARKER_CHECK',
    'MARKER_RULE',
    'MARKER_TIP',
    
    # Separators
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
    
    # Reusable rule blocks
    'FILENAME_RULES',
    'NO_ARGS_RULES',
    'JSON_SAFETY_RULES',
    'ACCESS_PATTERN_RULES',
]

