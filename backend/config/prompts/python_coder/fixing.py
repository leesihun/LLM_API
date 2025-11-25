"""
Code Fixing Prompts
Prompts for fixing code issues and execution errors.
"""

from typing import Optional, List
from ..base import section_border, MARKER_OK, MARKER_ERROR


def get_modification_prompt(
    query: str,
    context: Optional[str],
    code: str,
    issues: List[str]
) -> str:
    """Fix code based on verification issues."""
    return f"""Fix the Python code to address these issues:

Original request: {query}
{f"Context: {context}" if context else ""}

Current code:
```python
{code}
```

Issues to fix:
{chr(10).join(f"- {issue}" for issue in issues)}

{section_border("FIXING RULES")}

1. Address each issue listed
2. Keep original logic and approach
3. Filenames must remain HARDCODED
4. {MARKER_ERROR} NO new dependencies if possible

Generate corrected Python code only:"""


def get_execution_fix_prompt(
    query: str,
    context: Optional[str],
    code: str,
    error_message: str
) -> str:
    """Fix code that failed during execution."""
    return f"""Fix the Python code that failed during execution:

Original request: {query}
{f"Context: {context}" if context else ""}

Current code:
```python
{code}
```

Execution error:
{error_message}

{section_border("FIXING GUIDELINES")}

1. Analyze error - what caused it?
2. Fix root cause, not symptoms
3. Common fixes:
   - FileNotFoundError: Check filename, add error handling
   - KeyError: Use .get() for dict access
   - IndexError: Check list length first
   - TypeError: Validate data types
4. Add try/except where needed
5. Keep original approach

Analyze and fix. Output ONLY corrected code:"""
