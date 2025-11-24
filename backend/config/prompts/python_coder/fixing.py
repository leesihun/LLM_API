"""
Code Fixing Prompts
Provides prompts for fixing code issues and execution errors.
"""

from typing import Optional, List


def get_modification_prompt(
    query: str,
    context: Optional[str],
    code: str,
    issues: List[str]
) -> str:
    """
    Fix code based on verification issues.

    Args:
        query: Original user query
        context: Optional additional context
        code: Current code with issues
        issues: List of issues to fix

    Returns:
        Modification prompt
    """
    context_line = f"Context: {context}" if context else ""

    return f"""Fix the following Python code to address these issues:

IMPORTANT: Do NOT use Unicode emojis in your response. Use ASCII-safe markers like [OK], [X], [WARNING], [!!!] instead.

Original request: {query}
{context_line}

Current code:
```python
{code}
```

Issues to fix:
{chr(10).join(f"- {issue}" for issue in issues)}

[FIXING GUIDELINES]:
1. Address each issue listed above
2. Keep the original logic and approach
3. Maintain code readability
4. Ensure all filenames remain HARDCODED
5. Do NOT introduce new dependencies if possible

Generate the corrected Python code. Output ONLY the code, no explanations:"""


def get_execution_fix_prompt(
    query: str,
    context: Optional[str],
    code: str,
    error_message: str
) -> str:
    """
    Fix code that failed during execution.

    Args:
        query: Original user query
        context: Optional additional context
        code: Current code that failed
        error_message: Error from execution

    Returns:
        Execution fix prompt
    """
    context_line = f"Context: {context}" if context else ""

    return f"""Fix the following Python code that failed during execution:

IMPORTANT: Do NOT use Unicode emojis in your response. Use ASCII-safe markers like [OK], [X], [WARNING], [!!!] instead.

Original request: {query}
{context_line}

Current code:
```python
{code}
```

Execution error:
{error_message}

[FIXING GUIDELINES]:
1. Analyze the error carefully - what caused it?
2. Fix the root cause, not just symptoms
3. Common error types:
   - FileNotFoundError: Check filename, add error handling
   - KeyError: Use .get() for dict access
   - IndexError: Check list length before access
   - TypeError: Validate data types before operations
   - AttributeError: Check if object/attribute exists
4. Add appropriate error handling (try/except)
5. Keep the original approach if possible

Analyze the error and fix the code. Output ONLY the corrected code, no explanations:"""
