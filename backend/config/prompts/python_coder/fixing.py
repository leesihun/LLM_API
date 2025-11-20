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


def get_smart_fix_prompt(
    query: str,
    context: Optional[str],
    code: str,
    error_message: str,
    previous_attempts: List[dict]
) -> str:
    """
    Smart fix with context from previous attempts.

    Args:
        query: Original user query
        context: Optional additional context
        code: Current code that failed
        error_message: Current error message
        previous_attempts: List of previous attempts with code and errors

    Returns:
        Smart fix prompt with historical context
    """
    context_line = f"Context: {context}" if context else ""

    prompt_parts = [
        "Fix the following Python code that failed during execution.",
        "You have context from previous failed attempts - learn from them!",
        "",
        "IMPORTANT: Do NOT use Unicode emojis in your response. Use ASCII-safe markers like [OK], [X], [WARNING], [!!!] instead.",
        "",
        f"Original request: {query}",
        context_line,
        ""
    ]

    # Show previous attempts
    if previous_attempts:
        prompt_parts.append("=== PREVIOUS FAILED ATTEMPTS ===")
        for idx, attempt in enumerate(previous_attempts, 1):
            prompt_parts.append(f"\n[Attempt {idx}]")
            prompt_parts.append("Code:")
            prompt_parts.append("```python")
            # Show first 20 lines
            code_lines = attempt.get('code', '').split('\n')
            for line in code_lines[:20]:
                prompt_parts.append(line)
            if len(code_lines) > 20:
                prompt_parts.append(f"... [{len(code_lines) - 20} more lines]")
            prompt_parts.append("```")

            prompt_parts.append(f"\nError: {attempt.get('error', 'Unknown error')}")

        prompt_parts.append("\n=== CURRENT ATTEMPT ===\n")

    prompt_parts.extend([
        "Current code:",
        "```python",
        code,
        "```",
        "",
        "Current error:",
        error_message,
        "",
        "[SMART FIXING STRATEGY]:",
        "1. Review previous attempts - what patterns of errors occurred?",
        "2. Identify if you're making the same mistake repeatedly",
        "3. Try a DIFFERENT approach if previous fixes didn't work",
        "4. Consider alternative methods/libraries if current approach keeps failing",
        "5. Simplify the solution if complexity is causing issues",
        "",
        "[COMMON PATTERNS TO AVOID]:",
        "- Repeating the same fix that already failed",
        "- Adding more complexity instead of simplifying",
        "- Ignoring error messages - they tell you what's wrong!",
        "",
        "Analyze all attempts and generate a DIFFERENT, WORKING solution. Output ONLY the corrected code:"
    ])

    return "\n".join(prompt_parts)
