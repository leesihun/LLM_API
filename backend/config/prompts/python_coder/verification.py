"""
Code Verification Prompts
Semantic verification focused on correctness, not style.

Version: 2.0.0 - Modernized for Anthropic/Claude Code style
Changes: Removed ASCII markers, markdown structure, thinking triggers
"""

from typing import Optional


def get_verification_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    code: str,
    has_json_files: bool = False
) -> str:
    """
    Semantic code verification focused on execution errors.
    Identifies blocking issues before execution.

    Args:
        query: User's question
        context: Optional additional context
        file_context: File metadata and context
        code: Code to verify
        has_json_files: Whether JSON files are present

    Returns:
        Verification prompt with JSON response format
    """
    context_line = f"\n**Context:** {context}\n" if context else ""

    json_check = """
### JSON Safety Checks
- Uses exact filename from list?
- Has `isinstance()` check?
- Uses `.get()` for dict access?
- Only uses keys from Access Patterns?
""" if has_json_files else ""

    return f"""You are a Python code verification specialist. Identify potential execution errors before running code.

## User Question
{query}
{context_line}
{file_context}

## Code to Verify
```python
{code}
```

## Verification Checklist

### Logic Correctness
- Does code address the question?
- Are calculations correct?

### Execution Blockers
- Syntax errors present?
- Undefined variables or functions?
- Uses `sys.argv` or `input()`? **Critical error**

### File Handling
- Uses exact filenames from list?
- No generic names: `file.json`, `data.csv`?
- Filenames hardcoded (no `sys.argv`)?
{json_check}
### Critical Errors to Flag
Must flag these patterns:
- `if len(sys.argv) > 1:`
- `main(sys.argv[1])`
- `input('Enter:')`
- `argparse`

## Response Format
Return JSON:
```json
{{
  "verified": true/false,
  "issues": ["issue1", "issue2", ...]
}}
```

**Examples:**
- `{{"verified": true, "issues": []}}` - Code will execute, answers question, correct filenames
- `{{"verified": false, "issues": [...]}}` - Execution blocker detected

Focus on execution errors, not style.

Think hard about potential runtime failures."""


def get_self_verification_section(query: str, has_json_files: bool = False) -> str:
    """Self-verification for combined generation+verification."""
    task_ref = query.split('\n')[0][:10000000]

    json_step = """
**3. JSON Safety:** .get() for access? isinstance() check? try/except?
""" if has_json_files else """
**3. File Safety:** try/except for FileNotFoundError?
"""

    return f"""

## Self-Verification

**1. Task:** Does code answer "{task_ref}"?
   - Bad: Reject if partial answer or different task

**2. Filenames:** ALL hardcoded and exact?
   - Bad: Reject if generic names, sys.argv, input(), argparse
{json_step}
## Response Format

```json
{{
  "code": "python code string",
  "self_check_passed": true/false,
  "issues": ["list of issues or empty"]
}}
```

Set "self_check_passed": true ONLY if ALL checks pass.
Respond with ONLY the JSON object:"""


def get_output_adequacy_prompt(
    query: str,
    code: str,
    output: str,
    context: Optional[str] = None
) -> str:
    """
    Verify if code output adequately answers the user's query.
    Returns: adequate (bool), reason (str), suggestion (str).

    Args:
        query: User's question
        code: Generated code
        output: Code execution output
        context: Optional additional context

    Returns:
        Output adequacy verification prompt
    """
    context_section = f"\n## Additional Context\n{context}\n" if context else ""

    return f"""You are a code output evaluator specializing in data analysis quality assurance.

## Original Query
{query}
{context_section}
## Generated Code
```python
{code}
```

## Code Output
```
{output[:5000]}
```

## Your Task
Evaluate whether the output adequately answers the query. Consider:
- Does it directly address the question?
- Are results complete and specific?
- Are calculations/analysis correct?
- Is output format appropriate?
- Any errors or warnings present?

## Response Format
Return JSON only:
```json
{{
  "adequate": true/false,
  "reason": "Brief explanation of your decision",
  "suggestion": "How to improve if inadequate (empty if adequate)"
}}
```

Be lenient - if output provides useful information, consider it adequate.

Think hard about whether the user's question is truly answered."""
