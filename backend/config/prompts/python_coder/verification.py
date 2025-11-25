"""
Code Verification Prompts
Provides semantic verification focused on correctness, not style.
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
    Semantic code verification - focuses on errors, not style.

    Args:
        query: User's question
        context: Optional additional context
        file_context: File information
        code: Code to verify
        has_json_files: Whether JSON files are present

    Returns:
        Verification prompt
    """
    prompt_parts = [
        "You are a Python code verifier. Your job is to identify potential EXECUTION ERRORS that will cause code to fail.",
        "",
        "IMPORTANT: Do NOT use Unicode emojis in your response. Use ASCII-safe markers like [OK], [X], [WARNING], [!!!] instead.",
        "",
        "[!!!] VERIFICATION MODE: Find problems that could cause execution failures or incorrect results.",
        "",
        f"User Question: {query}",
        ""
    ]

    if context:
        prompt_parts.append(f"Context: {context}")
        prompt_parts.append("")

    prompt_parts.extend([
        file_context,
        "",
        "Code to verify:",
        "```python",
        code,
        "```",
        "",
        "[CHECK] CRITICAL VERIFICATION CHECKLIST:",
        "",
        "[1] LOGIC & CORRECTNESS:",
        "   - Does the code address the user's specific question?",
        "   - Will it produce the expected output?",
        "   - Are calculations/operations logically correct?",
        "",
        "[2] EXECUTION BLOCKERS:",
        "   - Syntax errors (missing colons, parentheses, quotes)?",
        "   - Undefined variables or functions?",
        "   - Import statements correct?",
        "   - Uses sys.argv or input()? (CRITICAL ERROR - forbidden)",
        ""
    ])

    if file_context:
        prompt_parts.extend([
            "[3] FILE HANDLING:",
            "   - Uses EXACT filenames from the file list?",
            "   - NO generic names like 'file.json', 'data.csv'?",
            "   - All filenames are HARDCODED (no variables from sys.argv)?",
            ""
        ])

    if has_json_files:
        prompt_parts.extend([
            "[4] JSON SAFETY:",
            "   - Uses EXACT JSON filename from file list?",
            "   - Has isinstance() check for data structure?",
            "   - Uses .get() for dict access (safer than data['key'])?",
            "   - ONLY uses keys from \"Access Patterns\" section?",
            ""
        ])

    prompt_parts.extend([
        "[!!!] COMMON CRITICAL ERRORS TO CATCH:",
        "- if len(sys.argv) > 1: ... (MUST FLAG - no arguments available)",
        "- main(sys.argv[1]) (MUST FLAG)",
        "- input('Enter filename:') (MUST FLAG - non-interactive)",
        "- parser = argparse.ArgumentParser() (MUST FLAG)",
        "",
        "[RESPONSE FORMAT]:",
        'Return a JSON object: {"verified": true/false, "issues": ["issue1", "issue2", ...]}',
        "",
        '[OK] Return {"verified": true, "issues": []} ONLY IF:',
        "   - Code will execute without errors",
        "   - Code answers the user's question",
        "   - All filenames are exact matches and HARDCODED",
        "   - NO sys.argv or input() usage",
        "",
        '[X] Return {"verified": false, "issues": [...]} IF:',
        "   - ANY execution blocker detected (syntax, undefined vars, sys.argv)",
        "   - Filenames don't match EXACTLY",
        "   - Missing critical error handling that will cause crashes",
        "",
        "[NOTE] Focus on EXECUTION ERRORS, not style issues. Be lenient with minor improvements.",
        ""
    ])

    return "\n".join(prompt_parts)


def get_self_verification_section(query: str, has_json_files: bool = False) -> str:
    """
    Self-verification checklist for combined generation+verification.

    Args:
        query: User's question (only used for reference, not embedded verbatim)
        has_json_files: Whether JSON files are present

    Returns:
        Self-verification instructions
    """
    # Extract a concise task reference (first 150 chars or first sentence)
    task_ref = query.split('\n')[0][:150]  # First line, max 150 chars
    if len(task_ref) < len(query):
        task_ref += "..."

    return f"""

[CHECK] SELF-VERIFICATION CHECKLIST - Step-by-Step Validation:

[STEP 1] Task Validation
   ? Question: Does my code directly answer the task described in the prompt above?
   >> Task Reference: {task_ref}
   >> Check: Code produces the requested output (not just partial answer)
   X Reject if: Code does something different or only partially addresses the question

[STEP 2] Filename Validation
   ? Question: Are ALL filenames HARDCODED and EXACT?
   >> Search for: Filenames from file list above (exact string match)
   X Reject if: ANY of these appear:
      - Generic names: 'data.json', 'file.json', 'input.csv', 'data.csv'
      - sys.argv (ANY use, including sys.argv[1], len(sys.argv))
      - input() function (user input)
      - argparse module

[STEP 3] Safety Validation{" (JSON)" if has_json_files else ""}
   ? Question: Does the code use safe patterns?{'''
   >> Check JSON access uses:
      - .get() for dict access (NOT data['key'])
      - isinstance() to check data type
      - try/except for json.JSONDecodeError
   X Reject if: Direct dict access data['key'] without .get()''' if has_json_files else '''
   >> Check file operations use:
      - try/except for FileNotFoundError
      - Error handling for file operations'''}

[STEP 4] Template Validation{" (JSON)" if has_json_files else ""}
   ? Question: Did I copy the template or write from scratch?{'''
   >> Look for: Complete template structure from file section
   X Reject if: Manually wrote JSON loading instead of using template''' if has_json_files else '''
   >> Look for: Proper file loading with error handling'''}

[RESPONSE FORMAT] REQUIRED RESPONSE FORMAT (JSON):
{{
  "code": "your python code here (as a string)",
  "self_check_passed": true or false,
  "issues": ["list of issues found during validation steps above - empty array if all steps passed"]
}}

IMPORTANT:
- Set "self_check_passed": true ONLY if ALL 4 validation steps pass
- If ANY step fails, set "self_check_passed": false and list specific issues
- Be strict with filename checking - this is the #1 cause of failures
- The code should be executable Python (no markdown, no explanations)

Generate code and self-verify using the 4-step checklist. Respond with ONLY the JSON object:"""


def get_output_adequacy_prompt(
    query: str,
    code: str,
    output: str,
    context: Optional[str] = None
) -> str:
    """
    Check if code execution output adequately answers the user's question.

    Args:
        query: Original user query
        code: The Python code that was executed
        output: The output from executing the code
        context: Optional additional context

    Returns:
        Output adequacy check prompt
    """
    return f"""Analyze if this code execution output adequately answers the user's question.

IMPORTANT: Do NOT use Unicode emojis in your response. Use ASCII-safe markers like [OK], [X], [WARNING], [!!!] instead.

User Question: {query}
{f"Context: {context}" if context else ""}

Generated Code:
```python
{code}
```

Execution Output:
```
{output[:]}
```

[CHECK] EVALUATION CRITERIA:
1. Does the output contain the information requested by the user?
2. Is the output clear and understandable?
3. Are there any errors or warnings in the output?
4. Is the output complete (not truncated or missing data)?

[RESPONSE FORMAT] REQUIRED RESPONSE FORMAT (JSON):
{{
  "adequate": true or false,
  "reason": "Brief explanation of why adequate or not",
  "suggestion": "If not adequate, what changes are needed to the code (empty string if adequate)"
}}

IMPORTANT:
- Set "adequate": true if output answers the question, even if not perfect
- Set "adequate": false only if output is clearly wrong, missing, or contains errors
- Be lenient - if the output provides useful information, consider it adequate

Respond with ONLY the JSON object:"""
