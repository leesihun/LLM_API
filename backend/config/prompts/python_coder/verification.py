"""
Code Verification Prompts
Semantic verification focused on correctness, not style.
"""

from typing import Optional
from ..base import section_border, MARKER_OK, MARKER_ERROR, MARKER_CRITICAL


def get_verification_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    code: str,
    has_json_files: bool = False
) -> str:
    """Semantic code verification - focuses on execution errors."""
    
    json_check = f"""
[4] JSON SAFETY:
   - Uses EXACT filename from list?
   - Has isinstance() check?
   - Uses .get() for dict access?
   - ONLY uses keys from Access Patterns?
""" if has_json_files else ""
    
    return f"""You are a Python code verifier. Find potential EXECUTION ERRORS.

{MARKER_CRITICAL} Find problems causing execution failures or incorrect results.

User Question: {query}
{f"Context: {context}" if context else ""}

{file_context}

Code to verify:
```python
{code}
```

{section_border("VERIFICATION CHECKLIST")}

[1] LOGIC: Does code address the question? Correct calculations?

[2] EXECUTION BLOCKERS:
   - Syntax errors?
   - Undefined variables/functions?
   - Uses sys.argv or input()? {MARKER_ERROR} CRITICAL ERROR
   
[3] FILE HANDLING:
   - Uses EXACT filenames from list?
   - NO generic names (file.json, data.csv)?
   - Filenames HARDCODED (no sys.argv)?
{json_check}

{MARKER_ERROR} CRITICAL ERRORS TO CATCH:
- if len(sys.argv) > 1: ... {MARKER_ERROR} MUST FLAG
- main(sys.argv[1]) {MARKER_ERROR} MUST FLAG  
- input('Enter:') {MARKER_ERROR} MUST FLAG
- argparse {MARKER_ERROR} MUST FLAG

{section_border("RESPONSE FORMAT")}

Return JSON: {{"verified": true/false, "issues": ["issue1", ...]}}

{MARKER_OK} {{"verified": true, "issues": []}} - Code will execute, answers question, correct filenames
{MARKER_ERROR} {{"verified": false, "issues": [...]}} - Any execution blocker detected

Focus on EXECUTION ERRORS, not style."""


def get_self_verification_section(query: str, has_json_files: bool = False) -> str:
    """Self-verification for combined generation+verification."""
    task_ref = query.split('\n')[0][:100]
    
    json_step = """
[3] JSON Safety: .get() for access? isinstance() check? try/except?
""" if has_json_files else """
[3] File Safety: try/except for FileNotFoundError?
"""
    
    return f"""

{section_border("SELF-VERIFICATION")}

[1] Task: Does code answer "{task_ref}"?
    {MARKER_ERROR} Reject if: partial answer or different task

[2] Filenames: ALL hardcoded and exact?
    {MARKER_ERROR} Reject if: generic names, sys.argv, input(), argparse
{json_step}
{section_border("RESPONSE FORMAT")}

{{
  "code": "python code string",
  "self_check_passed": true/false,
  "issues": ["list of issues or empty"]
}}

Set "self_check_passed": true ONLY if ALL checks pass.
Respond with ONLY the JSON object:"""


def get_output_adequacy_prompt(
    query: str,
    code: str,
    output: str,
    context: Optional[str] = None
) -> str:
    """Check if output adequately answers the question."""
    return f"""Analyze if this output adequately answers the user's question.

User Question: {query}
{f"Context: {context}" if context else ""}

Code:
```python
{code}
```

Output:
```
{output[:5000]}
```

{section_border("EVALUATION")}

1. Does output contain requested information?
2. Is output clear and understandable?
3. Any errors or warnings?
4. Is output complete?

{section_border("RESPONSE FORMAT")}

{{
  "adequate": true/false,
  "reason": "Brief explanation",
  "suggestion": "Changes needed if not adequate, empty if adequate"
}}

Be lenient - if output provides useful information, consider it adequate.
Respond with ONLY JSON:"""
