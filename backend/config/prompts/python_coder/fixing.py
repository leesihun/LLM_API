"""
Code Fixing Prompts
Prompts for fixing code issues and execution errors.
"""

from typing import Optional, List, Dict, Any, Tuple
from ..base import section_border, MARKER_OK, MARKER_ERROR, MARKER_CRITICAL


def _format_namespace_for_prompt(namespace: Dict[str, Any], max_items: int = 8) -> str:
    """
    Format namespace dict for inclusion in LLM prompt.
    
    Args:
        namespace: Variable namespace from execution
        max_items: Maximum number of variables to include
        
    Returns:
        Formatted string showing variable states
    """
    if not namespace:
        return "  (No variables captured)"
    
    lines = []
    for i, (name, info) in enumerate(namespace.items()):
        if i >= max_items:
            lines.append(f"  ... and {len(namespace) - max_items} more variables")
            break
            
        var_type = info.get("type", "unknown")
        
        # Format based on type
        if "shape" in info:
            # DataFrame or ndarray
            shape = info.get("shape", [])
            if "columns" in info:
                cols = info.get("columns", [])[:5]
                cols_str = ", ".join(str(c) for c in cols) + ("..." if len(info.get("columns", [])) > 5 else "")
                lines.append(f"  - {name}: {var_type} (shape={shape}, columns=[{cols_str}])")
            else:
                lines.append(f"  - {name}: {var_type} (shape={shape})")
        elif "length" in info:
            # List
            length = info.get("length", 0)
            if length == 0:
                lines.append(f"  - {name}: {var_type} (EMPTY - length=0) <-- LIKELY CAUSE OF ERROR")
            else:
                lines.append(f"  - {name}: {var_type} (length={length})")
        elif "keys" in info:
            # Dict
            keys = info.get("keys", [])[:5]
            keys_str = ", ".join(f"'{k}'" for k in keys) + ("..." if len(info.get("keys", [])) > 5 else "")
            lines.append(f"  - {name}: {var_type} (keys=[{keys_str}])")
        elif "value" in info:
            # Simple type
            val = str(info.get("value", ""))[:50]
            lines.append(f"  - {name}: {var_type} = {val}")
        else:
            lines.append(f"  - {name}: {var_type}")
    
    return "\n".join(lines) if lines else "  (No variables captured)"


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
2. Filenames must remain HARDCODED
3. {MARKER_ERROR}
4. You may use other logics and approaches if needed
5. Think hard for making changes, Aim for the best solution

Generate corrected Python code only:"""


def get_execution_fix_prompt(
    query: str,
    context: Optional[str],
    code: str,
    error_message: str,
    error_namespace: Optional[Dict[str, Any]] = None
) -> str:
    """Fix code that failed during execution."""
    
    # Build debug context section if namespace available
    debug_section = ""
    if error_namespace:
        formatted_ns = _format_namespace_for_prompt(error_namespace)
        debug_section = f"""
{section_border("DEBUG CONTEXT - Variable state when error occurred")}

{formatted_ns}

Use this information to understand WHY the error happened, not just WHAT the error was.
"""
    
    return f"""Fix the Python code that failed during execution:

Original request: {query}
{f"Context: {context}" if context else ""}

Current code:
```python
{code}
```

Execution error:
{error_message}
{debug_section}
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


def get_retry_prompt_with_history(
    query: str,
    file_context: str,
    attempt_history: List[Dict[str, Any]],
    current_attempt: int,
    max_attempts: int,
    has_json_files: bool = False
) -> str:
    """
    Generate code with full knowledge of what already failed.
    
    This prompt shows the model its previous failed attempts so it can
    learn from mistakes and try fundamentally different approaches.
    
    Args:
        query: User's original query/task
        file_context: File metadata and context
        attempt_history: List of previous failed attempts with code, error, namespace
        current_attempt: Current attempt number (1-indexed)
        max_attempts: Maximum number of attempts allowed
        has_json_files: Whether JSON files are involved
        
    Returns:
        Complete prompt for code generation with history context
    """
    
    # Build history section
    history_section = ""
    if attempt_history:
        history_lines = []
        history_lines.append(section_border("PREVIOUS FAILED ATTEMPTS - DO NOT REPEAT THESE MISTAKES"))
        
        for prev in attempt_history:
            attempt_num = prev.get("attempt", "?")
            error_type_info = prev.get("error_type", ("Unknown", "No guidance"))
            error_type = error_type_info[0] if isinstance(error_type_info, tuple) else str(error_type_info)
            error_guidance = error_type_info[1] if isinstance(error_type_info, tuple) and len(error_type_info) > 1 else ""
            error_msg = str(prev.get("error", "") or prev.get("execution_error", ""))[:500]
            prev_code = prev.get("code", "")
            namespace = prev.get("namespace", {})
            
            history_lines.append(f"\n{'='*50}")
            history_lines.append(f"ATTEMPT {attempt_num} - FAILED with {error_type}")
            history_lines.append(f"{'='*50}")
            
            if error_guidance:
                history_lines.append(f"Guidance: {error_guidance}")
            
            history_lines.append(f"\nError message:\n{error_msg}")
            
            if prev_code:
                # Show code with truncation for very long code
                code_preview = prev_code[:1200]
                if len(prev_code) > 1200:
                    code_preview += "\n# ... (code truncated for brevity) ..."
                history_lines.append(f"\nCode that failed:\n```python\n{code_preview}\n```")
            
            if namespace:
                ns_formatted = _format_namespace_for_prompt(namespace, max_items=6)
                history_lines.append(f"\nVariables at failure point:\n{ns_formatted}")
        
        history_section = "\n".join(history_lines)
    
    # Determine escalating strategy based on attempt number
    if current_attempt == 2:
        strategy = f"""{section_border("RETRY STRATEGY - ATTEMPT 2")}

Try a DIFFERENT approach than attempt 1:
- If you used pandas before, try pure Python (or vice versa)
- If you accessed data one way, try a different access pattern
- If you assumed a data structure, verify it first with print() or type()
- Add defensive checks: len(), isinstance(), .get() for dicts
"""
    elif current_attempt >= 3:
        strategy = f"""{section_border(f"CRITICAL - ATTEMPT {current_attempt}/{max_attempts}")}

{MARKER_CRITICAL} PREVIOUS APPROACHES FUNDAMENTALLY DON'T WORK!

You MUST:
1. COMPLETELY RETHINK the approach - do something DIFFERENT
2. Add debugging: print(type(data)), print(len(data)), print(data.keys() if dict else data[:3])
3. Consider: Is the data format different than expected?
4. Use maximum defensive coding: check everything before accessing
5. If a library isn't working, try a different one or pure Python

{MARKER_ERROR} DO NOT just add try/except around the same broken code!
"""
    else:
        strategy = ""
    
    # Build the complete prompt
    prompt_parts = [
        f"{section_border('TASK')}",
        "",
        query,
        "",
        file_context,
        "",
    ]
    
    if history_section:
        prompt_parts.append(history_section)
        prompt_parts.append("")
    
    if strategy:
        prompt_parts.append(strategy)
        prompt_parts.append("")
    
    # Add JSON-specific guidance if needed
    if has_json_files:
        prompt_parts.append(f"""{section_border("JSON HANDLING REMINDERS")}

- Use: with open('file.json', 'r', encoding='utf-8') as f: data = json.load(f)
- ALWAYS check type first: print(type(data), isinstance(data, dict))
- For dicts: use data.get('key', default) not data['key']
- For lists: check len(data) before accessing data[0]
- Print structure first: print(json.dumps(data, indent=2)[:500])
""")
    
    prompt_parts.append(f"""{section_border("OUTPUT REQUIREMENTS")}

{MARKER_OK} CORRECT approach:
- Verify data loaded correctly before processing
- Use defensive access patterns (.get(), len() checks, isinstance())
- Print results directly: print(df) or print(result)
- Pandas will show ALL data (display options pre-configured)

{MARKER_ERROR} WRONG approach:
- Assuming data structure without checking
- Direct indexing without length check: data[0]
- Direct dict access without .get(): data['key']
- Saving ordinary results to files (CSV, TXT) - just print them

Generate ONLY executable Python code (no markdown, no explanations):
""")
    
    return "\n".join(prompt_parts)
