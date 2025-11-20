"""
Python Code Generation Prompts
Prompts for Python code generation, verification, modification, and error fixing.
"""

from typing import Optional, List


def get_python_code_generation_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    is_prestep: bool = False,
    has_json_files: bool = False,
    conversation_history: Optional[List[dict]] = None,
    plan_context: Optional[dict] = None,
    react_context: Optional[dict] = None
) -> str:
    """
    Prompt for generating Python code.
    Used in: backend/tools/python_coder_tool.py (_generate_code)

    Args:
        query: User's task (the ACTUAL question user asked)
        context: Optional additional context
        file_context: File information
        is_prestep: Whether this is pre-step (fast analysis mode)
        has_json_files: Whether JSON files are present
        conversation_history: List of past conversation turns (dicts with 'role', 'content', 'timestamp')
        plan_context: Plan-Execute context (dict with 'current_step', 'total_steps', 'plan', 'previous_results')
        react_context: ReAct context (dict with 'iteration', 'history' containing failed attempts with code and errors)
    """
    if is_prestep:
        # Pre-step mode: fast analysis
        prompt_parts = [
            "You are a Python code generator in FAST PRE-ANALYSIS MODE.",
            "Your goal is to quickly analyze the attached files and provide an immediate answer to the user's question.",
            "",
            "IMPORTANT: Do NOT use Unicode emojis in your response. Use ASCII-safe markers like [OK], [X], [WARNING], [!!!] instead.",
            "",
            f"Task: {query}",
            "",
            file_context,
            "",
            "PRE-STEP MODE INSTRUCTIONS:",
            "- This is the FIRST attempt to answer the question using ONLY the provided files",
            "- Generate DIRECT, FOCUSED code that answers the specific question",
            "- Prioritize SPEED and CLARITY over comprehensive analysis"
        ]

        if file_context:
            prompt_parts.extend([
                "[!!!] CRITICAL: Use the EXACT filenames shown in the file list above",
                "[!!!] DO NOT use generic names like 'file.json', 'data.csv', 'input.json', etc.",
                "[!!!] COPY the actual filename from the list - character by character",
                "- NEVER makeup data, ALWAYS use the real files provided"
            ])

        prompt_parts.extend([
            "",
            "[!!!] EXECUTION ENVIRONMENT (CRITICAL - READ CAREFULLY):",
            "- Code will be executed via subprocess WITHOUT command-line arguments",
            "- DO NOT use sys.argv - it will be empty (only script name)",
            "- DO NOT use input() - this is non-interactive execution",
            "- ALL filenames MUST be HARDCODED directly in the code",
            "- Files are in the current working directory - use filenames directly",
            "- If you create functions, call them with HARDCODED filenames in main code",
            "",
            "[X] FORBIDDEN PATTERNS:",
            "  if __name__ == '__main__':",
            "      import sys",
            "      if len(sys.argv) > 1:",
            "          main(sys.argv[1])  # [X] WRONG - no arguments available!",
            "",
            "[OK] CORRECT PATTERN:",
            "  if __name__ == '__main__':",
            "      main('complex_json.json')  # [OK] CORRECT - hardcoded filename",
            "",
            "- Output results using print() statements with clear labels",
            "- Include basic error handling (try/except)",
            "- Focus on the MOST RELEVANT data columns/fields for the question",
            "",
            "CODE STYLE:",
            "- Keep it simple and direct",
            "- Use pandas/numpy for data files",
            "- Print intermediate steps for transparency",
            "- Always use real data from files, NO fake data, NO placeholders"
        ])

        if has_json_files:
            prompt_parts.extend([
                "",
                "JSON FILE HANDLING (CRITICAL - READ CAREFULLY):",
                "1. ALWAYS use: with open('EXACT_FILENAME_FROM_LIST.json', 'r', encoding='utf-8') as f: data = json.load(f)",
                "   [!!!] Replace 'EXACT_FILENAME_FROM_LIST.json' with the ACTUAL filename from the file list above!",
                "2. Wrap in try/except json.JSONDecodeError for error handling",
                "3. Check structure type FIRST: isinstance(data, dict) or isinstance(data, list)",
                "4. Use .get() method for dict access: data.get('key', default) NEVER data['key']",
                "5. ONLY use keys from \"Access Patterns\" section - DO NOT make up or guess keys",
                "6. For nested access, validate each level: data.get('parent', {}).get('child', default)",
                "7. For arrays, check length first: if len(data) > 0: item = data[0]",
                "8. COPY the \"Access Patterns\" shown above - they are structure-validated",
                "9. Handle None/null values: if value is not None: process(value)",
                "10. Add debug prints: print(\"Data type:\", type(data), \"Keys:\", list(data.keys()) if isinstance(data, dict) else 'N/A')"
            ])

        prompt_parts.append("\nGenerate ONLY the Python code, no explanations or markdown:")
        return "\n".join(prompt_parts)

    else:
        # Normal mode - NEW STRUCTURE
        # Order: HISTORY → INPUT → PLANS → REACTS → TASK → METADATA → RULES → CHECKLISTS
        prompt_parts = []

        # ═══════════════════════════════════════════════════════════════
        # SECTION 1: PAST HISTORIES (conversation context)
        # ═══════════════════════════════════════════════════════════════
        if conversation_history and len(conversation_history) > 0:
            prompt_parts.extend([
                "="*80,
                "PAST HISTORIES".center(80),
                "="*80,
                ""
            ])

            for idx, turn in enumerate(conversation_history, 1):
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                timestamp = turn.get('timestamp', '')

                if role == 'user':
                    prompt_parts.append(f"=== Turn {idx} (User) ===")
                elif role == 'assistant':
                    prompt_parts.append(f"=== Turn {idx} (AI) ===")
                else:
                    prompt_parts.append(f"=== Turn {idx} ({role}) ===")

                if timestamp:
                    prompt_parts.append(f"Time: {timestamp}")

                # Truncate very long content
                if len(content) > 500:
                    prompt_parts.append(f"{content[:500]}...")
                    prompt_parts.append(f"[Content truncated - {len(content)} chars total]")
                else:
                    prompt_parts.append(content)

                prompt_parts.append("")

            prompt_parts.extend([
                "="*80,
                ""
            ])

        # ═══════════════════════════════════════════════════════════════
        # SECTION 2: MY ORIGINAL INPUT PROMPT (User's actual question)
        # ═══════════════════════════════════════════════════════════════
        prompt_parts.extend([
            "="*80,
            "MY ORIGINAL INPUT PROMPT".center(80),
            "="*80,
            "",
            f"{query}",
            ""
        ])

        if context:
            prompt_parts.append(f"[ADDITIONAL CONTEXT] {context}")
            prompt_parts.append("")

        prompt_parts.extend([
            "="*80,
            ""
        ])

        # ═══════════════════════════════════════════════════════════════
        # SECTION 3: PLANS (from plan-execute workflow)
        # ═══════════════════════════════════════════════════════════════
        if plan_context:
            prompt_parts.extend([
                "="*80,
                "PLANS".center(80),
                "="*80,
                "",
                f"[Plan-Execute Workflow - Step {plan_context.get('current_step', '?')} of {plan_context.get('total_steps', '?')}]",
                ""
            ])

            if 'plan' in plan_context:
                prompt_parts.append("Full Plan:")
                for step in plan_context['plan']:
                    step_num = step.get('step_number', '?')
                    goal = step.get('goal', '')
                    status = step.get('status', 'pending')

                    status_marker = ""
                    if status == 'completed':
                        status_marker = " [OK] COMPLETED"
                    elif status == 'current':
                        status_marker = " <- CURRENT STEP"
                    elif status == 'failed':
                        status_marker = " [X] FAILED"

                    prompt_parts.append(f"  Step {step_num}: {goal}{status_marker}")

                    if 'success_criteria' in step:
                        prompt_parts.append(f"    Success criteria: {step['success_criteria']}")
                    if 'primary_tools' in step:
                        prompt_parts.append(f"    Primary tools: {', '.join(step['primary_tools'])}")

                prompt_parts.append("")

            if 'previous_results' in plan_context and plan_context['previous_results']:
                prompt_parts.append("Previous Steps Results:")
                for result in plan_context['previous_results']:
                    step_num = result.get('step_number', '?')
                    summary = result.get('summary', '')
                    prompt_parts.append(f"  Step {step_num} -> {summary}")
                prompt_parts.append("")

            prompt_parts.extend([
                "="*80,
                ""
            ])

        # ═══════════════════════════════════════════════════════════════
        # SECTION 4: REACTS (from ReAct iterations with FAILED CODES)
        # ═══════════════════════════════════════════════════════════════
        if react_context and 'history' in react_context:
            prompt_parts.extend([
                "="*80,
                "REACTS".center(80),
                "="*80,
                "",
                "[ReAct Iteration History]",
                ""
            ])

            current_iteration = react_context.get('iteration', 1)

            for idx, iteration in enumerate(react_context['history'], 1):
                prompt_parts.append(f"=== Iteration {idx} ===")

                # Thought
                if 'thought' in iteration:
                    prompt_parts.append(f"Thought: {iteration['thought']}")

                # Action
                if 'action' in iteration:
                    prompt_parts.append(f"Action: {iteration['action']}")

                # Tool Input (if any)
                if 'tool_input' in iteration:
                    prompt_parts.append(f"Tool Input: {iteration['tool_input']}")

                # Generated Code (if failed)
                if 'code' in iteration:
                    prompt_parts.append("")
                    prompt_parts.append("Generated Code:")
                    prompt_parts.append("```python")
                    # Show first 30 lines of code
                    code_lines = iteration['code'].split('\n')
                    for line in code_lines[:30]:
                        prompt_parts.append(line)
                    if len(code_lines) > 30:
                        prompt_parts.append(f"... [{len(code_lines) - 30} more lines]")
                    prompt_parts.append("```")

                # Observation (error/result)
                if 'observation' in iteration:
                    obs = iteration['observation']
                    prompt_parts.append("")
                    if iteration.get('status') == 'error':
                        prompt_parts.append(f"Observation: [ERROR] {obs}")

                        # Add error reason if available
                        if 'error_reason' in iteration:
                            prompt_parts.append(f"Error Reason: {iteration['error_reason']}")
                    else:
                        prompt_parts.append(f"Observation: {obs}")

                prompt_parts.append("")

            # Current iteration marker
            prompt_parts.append(f"=== Iteration {current_iteration} (CURRENT) ===")
            prompt_parts.append("Awaiting code generation...")
            prompt_parts.append("")

            prompt_parts.extend([
                "="*80,
                ""
            ])

        # ═══════════════════════════════════════════════════════════════
        # SECTION 5: FINAL TASK FOR LLM AT THIS STAGE
        # ═══════════════════════════════════════════════════════════════
        prompt_parts.extend([
            "="*80,
            "FINAL TASK FOR LLM AT THIS STAGE".center(80),
            "="*80,
            ""
        ])

        # Detect task type and provide specific guidance
        query_lower = query.lower()
        is_aggregation = any(word in query_lower for word in ['sum', 'total', 'count', 'average', 'mean', 'median', 'calculate'])
        is_visualization = any(word in query_lower for word in ['plot', 'graph', 'chart', 'visualize', 'draw'])
        is_analysis = any(word in query_lower for word in ['analyze', 'report', 'summary', 'summarize', 'insights'])

        if is_visualization:
            prompt_parts.extend([
                "[TASK TYPE] Visualization/Plotting",
                "",
                "Workflow:",
                "  1. Import matplotlib: import matplotlib.pyplot as plt",
                "  2. Load file data (use access patterns from METADATA below)",
                "  3. Extract x, y values",
                "  4. Create plot",
                "  5. Save: plt.savefig('output.png')",
                ""
            ])
        elif is_aggregation:
            prompt_parts.extend([
                "[TASK TYPE] Calculation/Aggregation",
                "",
                "Workflow:",
                "  1. Load file data (use access patterns from METADATA below)",
                "  2. Extract relevant field",
                "  3. Calculate result",
                "  4. Print result with label",
                ""
            ])
        elif is_analysis:
            prompt_parts.extend([
                "[TASK TYPE] Analysis/Reporting",
                "",
                "Workflow:",
                "  1. Load file data once",
                "  2. Calculate multiple metrics",
                "  3. Print results clearly",
                ""
            ])
        else:
            prompt_parts.extend([
                "[TASK TYPE] General",
                "",
                "Generate Python code to complete the task above.",
                ""
            ])

        prompt_parts.extend([
            "="*80,
            ""
        ])

        # ═══════════════════════════════════════════════════════════════
        # SECTION 6: META DATA (AVAILABLE FILES)
        # ═══════════════════════════════════════════════════════════════
        prompt_parts.extend([
            "="*80,
            "META DATA (AVAILABLE FILES)".center(80),
            "="*80,
            file_context,
            ""
        ])

        # ═══════════════════════════════════════════════════════════════
        # SECTION 7: RULES
        # ═══════════════════════════════════════════════════════════════
        prompt_parts.extend([
            "="*80,
            "RULES".center(80),
            "="*80,
            "",
            "[RULE 1] EXACT FILENAMES",
            "   - Copy EXACT filename from META DATA section above",
            "   - [X] NO generic names: 'data.json', 'file.json', 'input.csv'",
            "   - [OK] Example: filename = 'sales_report_Q4_2024.json'",
            "",
            "[RULE 2] NO COMMAND-LINE ARGS / USER INPUT",
            "   - Code runs via subprocess WITHOUT arguments",
            "   - [X] NO sys.argv, NO input(), NO argparse",
            "   - [OK] All filenames must be HARDCODED",
            "",
            "[RULE 3] USE ACCESS PATTERNS",
            "   - Copy access patterns from META DATA section",
            "   - [X] DON'T guess keys or field names",
            "   - [OK] Use .get() for safe dict access",
            ""
        ])

        if has_json_files:
            prompt_parts.extend([
                "[RULE 4] JSON SAFETY",
                "   - Use .get() for dict access: data.get('key', default)",
                "   - Check type: isinstance(data, dict) or isinstance(data, list)",
                "   - Add error handling: try/except json.JSONDecodeError",
                ""
            ])

        prompt_parts.extend([
            "="*80,
            ""
        ])

        # ═══════════════════════════════════════════════════════════════
        # SECTION 8: CHECKLISTS
        # ═══════════════════════════════════════════════════════════════
        prompt_parts.extend([
            "="*80,
            "CHECKLISTS".center(80),
            "="*80,
            "",
            "[1] Task Completion",
            "    ? Does code answer the original prompt?",
            "    ? Does code produce the expected output?",
            "",
            "[2] Filename Validation",
            "    ? Are ALL filenames from META DATA section (exact match)?",
            "    ? NO generic names (data.json, file.csv, input.xlsx)?",
            "    ? NO sys.argv, input(), or argparse?",
            "",
            "[3] Safety & Error Handling",
            "    ? try/except for file operations?",
            "    ? .get() for dict access (JSON)?",
            "    ? Type checks with isinstance()?",
            "",
            "[4] Access Patterns",
            "    ? Using access patterns from META DATA section?",
            "    ? NOT guessing keys or field names?",
            "",
            "="*80,
            "",
            "Generate ONLY executable Python code (no markdown, no explanations):",
            ""
        ])
        return "\n".join(prompt_parts)


def get_python_code_verification_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    code: str,
    has_json_files: bool = False
) -> str:
    """
    Prompt for verifying Python code before execution.
    Used in: backend/tools/python_coder_tool.py (_llm_verify_answers_question)

    Args:
        query: User's question
        context: Optional additional context
        file_context: File information
        code: Code to verify
        has_json_files: Whether JSON files are present
    """
    prompt_parts = [
        "You are a STRICT Python code verifier. Your job is to identify ANY potential errors or issues in the code.",
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
        "[2] SYNTAX & RUNTIME ERRORS:",
        "   - Any syntax errors (missing colons, parentheses, quotes)?",
        "   - Undefined variables or functions?",
        "   - Import statements correct?",
        "   - Blocked/dangerous modules (socket, subprocess, eval, exec)?",
        "",
        "[3] ERROR HANDLING:",
        "   - Try/except blocks present where needed?",
        "   - File operations wrapped in error handling?",
        "   - Division by zero checks if applicable?",
        ""
    ])

    if file_context:
        prompt_parts.extend([
            "[4] FILE HANDLING:",
            "   - Uses EXACT filenames from the file list?",
            "   - NO generic names like 'file.json', 'data.csv', 'input.xlsx'?",
            "   - File paths are strings, properly quoted?",
            "   - Uses ONLY real data (NO fake/placeholder data)?",
            "   - File reading has error handling (FileNotFoundError)?",
            ""
        ])

    if has_json_files:
        prompt_parts.extend([
            "[5] JSON FILE HANDLING (CRITICAL):",
            "   - Uses EXACT JSON filename from file list (NOT 'file.json', 'data.json')?",
            "   - Has isinstance() check for data structure validation?",
            "   - Uses .get() for dict access (NEVER data['key'])?",
            "   - Checks for None/null values before nested access?",
            "   - ONLY uses keys from \"[PATTERNS] Access Patterns\" (NO guessing keys)?",
            "   - Arrays checked with len() before indexing?",
            "   - Follows the \"[PATTERNS] Access Patterns\" exactly?",
            "   - Has json.JSONDecodeError handling?",
            ""
        ])

    prompt_parts.extend([
        "[6] EXECUTION COMPATIBILITY (CRITICAL - MUST CATCH THESE!):",
        "   - [X] Does code use sys.argv? (FORBIDDEN - no command-line arguments available!)",
        "   - [X] Does code check len(sys.argv) or access sys.argv[1], sys.argv[2], etc.?",
        "   - [X] Does code use input() for user interaction? (FORBIDDEN - non-interactive)",
        "   - [X] Does code print 'Usage:' messages expecting command-line args?",
        "   - [OK] Are ALL filenames HARDCODED in the code?",
        "   - [OK] If main() or other functions are called, are filenames passed as HARDCODED strings?",
        "   - [OK] Does code run standalone without any external input or arguments?",
        "",
        "   [!!!] COMMON BAD PATTERNS TO FLAG:",
        "   - if len(sys.argv) > 1: ... (MUST FLAG THIS!)",
        "   - main(sys.argv[1]) (MUST FLAG THIS!)",
        "   - filename = sys.argv[1] if len(sys.argv) > 1 else 'default.json' (MUST FLAG THIS!)",
        "   - parser = argparse.ArgumentParser() (MUST FLAG THIS!)",
        "   - input('Enter filename:') (MUST FLAG THIS!)",
        "",
        "[!!!] ERROR DETECTION PRIORITY:",
        "- Your primary goal is to find potential ERRORS (not style issues)",
        "- Focus on issues that will cause EXECUTION FAILURES or WRONG RESULTS",
        "- Be STRICT - even small issues can cause failures",
        "- If uncertain about filename correctness, mark it as an issue",
        "- ESPECIALLY check for sys.argv usage - this is the #1 failure cause",
        "",
        "[RESPONSE FORMAT]:",
        'Return a JSON object: {"verified": true/false, "issues": ["issue1", "issue2", ...]}',
        "",
        '[OK] Return {"verified": true, "issues": []} ONLY IF:',
        "   - Code is 100% correct and will execute without errors",
        "   - All filenames are exact matches from the file list",
        "   - All filenames are HARDCODED (no sys.argv, no input())",
        "   - All required safety checks are present",
        f"{'   - All JSON safety patterns are followed' if has_json_files else ''}",
        "",
        '[X] Return {"verified": false, "issues": [...]} IF:',
        "   - ANY potential error detected (syntax, runtime, logic)",
        "   - Uses sys.argv or input() (CRITICAL ERROR)",
        "   - Filenames don't match EXACTLY",
        "   - Filenames not hardcoded in the code",
        "   - Missing error handling",
        "   - Unsafe data access patterns",
        f"{'   - JSON access patterns not followed' if has_json_files else ''}",
        "",
        "[WARNING] BE THOROUGH: It's better to flag a potential issue than miss a real error.",
        ""
    ])

    return "\n".join(prompt_parts)


def get_python_code_modification_prompt(
    query: str,
    context: Optional[str],
    code: str,
    issues: List[str]
) -> str:
    """
    Prompt for modifying Python code to fix issues.
    Used in: backend/tools/python_coder_tool.py (_modify_code)

    Args:
        query: Original user query
        context: Optional additional context
        code: Current code
        issues: List of issues to fix
    """
    return f"""Fix the following Python code to address these issues:

IMPORTANT: Do NOT use Unicode emojis in your response. Use ASCII-safe markers like [OK], [X], [WARNING], [!!!] instead.

Original request: {query}
{f"Context: {context}" if context else ""}

Current code:
```python
{code}
```

Issues to fix:
{chr(10).join(f"- {issue}" for issue in issues)}

Generate the corrected Python code. Output ONLY the code, no explanations:"""


def get_python_code_execution_fix_prompt(
    query: str,
    context: Optional[str],
    code: str,
    error_message: str
) -> str:
    """
    Prompt for fixing Python code after execution error.
    Used in: backend/tools/python_coder_tool.py (_fix_execution_error)

    Args:
        query: Original user query
        context: Optional additional context
        code: Current code that failed
        error_message: Error from execution
    """
    return f"""Fix the following Python code that failed during execution:

IMPORTANT: Do NOT use Unicode emojis in your response. Use ASCII-safe markers like [OK], [X], [WARNING], [!!!] instead.

Original request: {query}
{f"Context: {context}" if context else ""}

Current code:
```python
{code}
```

Execution error:
{error_message}

Analyze the error and fix the code. Output ONLY the corrected code, no explanations:"""


def get_code_generation_with_self_verification_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    is_prestep: bool = False,
    has_json_files: bool = False,
    conversation_history: Optional[List[dict]] = None,
    plan_context: Optional[dict] = None,
    react_context: Optional[dict] = None
) -> str:
    """
    OPTIMIZED: Combined code generation + self-verification prompt.
    Generates code AND verifies it in a single LLM call.

    Args:
        query: User's task
        context: Optional additional context
        file_context: File information
        is_prestep: Whether this is pre-step (fast analysis mode)
        has_json_files: Whether JSON files are present
        conversation_history: List of past conversation turns
        plan_context: Plan-Execute context
        react_context: ReAct context with failed attempts

    Returns:
        Prompt that requests JSON response with code and self-check
    """
    # Build generation prompt (reuse existing logic)
    generation_prompt = get_python_code_generation_prompt(
        query=query,
        context=context,
        file_context=file_context,
        is_prestep=is_prestep,
        has_json_files=has_json_files,
        conversation_history=conversation_history,
        plan_context=plan_context,
        react_context=react_context
    )

    # Remove the last line "Generate ONLY the Python code, no explanations or markdown:"
    generation_lines = generation_prompt.split('\n')
    generation_prompt_clean = '\n'.join(generation_lines[:-1])

    # Add self-verification instructions
    verification_instructions = f"""

[CHECK] SELF-VERIFICATION CHECKLIST - Step-by-Step Validation:

[STEP 1] Task Validation
   ? Question: Does my code directly answer "{query}"?
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

    return generation_prompt_clean + verification_instructions


def get_output_adequacy_check_prompt(
    query: str,
    code: str,
    output: str,
    context: Optional[str] = None
) -> str:
    """
    OPTIMIZED: Check if code execution output adequately answers the user's question.

    Args:
        query: Original user query
        code: The Python code that was executed
        output: The output from executing the code
        context: Optional additional context

    Returns:
        Prompt for checking output adequacy
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
{output[:2000]}  # First 2000 chars
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
