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
    has_json_files: bool = False
) -> str:
    """
    Prompt for generating Python code.
    Used in: backend/tools/python_coder_tool.py (_generate_code)

    Args:
        query: User's task
        context: Optional additional context
        file_context: File information
        is_prestep: Whether this is pre-step (fast analysis mode)
        has_json_files: Whether JSON files are present
    """
    if is_prestep:
        # Pre-step mode: fast analysis
        prompt_parts = [
            "You are a Python code generator in FAST PRE-ANALYSIS MODE.",
            "Your goal is to quickly analyze the attached files and provide an immediate answer to the user's question.",
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
                "🚨 CRITICAL: Use the EXACT filenames shown in the file list above",
                "🚨 DO NOT use generic names like 'file.json', 'data.csv', 'input.json', etc.",
                "🚨 COPY the actual filename from the list - character by character",
                "- NEVER makeup data, ALWAYS use the real files provided"
            ])

        prompt_parts.extend([
            "",
            "🚨 EXECUTION ENVIRONMENT (CRITICAL - READ CAREFULLY):",
            "- Code will be executed via subprocess WITHOUT command-line arguments",
            "- DO NOT use sys.argv - it will be empty (only script name)",
            "- DO NOT use input() - this is non-interactive execution",
            "- ALL filenames MUST be HARDCODED directly in the code",
            "- Files are in the current working directory - use filenames directly",
            "- If you create functions, call them with HARDCODED filenames in main code",
            "",
            "❌ FORBIDDEN PATTERNS:",
            "  if __name__ == '__main__':",
            "      import sys",
            "      if len(sys.argv) > 1:",
            "          main(sys.argv[1])  # ❌ WRONG - no arguments available!",
            "",
            "✅ CORRECT PATTERN:",
            "  if __name__ == '__main__':",
            "      main('complex_json.json')  # ✅ CORRECT - hardcoded filename",
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
                "   🚨 Replace 'EXACT_FILENAME_FROM_LIST.json' with the ACTUAL filename from the file list above!",
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
        # Normal mode
        prompt_parts = [
            "You are a Python code generator. Generate clean, efficient Python code to accomplish the following task:",
            "",
            f"Task: {query}",
            ""
        ]

        if context:
            prompt_parts.append(f"Context: {context}")
            prompt_parts.append("")

        prompt_parts.append(file_context)
        prompt_parts.append("")
        prompt_parts.append("Important requirements:")

        if file_context:
            prompt_parts.extend([
                "🚨 CRITICAL: Use the EXACT filenames shown in the file list above",
                "🚨 DO NOT use generic names like 'file.json', 'data.csv', 'input.xlsx', 'output.txt', etc.",
                "🚨 COPY the actual filename from the list - including ALL special characters, numbers, Korean text",
                "- Never add raw data to the code, always use the actual filenames to read the data",
                "- Always use the real data. NEVER makeup data and ask user to input data."
            ])

        prompt_parts.extend([
            "",
            "🚨 EXECUTION ENVIRONMENT (CRITICAL - READ CAREFULLY):",
            "- Code will be executed via subprocess WITHOUT command-line arguments",
            "- DO NOT use sys.argv - it will be empty (only script name)",
            "- DO NOT use input() - this is non-interactive execution",
            "- ALL filenames MUST be HARDCODED directly in the code",
            "- Files are in the current working directory - use filenames directly",
            "- If you create functions, call them with HARDCODED filenames in main code",
            "",
            "❌ FORBIDDEN PATTERNS:",
            "  if __name__ == '__main__':",
            "      import sys",
            "      if len(sys.argv) > 1:",
            "          main(sys.argv[1])  # ❌ WRONG - no arguments available!",
            "",
            "✅ CORRECT PATTERN:",
            "  if __name__ == '__main__':",
            "      filename = 'data.json'  # Use actual filename from file list",
            "      main(filename)  # ✅ CORRECT - hardcoded filename",
            "",
            "- Output results using print() statements",
            "- Include error handling (try/except)",
            "- Add a docstring explaining what the code does",
            "- Keep code clean and readable"
        ])

        if has_json_files:
            prompt_parts.extend([
                "",
                "JSON FILE REQUIREMENTS (STRICT - FOLLOW EXACTLY):",
                "1. File loading: with open('EXACT_FILENAME_FROM_LIST.json', 'r', encoding='utf-8') as f: data = json.load(f)",
                "   🚨 Replace 'EXACT_FILENAME_FROM_LIST.json' with the ACTUAL filename from the file list!",
                "   🚨 DO NOT use 'file.json', 'data.json', 'input.json' - use the REAL name!",
                "2. Error handling: Wrap in try/except json.JSONDecodeError",
                "3. Type validation: Check isinstance(data, dict) or isinstance(data, list) BEFORE accessing",
                "4. Safe dict access: ALWAYS use data.get('key', default) NEVER data['key']",
                "5. Key validation: ONLY use keys from \"📋 Access Patterns\" section - NO guessing or making up keys",
                "6. Nested access: Use chained .get(): data.get('parent', {}).get('child', default)",
                "7. Array safety: Check length before indexing: if len(data) > 0: item = data[0]",
                "8. Copy patterns: The \"📋 Access Patterns\" are pre-validated - copy them exactly",
                "9. Null handling: Check if value is not None before using",
                "10. Debugging: Print data structure first: print(\"Type:\", type(data), \"Keys:\", list(data.keys()) if isinstance(data, dict) else len(data))"
            ])

        prompt_parts.append("\nGenerate ONLY the Python code, no explanations or markdown:")
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
        "🚨 VERIFICATION MODE: Find problems that could cause execution failures or incorrect results.",
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
        "🔍 CRITICAL VERIFICATION CHECKLIST:",
        "",
        "1️⃣ LOGIC & CORRECTNESS:",
        "   - Does the code address the user's specific question?",
        "   - Will it produce the expected output?",
        "   - Are calculations/operations logically correct?",
        "",
        "2️⃣ SYNTAX & RUNTIME ERRORS:",
        "   - Any syntax errors (missing colons, parentheses, quotes)?",
        "   - Undefined variables or functions?",
        "   - Import statements correct?",
        "   - Blocked/dangerous modules (socket, subprocess, eval, exec)?",
        "",
        "3️⃣ ERROR HANDLING:",
        "   - Try/except blocks present where needed?",
        "   - File operations wrapped in error handling?",
        "   - Division by zero checks if applicable?",
        ""
    ])

    if file_context:
        prompt_parts.extend([
            "4️⃣ FILE HANDLING:",
            "   - Uses EXACT filenames from the file list?",
            "   - NO generic names like 'file.json', 'data.csv', 'input.xlsx'?",
            "   - File paths are strings, properly quoted?",
            "   - Uses ONLY real data (NO fake/placeholder data)?",
            "   - File reading has error handling (FileNotFoundError)?",
            ""
        ])

    if has_json_files:
        prompt_parts.extend([
            "5️⃣ JSON FILE HANDLING (CRITICAL):",
            "   - Uses EXACT JSON filename from file list (NOT 'file.json', 'data.json')?",
            "   - Has isinstance() check for data structure validation?",
            "   - Uses .get() for dict access (NEVER data['key'])?",
            "   - Checks for None/null values before nested access?",
            "   - ONLY uses keys from \"📋 Access Patterns\" (NO guessing keys)?",
            "   - Arrays checked with len() before indexing?",
            "   - Follows the \"📋 Access Patterns\" exactly?",
            "   - Has json.JSONDecodeError handling?",
            ""
        ])

    prompt_parts.extend([
        "6️⃣ EXECUTION COMPATIBILITY (CRITICAL - MUST CATCH THESE!):",
        "   - ❌ Does code use sys.argv? (FORBIDDEN - no command-line arguments available!)",
        "   - ❌ Does code check len(sys.argv) or access sys.argv[1], sys.argv[2], etc.?",
        "   - ❌ Does code use input() for user interaction? (FORBIDDEN - non-interactive)",
        "   - ❌ Does code print 'Usage:' messages expecting command-line args?",
        "   - ✅ Are ALL filenames HARDCODED in the code?",
        "   - ✅ If main() or other functions are called, are filenames passed as HARDCODED strings?",
        "   - ✅ Does code run standalone without any external input or arguments?",
        "",
        "   🚨 COMMON BAD PATTERNS TO FLAG:",
        "   - if len(sys.argv) > 1: ... (MUST FLAG THIS!)",
        "   - main(sys.argv[1]) (MUST FLAG THIS!)",
        "   - filename = sys.argv[1] if len(sys.argv) > 1 else 'default.json' (MUST FLAG THIS!)",
        "   - parser = argparse.ArgumentParser() (MUST FLAG THIS!)",
        "   - input('Enter filename:') (MUST FLAG THIS!)",
        "",
        "🚨 ERROR DETECTION PRIORITY:",
        "- Your primary goal is to find potential ERRORS (not style issues)",
        "- Focus on issues that will cause EXECUTION FAILURES or WRONG RESULTS",
        "- Be STRICT - even small issues can cause failures",
        "- If uncertain about filename correctness, mark it as an issue",
        "- ESPECIALLY check for sys.argv usage - this is the #1 failure cause",
        "",
        "📋 RESPONSE FORMAT:",
        'Return a JSON object: {"verified": true/false, "issues": ["issue1", "issue2", ...]}',
        "",
        '✅ Return {"verified": true, "issues": []} ONLY IF:',
        "   - Code is 100% correct and will execute without errors",
        "   - All filenames are exact matches from the file list",
        "   - All filenames are HARDCODED (no sys.argv, no input())",
        "   - All required safety checks are present",
        f"{'   - All JSON safety patterns are followed' if has_json_files else ''}",
        "",
        '❌ Return {"verified": false, "issues": [...]} IF:',
        "   - ANY potential error detected (syntax, runtime, logic)",
        "   - Uses sys.argv or input() (CRITICAL ERROR)",
        "   - Filenames don't match EXACTLY",
        "   - Filenames not hardcoded in the code",
        "   - Missing error handling",
        "   - Unsafe data access patterns",
        f"{'   - JSON access patterns not followed' if has_json_files else ''}",
        "",
        "⚠️  BE THOROUGH: It's better to flag a potential issue than miss a real error.",
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

Original request: {query}
{f"Context: {context}" if context else ""}

Current code:
```python
{code}
```

Execution error:
{error_message}

Analyze the error and fix the code. Output ONLY the corrected code, no explanations:"""
