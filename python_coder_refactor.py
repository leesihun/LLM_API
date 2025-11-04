# This is a reference file showing the changes needed to python_coder_tool.py
# The key changes are:
# 1. Extract file_context building into a separate method
# 2. Pass file_context to the verifier
# 3. Build file_context once and reuse it

# Add this new method to PythonCoderTool class (around line 540):

def _build_file_context(self, validated_files: Dict[str, str], file_metadata: Dict[str, Any]) -> str:
    """
    Build file context string from validated files and metadata.

    Args:
        validated_files: Dict mapping file paths to filenames
        file_metadata: Dict of file metadata

    Returns:
        Formatted file context string
    """
    if not validated_files:
        return ""

    file_context = """

IMPORTANT - FILE ACCESS:
All files are in the current working directory. Use the exact filenames shown below.

Available files:
"""

    for idx, (original_path, original_filename) in enumerate(validated_files.items(), 1):
        metadata = file_metadata.get(original_path, {})
        file_type = metadata.get('type', 'unknown')

        file_context += f"\n{idx}. \"{original_filename}\" - {file_type.upper()} ({metadata.get('size_mb', 0)}MB)\n"

        # Add relevant metadata
        if 'columns' in metadata:
            cols = metadata['columns'][:10]
            file_context += f"   Columns: {', '.join(cols)}"
            if len(metadata.get('columns', [])) > 10:
                file_context += f" ... (+{len(metadata['columns']) - 10} more)"
            file_context += "\n"

        if 'structure' in metadata:
            file_context += f"   Structure: {metadata['structure']} ({metadata.get('item_count', 0)} items)\n"

        if 'line_count' in metadata:
            file_context += f"   Lines: {metadata['line_count']}\n"

        if 'preview' in metadata:
            preview = metadata['preview'][:100]
            file_context += f"   Preview: {preview}...\n"

        # File access example
        if file_type == 'csv':
            file_context += f"   Example: df = pd.read_csv('{original_filename}')\n"
        elif file_type == 'json':
            file_context += f"   Example: data = json.load(open('{original_filename}'))\n"
        elif file_type == 'excel':
            file_context += f"   Example: df = pd.read_excel('{original_filename}')\n"

    file_context += "\n"
    return file_context


# Update execute_code_task method (around line 302-321):
# BEFORE:
# code = await self._generate_code(query, context, validated_files, file_metadata)
# ...
# verified, issues = await self._verify_code_answers_question(code, query)

# AFTER:
# Build file context once (reuse for generation, verification, and modification)
file_context = self._build_file_context(validated_files, file_metadata)

# Phase 1: Generate initial code
code = await self._generate_code(query, context, file_context)

# Phase 2: Iterative verification (pass file_context to verifier)
verified, issues = await self._verify_code_answers_question(code, query, file_context)

# When modifying code (line 335):
code, changes = await self._modify_code(code, issues, query, context, file_context)


# Update _generate_code signature (line 543):
# BEFORE:
# async def _generate_code(self, query: str, context: Optional[str], validated_files: Dict[str, str], file_metadata: Dict[str, Any]) -> str:

# AFTER:
async def _generate_code(self, query: str, context: Optional[str], file_context: str) -> str:
    """
    Generate Python code using LLM.

    Args:
        query: User's question
        context: Optional additional context
        file_context: Pre-built file context string

    Returns:
        Generated code
    """
    # Remove the file_context building code (lines 562-605) since it's now passed in

    prompt = f"""You are a Python code generator. Generate clean, efficient Python code to accomplish the following task:

Task: {query}

{f"Context: {context}" if context else ""}
{file_context}

Important requirements:
- Never add raw data to the code, always use the actual filenames to read the data
- Use the EXACT filenames shown above (they are in the current directory)
- Output results using print() statements
- Include error handling (try/except)
- Add a docstring explaining what the code does
- Keep code clean and readable
- Always use the real data. NEVER makeup data and ask user to input data.

Generate ONLY the Python code, no explanations or markdown:"""

    # Rest of the method stays the same...


# Update _verify_code_answers_question signature (line 687):
# BEFORE:
# async def _verify_code_answers_question(self, code: str, query: str) -> Tuple[bool, List[str]]:

# AFTER:
async def _verify_code_answers_question(self, code: str, query: str, file_context: str = "") -> Tuple[bool, List[str]]:
    # Pass file_context to LLM verifier
    semantic_issues = await self._llm_verify_answers_question(code, query, file_context)


# Update _llm_verify_answers_question signature (line 702):
# BEFORE:
# async def _llm_verify_answers_question(self, code: str, query: str) -> List[str]:

# AFTER:
async def _llm_verify_answers_question(self, code: str, query: str, file_context: str = "") -> List[str]:
    """
    Use LLM to verify if code answers the user's question.

    Args:
        code: Python code to verify
        query: Original user query
        file_context: File context with available files and metadata

    Returns:
        List of issues found
    """
    prompt = f"""Review this Python code and determine if it correctly answers the user's question.

User Question: {query}

{file_context}

Code:
```python
{code}
```

Check ONLY these critical points:
1. Does the code address the user's specific question?
2. Will the code produce output that answers the question (using print statements)?
3. Are there any obvious syntax errors?
4. Are any imports from blocked/dangerous modules?
5. Does the code use the correct filenames shown above (if files are provided)?
6. Does the code use ONLY the real data? (NO fake data, NO user input, NO make up data, NO placeholder data)

However, it is OK to read data from different filenames to read the data as the provided file names may be different.

Respond with a JSON object:
{{"verified": true/false, "issues": ["issue1", "issue2", ...]}}

If code correctly answers the question, return {{"verified": true, "issues": []}}
Only report issues that prevent answering the user's question."""

    # Rest stays the same...


# Update _modify_code signature (line 742):
# BEFORE:
# async def _modify_code(self, code: str, issues: List[str], query: str, context: Optional[str]) -> Tuple[str, List[str]]:

# AFTER:
async def _modify_code(self, code: str, issues: List[str], query: str, context: Optional[str], file_context: str = "") -> Tuple[str, List[str]]:
    """
    Modify code to fix issues.

    Args:
        code: Current Python code
        issues: List of issues to fix
        query: Original user query
        context: Optional additional context
        file_context: File context with available files

    Returns:
        Tuple of (modified_code, list of changes made)
    """
    prompt = f"""Fix the following Python code to address these issues:

Original request: {query}
{f"Context: {context}" if context else ""}
{file_context}

Current code:
```python
{code}
```

Issues to fix:
{chr(10).join(f"- {issue}" for issue in issues)}

Generate the corrected Python code. Output ONLY the code, no explanations:"""

    # Rest stays the same...
