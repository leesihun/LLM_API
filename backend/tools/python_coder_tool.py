"""
Python Coder Tool - Unified Implementation

Combines code generation, verification, and execution in a single module.
This tool orchestrates the full workflow: file preparation, code generation,
verification (focused on answering user's question), execution with retry logic.
"""

import ast
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from backend.config.settings import settings

logger = logging.getLogger(__name__)

# ============================================================================
# Constants and Configuration
# ============================================================================

BLOCKED_IMPORTS = [
    "socket", "subprocess", "os.system", "eval", "exec",
    "__import__", "importlib", "shutil.rmtree", "pickle",
]

SUPPORTED_FILE_TYPES = [
    ".txt", ".md", ".log", ".rtf",  # Text
    ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",  # Data
    ".xlsx", ".xls", ".xlsm", ".docx", ".doc",  # Office
    ".pdf",  # PDF
    ".dat", ".h5", ".hdf5", ".nc", ".parquet", ".feather",  # Scientific
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg",  # Images
    ".zip", ".tar", ".gz", ".bz2", ".7z",  # Compressed
]


# ============================================================================
# Code Executor (Low-level subprocess execution)
# ============================================================================

class CodeExecutor:
    """
    Executes Python code in isolated subprocess with security restrictions.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 512,
        execution_base_dir: str = "./data/scratch"
    ):
        """
        Initialize code executor.

        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            execution_base_dir: Base directory for temporary execution folders
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.execution_base_dir = Path(execution_base_dir).resolve()
        self.execution_base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[CodeExecutor] Initialized with timeout={timeout}s, max_memory={max_memory_mb}MB")

    def validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate that code only imports safe packages.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module in BLOCKED_IMPORTS:
                        issues.append(f"Blocked import detected: {module}")

            elif isinstance(node, ast.ImportFrom):
                module = node.module.split('.')[0] if node.module else ''
                if module in BLOCKED_IMPORTS:
                    issues.append(f"Blocked import detected: {module}")

            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', '__import__']:
                        issues.append(f"Dangerous function call detected: {node.func.id}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def execute(
        self,
        code: str,
        input_files: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code in isolated subprocess.

        Args:
            code: Python code to execute
            input_files: Optional dict mapping original file paths to their basenames
            session_id: Optional session ID to use as execution directory name

        Returns:
            Dict with keys: success, output, error, execution_time, return_code
        """
        # Use session_id if provided, otherwise generate unique ID
        execution_id = session_id if session_id else uuid.uuid4().hex
        execution_dir = self.execution_base_dir / execution_id

        try:
            # Create execution directory
            execution_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[CodeExecutor] Using execution directory: {execution_dir}")

            # Copy input files to execution directory
            if input_files:
                for original_path, basename in input_files.items():
                    target_path = execution_dir / basename
                    shutil.copy2(original_path, target_path)
                    logger.debug(f"[CodeExecutor] Copied {original_path} -> {target_path}")

            # Write code to script.py
            script_path = execution_dir / "script.py"
            script_path.write_text(code, encoding='utf-8')
            logger.debug(f"[CodeExecutor] Wrote code to {script_path}")

            # Execute code
            start_time = time.time()
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                timeout=self.timeout,
                cwd=str(execution_dir),
                text=True
            )
            execution_time = time.time() - start_time

            logger.info(f"[CodeExecutor] Execution completed in {execution_time:.2f}s (return code: {result.returncode})")

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "execution_time": execution_time,
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            logger.error(f"[CodeExecutor] Execution timeout after {self.timeout}s")
            return {
                "success": False,
                "output": "",
                "error": f"Execution timeout after {self.timeout} seconds",
                "execution_time": self.timeout,
                "return_code": -1
            }

        except Exception as e:
            logger.error(f"[CodeExecutor] Execution failed: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": 0,
                "return_code": -1
            }

        finally:
            # Cleanup execution directory (only if temporary, not session-based)
            if not session_id:
                try:
                    if execution_dir.exists():
                        shutil.rmtree(execution_dir)
                        logger.debug(f"[CodeExecutor] Cleaned up temporary directory {execution_dir}")
                except Exception as e:
                    logger.warning(f"[CodeExecutor] Failed to cleanup {execution_dir}: {e}")
            else:
                logger.debug(f"[CodeExecutor] Keeping session directory {execution_dir}")

    def validate_file_type(self, file_path: str) -> bool:
        """
        Check if file type is supported.

        Args:
            file_path: Path to file

        Returns:
            True if file type is supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in SUPPORTED_FILE_TYPES


# ============================================================================
# Main Python Coder Tool
# ============================================================================

class PythonCoderTool:
    """
    Python code generator tool with iterative verification and execution with retry logic.
    
    Features:
    - Code generation using LLM
    - Verification focused on answering user's question (max 3 iterations)
    - Code execution with retry on failure (max 5 attempts)
    - File handling and metadata extraction
    """

    def __init__(self):
        """Initialize the Python coder tool."""
        self.llm = ChatOllama(
            base_url=settings.ollama_host,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            num_ctx=settings.ollama_num_ctx,
            top_p=settings.ollama_top_p,
            top_k=settings.ollama_top_k
        )
        self.executor = CodeExecutor(
            timeout=settings.python_code_timeout,
            max_memory_mb=settings.python_code_max_memory,
            execution_base_dir=settings.python_code_execution_dir
        )
        # Verifier max iterations: 3
        self.max_verification_iterations = 3
        # Execution retry max attempts: 5
        self.max_execution_attempts = 5
        self.allow_partial_execution = settings.python_code_allow_partial_execution

        logger.info(f"[PythonCoderTool] Initialized with verification_iterations={self.max_verification_iterations}, execution_attempts={self.max_execution_attempts}")

    async def execute_code_task(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for code generation and execution.

        Args:
            query: User's question/task
            context: Optional additional context
            file_paths: Optional list of input file paths
            session_id: Optional session ID

        Returns:
            Dict with execution results
        """
        if not settings.python_code_enabled:
            return {
                "success": False,
                "error": "Python code execution is disabled in settings",
                "code": "",
                "output": ""
            }

        logger.info(f"[PythonCoderTool] Executing task: {query[:100]}...")

        # Phase 0: Prepare input files
        validated_files = {}
        file_metadata = {}
        if file_paths:
            validated_files, file_metadata = self._prepare_files(file_paths)
            if not validated_files and file_paths:
                return {
                    "success": False,
                    "error": "Failed to validate input files",
                    "code": "",
                    "output": ""
                }

        # Phase 1: Generate initial code
        code = await self._generate_code(query, context, validated_files, file_metadata)
        if not code:
            return {
                "success": False,
                "error": "Failed to generate code",
                "code": "",
                "output": ""
            }

        # Phase 2: Iterative verification (max 3 iterations)
        # Focus: Does the code answer the user's question?
        verification_history = []
        modifications = []

        for iteration in range(self.max_verification_iterations):
            logger.info(f"[PythonCoderTool] Verification iteration {iteration + 1}/{self.max_verification_iterations}")

            # Verify code (focused on answering user's question)
            verified, issues = await self._verify_code_answers_question(code, query)

            verification_history.append({
                "iteration": iteration + 1,
                "issues": issues,
                "action": "approved" if verified else "needs_modification"
            })

            if verified:
                logger.info(f"[PythonCoderTool] Code verified successfully at iteration {iteration + 1}")
                break

            # Modify code
            logger.info(f"[PythonCoderTool] Modifying code due to issues: {issues}")
            code, changes = await self._modify_code(code, issues, query, context)
            modifications.extend(changes)

            verification_history[-1]["action"] = "modified"

            # Check if we've reached max iterations
            if iteration == self.max_verification_iterations - 1:
                if self.allow_partial_execution:
                    logger.warning("[PythonCoderTool] Max verification iterations reached, proceeding with execution")
                    break
                else:
                    logger.warning("[PythonCoderTool] Max verification iterations reached but partial execution not allowed")
                    # Continue to execution anyway, let retry handle failures

        # Phase 3: Execute code with retry logic (max 5 attempts)
        execution_result = None
        execution_attempts = []

        for attempt in range(self.max_execution_attempts):
            logger.info(f"[PythonCoderTool] Execution attempt {attempt + 1}/{self.max_execution_attempts}")

            execution_result = self.executor.execute(code, validated_files, session_id=session_id)
            execution_attempts.append({
                "attempt": attempt + 1,
                "success": execution_result["success"],
                "error": execution_result.get("error"),
                "execution_time": execution_result["execution_time"]
            })

            if execution_result["success"]:
                logger.info(f"[PythonCoderTool] Execution succeeded on attempt {attempt + 1}")
                break

            # If failed and not last attempt, try to fix the code
            if attempt < self.max_execution_attempts - 1:
                logger.warning(f"[PythonCoderTool] Execution failed on attempt {attempt + 1}, attempting to fix code")
                error_message = execution_result.get("error", "Unknown error")
                code, fix_changes = await self._fix_execution_error(code, query, error_message)
                modifications.extend(fix_changes)
            else:
                logger.error(f"[PythonCoderTool] Execution failed after {self.max_execution_attempts} attempts")

        # Phase 4: Format and return result
        return {
            "success": execution_result["success"],
            "code": code,
            "output": execution_result["output"],
            "error": execution_result.get("error"),
            "execution_time": execution_result["execution_time"],
            "verification_iterations": len(verification_history),
            "execution_attempts": len(execution_attempts),
            "modifications": modifications,
            "input_files": list(validated_files.keys()),
            "file_metadata": file_metadata,
            "verification_history": verification_history,
            "execution_attempts_history": execution_attempts
        }

    def _prepare_files(
        self,
        file_paths: List[str]
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Prepare and validate input files.

        Args:
            file_paths: List of file paths

        Returns:
            Tuple of (validated_files, file_metadata)
        """
        validated_files = {}
        file_metadata = {}

        for file_path in file_paths:
            path = Path(file_path)

            # Validate existence
            if not path.exists():
                logger.error(f"[PythonCoderTool] File not found: {file_path}")
                continue

            # Validate size
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > settings.python_code_max_file_size:
                logger.error(f"[PythonCoderTool] File too large: {file_path} ({size_mb:.2f}MB)")
                continue

            # Validate type
            if not self.executor.validate_file_type(file_path):
                logger.error(f"[PythonCoderTool] Unsupported file type: {file_path}")
                continue

            # Extract ORIGINAL filename (remove temp_ prefix if present)
            original_filename = self._get_original_filename(path.name)

            # Extract metadata
            metadata = self._extract_file_metadata(path)
            metadata['original_filename'] = original_filename  # Add to metadata
            file_metadata[str(path)] = metadata

            # Store mapping (temp path -> ORIGINAL filename for execution)
            validated_files[str(path)] = original_filename

            logger.info(f"[PythonCoderTool] Validated file: {original_filename} (temp: {path.name}, {size_mb:.2f}MB)")

        return validated_files, file_metadata

    def _extract_file_metadata(self, path: Path) -> Dict[str, Any]:
        """
        Extract metadata from file for better code generation.

        Args:
            path: Path to file

        Returns:
            Dict with metadata
        """
        metadata = {
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2)
        }

        try:
            # CSV/TSV files
            if path.suffix.lower() in ['.csv', '.tsv']:
                try:
                    import pandas as pd
                    df = pd.read_csv(path, nrows=5)
                    metadata.update({
                        "type": "csv",
                        "columns": df.columns.tolist(),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "sample_rows": len(df)
                    })
                except Exception as e:
                    logger.warning(f"[PythonCoderTool] Could not extract CSV metadata: {e}")

            # Excel files
            elif path.suffix.lower() in ['.xlsx', '.xls', '.xlsm']:
                try:
                    import pandas as pd
                    df = pd.read_excel(path, nrows=5)
                    metadata.update({
                        "type": "excel",
                        "columns": df.columns.tolist(),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "sample_rows": len(df)
                    })
                except Exception as e:
                    logger.warning(f"[PythonCoderTool] Could not extract Excel metadata: {e}")

            # JSON files
            elif path.suffix.lower() == '.json':
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    metadata.update({
                        "type": "json",
                        "structure": "array" if isinstance(data, list) else "object",
                        "item_count": len(data) if isinstance(data, (list, dict)) else 1
                    })
                except Exception as e:
                    logger.warning(f"[PythonCoderTool] Could not extract JSON metadata: {e}")

            # Text files
            elif path.suffix.lower() in ['.txt', '.md', '.log', '.rtf']:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    metadata.update({
                        "type": "text",
                        "line_count": len(lines),
                        "preview": ''.join(lines[:5])[:200]
                    })
                except Exception as e:
                    logger.warning(f"[PythonCoderTool] Could not extract text metadata: {e}")

        except Exception as e:
            logger.error(f"[PythonCoderTool] Error extracting metadata for {path}: {e}")

        return metadata

    def _get_original_filename(self, temp_filename: str) -> str:
        """
        Extract original filename from temp filename.

        Temp files are named: temp_XXXXXXXX_originalname.ext
        This method strips the temp_ prefix to get: originalname.ext

        Args:
            temp_filename: Filename possibly with temp_ prefix

        Returns:
            Original filename without temp_ prefix
        """
        if temp_filename.startswith('temp_'):
            # Split on underscore: ['temp', 'XXXXXXXX', 'originalname.ext']
            parts = temp_filename.split('_', 2)
            if len(parts) >= 3:
                return parts[2]  # Return the original filename

        # If no temp_ prefix, return as-is
        return temp_filename

    async def _generate_code(
        self,
        query: str,
        context: Optional[str],
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any]
    ) -> str:
        """
        Generate Python code using LLM.

        Args:
            query: User's question
            context: Optional additional context
            validated_files: Dict of validated files
            file_metadata: Metadata for files

        Returns:
            Generated code
        """
        # Build file context
        file_context = ""
        if validated_files:
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

        prompt = f"""You are a Python code generator. Generate clean, efficient Python code to accomplish the following task:

Task: {query}

{f"Context: {context}" if context else ""}
{file_context}

Important requirements:
1. Use the EXACT filenames shown above (they are in the current directory)
2. Output results using print() statements
3. Include error handling (try/except)
4. Add a docstring explaining what the code does
5. Keep code clean and readable

Generate ONLY the Python code, no explanations or markdown:"""

        try:
            logger.info(f"\n\n\[PythonCoderTool] Generating code with prompt: {prompt[:]}...\n\n")
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            # Extract code from response (remove markdown if present)
            code = response.content.strip()
            if code.startswith("```python"):
                code = code.split("```python")[1]
                code = code.split("```")[0]
            elif code.startswith("```"):
                code = code.split("```")[1]
                code = code.split("```")[0]

            code = code.strip()
            logger.info(f"[PythonCoderTool] Generated code ({len(code)} chars)")
            return code

        except Exception as e:
            logger.error(f"[PythonCoderTool] Failed to generate code: {e}")
            return ""

    async def _verify_code_answers_question(self, code: str, query: str) -> Tuple[bool, List[str]]:
        """
        Verify code focuses on answering user's question.
        Simplified verification focused on the core goal.

        Args:
            code: Python code to verify
            query: Original user query

        Returns:
            Tuple of (is_verified, list of issues)
        """
        issues = []

        # Static analysis checks (safety)
        is_valid, static_issues = self.executor.validate_imports(code)
        if not is_valid:
            issues.extend(static_issues)

        # LLM-based semantic check: Does it answer the question?
        semantic_issues = await self._llm_verify_answers_question(code, query)
        issues.extend(semantic_issues)

        is_verified = len(issues) == 0
        return is_verified, issues

    async def _llm_verify_answers_question(self, code: str, query: str) -> List[str]:
        """
        Use LLM to verify if code answers the user's question.
        Simplified verification focused on core requirements.

        Args:
            code: Python code to verify
            query: Original user query

        Returns:
            List of issues found
        """
        prompt = f"""Review this Python code and determine if it correctly answers the user's question.

User Question: {query}

Code:
```python
{code}
```

Check ONLY these critical points:
1. Does the code address the user's specific question?
2. Will the code produce output that answers the question (using print statements)?
3. Are there any obvious syntax errors?
4. Are any imports from blocked/dangerous modules?

Respond with a JSON object:
{{"verified": true/false, "issues": ["issue1", "issue2", ...]}}

If code correctly answers the question, return {{"verified": true, "issues": []}}
Only report issues that prevent answering the user's question."""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            # Try to parse JSON response
            response_text = response.content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            result = json.loads(response_text.strip())
            return result.get("issues", [])

        except Exception as e:
            logger.warning(f"[PythonCoderTool] LLM verification failed: {e}")
            return []  # Don't block on LLM verification failure

    async def _modify_code(
        self,
        code: str,
        issues: List[str],
        query: str,
        context: Optional[str]
    ) -> Tuple[str, List[str]]:
        """
        Modify code to fix issues.

        Args:
            code: Current Python code
            issues: List of issues to fix
            query: Original user query
            context: Optional additional context

        Returns:
            Tuple of (modified_code, list of changes made)
        """
        prompt = f"""Fix the following Python code to address these issues:

Original request: {query}
{f"Context: {context}" if context else ""}

Current code:
```python
{code}
```

Issues to fix:
{chr(10).join(f"- {issue}" for issue in issues)}

Generate the corrected Python code. Output ONLY the code, no explanations:"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            # Extract code
            modified_code = response.content.strip()
            if modified_code.startswith("```python"):
                modified_code = modified_code.split("```python")[1].split("```")[0]
            elif modified_code.startswith("```"):
                modified_code = modified_code.split("```")[1].split("```")[0]

            modified_code = modified_code.strip()

            changes = [f"Fixed: {issue}" for issue in issues]
            logger.info(f"[PythonCoderTool] Modified code ({len(changes)} changes)")

            return modified_code, changes

        except Exception as e:
            logger.error(f"[PythonCoderTool] Failed to modify code: {e}")
            return code, []  # Return original code if modification fails

    async def _fix_execution_error(
        self,
        code: str,
        query: str,
        error_message: str
    ) -> Tuple[str, List[str]]:
        """
        Fix code based on execution error.

        Args:
            code: Current Python code
            query: Original user query
            error_message: Error from execution

        Returns:
            Tuple of (fixed_code, list of changes made)
        """
        prompt = f"""Fix the following Python code that failed during execution:

Original request: {query}

Current code:
```python
{code}
```

Execution error:
{error_message}

Analyze the error and fix the code. Output ONLY the corrected code, no explanations:"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            # Extract code
            fixed_code = response.content.strip()
            if fixed_code.startswith("```python"):
                fixed_code = fixed_code.split("```python")[1].split("```")[0]
            elif fixed_code.startswith("```"):
                fixed_code = fixed_code.split("```")[1].split("```")[0]

            fixed_code = fixed_code.strip()

            changes = [f"Fixed execution error: {error_message[:100]}"]
            logger.info(f"[PythonCoderTool] Fixed code after execution error")

            return fixed_code, changes

        except Exception as e:
            logger.error(f"[PythonCoderTool] Failed to fix execution error: {e}")
            return code, []  # Return original code if fix fails


# ============================================================================
# Global Instance
# ============================================================================

python_coder_tool = PythonCoderTool()


# ============================================================================
# Backward Compatibility Exports
# ============================================================================

# For backward compatibility with existing imports
PythonExecutor = CodeExecutor
SUPPORTED_FILE_TYPES = SUPPORTED_FILE_TYPES
