"""
Python Coder Tool

Main entry point for AI-driven Python code generation with iterative verification and modification.
This tool orchestrates the full workflow: file preparation, code generation, verification,
modification loops, execution, and result formatting.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from backend.config.settings import settings
from backend.tools.python_executor_engine import PythonExecutor, SUPPORTED_FILE_TYPES

logger = logging.getLogger(__name__)


class PythonCoderTool:
    """
    Python code generator tool with iterative verification and modification.
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
        self.executor = PythonExecutor(
            timeout=settings.python_code_timeout,
            max_memory_mb=settings.python_code_max_memory,
            execution_base_dir=settings.python_code_execution_dir
        )
        self.max_iterations = settings.python_code_max_iterations
        self.allow_partial_execution = settings.python_code_allow_partial_execution

        logger.info("[PythonCoderTool] Initialized with max_iterations=%d", self.max_iterations)

    async def execute_code_task(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:


        if not settings.python_code_enabled:
            return {
                "success": False,
                "error": "Python code execution is disabled in settings",
                "code": "",
                "output": ""
            }

        logger.info(f"[PythonCoderTool] Executing task: {query[:]}...")

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

        # Phase 2: Iterative verification and modification
        verification_history = []
        modifications = []

        for iteration in range(self.max_iterations):
            logger.info(f"[PythonCoderTool] Verification iteration {iteration + 1}/{self.max_iterations}")

            # Verify code
            verified, issues = await self._verify_code(code, query)

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
            if iteration == self.max_iterations - 1:
                if self.allow_partial_execution and self._only_minor_issues(issues):
                    logger.warning("[PythonCoderTool] Max iterations reached, executing with minor issues")
                    break
                else:
                    return {
                        "success": False,
                        "error": f"Failed verification after {self.max_iterations} iterations",
                        "issues": issues,
                        "code": code,
                        "output": "",
                        "iterations": iteration + 1,
                        "modifications": modifications,
                        "verification_history": verification_history
                    }

        # Phase 3: Execute code
        execution_result = self.executor.execute_code(code, validated_files)

        # Phase 4: Format and return result
        return {
            "success": execution_result["success"],
            "code": code,
            "output": execution_result["output"],
            "error": execution_result.get("error"),
            "execution_time": execution_result["execution_time"],
            "iterations": len(verification_history),
            "modifications": modifications,
            "input_files": list(validated_files.keys()),
            "file_metadata": file_metadata,
            "verification_history": verification_history
        }

    def _prepare_files(
        self,
        file_paths: List[str]
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:


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

            # Extract metadata
            metadata = self._extract_file_metadata(path)
            file_metadata[str(path)] = metadata

            # Store mapping (original path -> basename for execution)
            validated_files[str(path)] = path.name

            logger.info(f"[PythonCoderTool] Validated file: {path.name} ({size_mb:.2f}MB)")

        return validated_files, file_metadata

    def _extract_file_metadata(self, path: Path) -> Dict[str, Any]:
        

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
                        "preview": ''.join(lines[:])[:]
                    })
                except Exception as e:
                    logger.warning(f"[PythonCoderTool] Could not extract text metadata: {e}")

        except Exception as e:
            logger.error(f"[PythonCoderTool] Error extracting metadata for {path}: {e}")

        return metadata

    async def _generate_code(
        self,
        query: str,
        context: Optional[str],
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any]
    ) -> str:
    

        # Build file context
        file_context = ""
        if validated_files:
            file_context = "\n\nAvailable files:\n"
            for original_path, basename in validated_files.items():
                metadata = file_metadata.get(original_path, {})
                file_context += f"- {basename}: {metadata.get('type', 'unknown')} ({metadata.get('size_mb', 0)}MB)\n"
                if 'columns' in metadata:
                    file_context += f"  Columns: {', '.join(metadata['columns'][:])}\n"
                if 'preview' in metadata:
                    file_context += f"  Preview: {metadata['preview'][:]}...\n"

        prompt = f"""You are a Python code generator. Generate clean, efficient Python code to accomplish the following task:

Task: {query}

{f"Context: {context}" if context else ""}
{file_context}

Important requirements:
1. If files are provided, access them by their full path
2. Output results by printing JSON to stdout
3. Include error handling
4. Add docstring explaining what the code does
5. Keep code clean and readabl

Generate ONLY the Python code, no explanations or markdown:"""

        try:
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
            logger.info(f"\n\n\n[PythonCoderTool] Generated code: {code[:]}\n\n\n")
            return code

        except Exception as e:
            logger.error(f"[PythonCoderTool] Failed to generate code: {e}")
            return ""

    async def _verify_code(self, code: str, query: str) -> Tuple[bool, List[str]]:
        """
        Verify code for safety and correctness.

        Args:
            code: Python code to verify
            query: Original user query

        Returns:
            Tuple of (is_verified, list of issues)
        """
        issues = []

        # Static analysis checks
        is_valid, static_issues = self.executor.validate_imports(code)
        if not is_valid:
            issues.extend(static_issues)

        # LLM-based semantic checks
        semantic_issues = await self._llm_verify_code(code, query)
        issues.extend(semantic_issues)

        is_verified = len(issues) == 0
        return is_verified, issues

    async def _llm_verify_code(self, code: str, query: str) -> List[str]:
        """
        Use LLM to verify code semantically.

        Args:
            code: Python code to verify
            query: Original user query

        Returns:
            List of issues found
        """
        prompt = f"""Review this Python code for potential issues:

Original request: {query}

Code:
```python
{code}
```

Check for:
CORRECTNESS & INTENT ALIGNMENT
Does the code accomplish the user's stated goal?
Are all required inputs handled (files, parameters)?
Does the logic flow match the requested behavior?
Are edge cases addressed (empty data, null values, missing files)?
RUNTIME ERROR PREVENTION
Syntax validation (proper indentation, valid Python)
Import availability (are all imports from whitelisted packages?)
Error handling (try-except blocks around risky operations)
Type safety (appropriate type checks before operations)
Division by zero protection
Index out of bounds protection
Null/None checks before attribute access
OUTPUT FORMAT COMPLIANCE
Does code output JSON to stdout as required?
Is the JSON structure valid and parseable?
Are error states properly communicated in output?
Does it use print() for output (not return statements)?
PERFORMANCE & EFFICIENCY
Reasonable algorithmic complexity (no O(n³) or worse for large data)
Efficient data structure usage
Avoid unnecessary loops or redundant operations
Memory-conscious operations (streaming for large files)
CODE QUALITY
Readable variable names
Proper docstring present and accurate
Logical structure and flow
No dead code or unused imports
Follows Python conventions (PEP 8 style)
FILE HANDLING (if applicable)
Uses correct file access methods for file types
Proper encoding specified (utf-8)
File handles properly closed (use context managers)
Validates file existence before reading

Respond with a JSON object:
{{"verified": true/false, "issues": ["issue1", "issue2", ...]}}

If code is good, return {{"verified": true, "issues": []}}"""

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

    def _only_minor_issues(self, issues: List[str]) -> bool:
        """
        Check if issues are minor enough to allow execution.

        Args:
            issues: List of issues

        Returns:
            True if all issues are minor
        """
        # Define major issue keywords
        major_keywords = ['security', 'unsafe', 'blocked', 'dangerous', 'syntax error']

        for issue in issues:
            issue_lower = issue.lower()
            if any(keyword in issue_lower for keyword in major_keywords):
                return False

        return True


# Global instance
python_coder_tool = PythonCoderTool()
