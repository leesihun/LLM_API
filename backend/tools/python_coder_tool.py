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

            # Enhanced execution result logging
            logger.info("=" * 80)
            logger.info("🔥 [CODE EXECUTION RESULT] 🔥")
            logger.info("=" * 80)
            logger.info(f"Status: {'✅ SUCCESS' if result.returncode == 0 else '❌ FAILED'}")
            logger.info(f"Execution Time: {execution_time:.2f}s")
            logger.info(f"Return Code: {result.returncode}")
            logger.info("=" * 80)

            # Log stdout with clear visual indicators
            if result.stdout:
                logger.info("📤 [STDOUT OUTPUT]:")
                logger.info("-" * 80)
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"  {line}")
                logger.info("-" * 80)
            else:
                logger.info("📤 [STDOUT OUTPUT]: (empty)")

            # Log stderr with clear error indicators
            if result.stderr:
                if result.returncode != 0:
                    logger.error("❌ [STDERR - ERROR]:")
                else:
                    logger.warning("⚠️  [STDERR - WARNING]:")
                logger.info("-" * 80)
                for line in result.stderr.strip().split('\n'):
                    if result.returncode != 0:
                        logger.error(f"  {line}")
                    else:
                        logger.warning(f"  {line}")
                logger.info("-" * 80)
            else:
                logger.info("📤 [STDERR]: (empty)")

            logger.info("=" * 80)

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
        self.max_verification_iterations = 5
        # Execution retry max attempts: 5
        self.max_execution_attempts = 5
        self.allow_partial_execution = settings.python_code_allow_partial_execution

        logger.info(f"[PythonCoderTool] Initialized with verification_iterations={self.max_verification_iterations}, execution_attempts={self.max_execution_attempts}")

    async def execute_code_task(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        is_prestep: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point for code generation and execution.

        Args:
            query: User's question/task
            context: Optional additional context
            file_paths: Optional list of input file paths
            session_id: Optional session ID
            is_prestep: Whether this is called from ReAct pre-step (uses specialized prompt)

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
            logger.info(f"[PythonCoderTool] Preparing {len(file_paths)} file(s)...")
            validated_files, file_metadata = self._prepare_files(file_paths)

            if not validated_files and file_paths:
                logger.error("[PythonCoderTool] Failed to validate any input files!")
                return {
                    "success": False,
                    "error": "Failed to validate input files",
                    "code": "",
                    "output": ""
                }

            # Log file metadata summary
            logger.info(f"[PythonCoderTool] Validated {len(validated_files)} file(s):")
            for file_path, filename in validated_files.items():
                metadata = file_metadata.get(file_path, {})
                logger.info(f"  - '{filename}' ({metadata.get('type', 'unknown')})")
                if metadata.get('type') == 'json':
                    logger.info(f"    Structure: {metadata.get('structure')}, Keys: {len(metadata.get('keys', []))}, Depth: {metadata.get('max_depth')}")

        # Build file context (reused in both generation and verification)
        file_context = self._build_file_context(validated_files, file_metadata)

        if file_context:
            logger.info("[PythonCoderTool] File context built successfully")
            logger.debug(f"[PythonCoderTool] File context preview:\n{file_context[:500]}...")

        # Phase 1: Generate initial code
        code = await self._generate_code(query, context, validated_files, file_metadata, is_prestep=is_prestep)
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
            verified, issues = await self._verify_code_answers_question(code, query, context, file_context, file_metadata)

            verification_history.append({
                "iteration": iteration + 1,
                "issues": issues,
                "action": "approved" if verified else "needs_modification"
            })

            if verified:
                logger.info(f"[PythonCoderTool] ✅ Code verified successfully at iteration {iteration + 1}")
                break

            # Modify code
            logger.warning(f"[PythonCoderTool] ⚠️  Verification issues found ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                logger.warning(f"  {i}. {issue}")
            logger.info(f"[PythonCoderTool] Modifying code to address issues...")
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
                logger.info(f"[PythonCoderTool] ✅ Execution succeeded on attempt {attempt + 1}")
                logger.info(f"[PythonCoderTool] Output length: {len(execution_result['output'])} characters")
                break

            # If failed and not last attempt, try to fix the code
            if attempt < self.max_execution_attempts - 1:
                error_message = execution_result.get("error", "Unknown error")
                logger.error(f"[PythonCoderTool] ❌ Execution failed on attempt {attempt + 1}")
                logger.error(f"[PythonCoderTool] Error: {error_message[:200]}...")
                logger.info(f"[PythonCoderTool] Attempting to fix code...")
                # Pass context to help fix execution errors
                code, fix_changes = await self._fix_execution_error(code, query, error_message, context)
                modifications.extend(fix_changes)
            else:
                logger.error(f"[PythonCoderTool] ❌ Execution failed after {self.max_execution_attempts} attempts")
                logger.error(f"[PythonCoderTool] Final error: {execution_result.get('error', 'Unknown')}")

        # Phase 4: Format and return result
        result = {
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

        # Final summary log
        logger.info("=" * 80)
        logger.info("[PythonCoderTool] EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"  Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
        logger.info(f"  Verification iterations: {result['verification_iterations']}")
        logger.info(f"  Execution attempts: {result['execution_attempts']}")
        logger.info(f"  Total modifications: {len(modifications)}")
        logger.info(f"  Execution time: {result['execution_time']:.2f}s")
        if result['success']:
            logger.info(f"  Output length: {len(result['output'])} chars")
        else:
            logger.error(f"  Error: {result.get('error', 'Unknown')[:100]}...")
        logger.info("=" * 80)

        return result

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

            # JSON files - use file_analyzer for deep structure analysis
            elif path.suffix.lower() == '.json':
                try:
                    from backend.tools.file_analyzer_tool import file_analyzer

                    # Use file_analyzer's sophisticated JSON analysis
                    analysis = file_analyzer._analyze_json(str(path))

                    if "error" not in analysis:
                        # Extract rich structure information
                        depth_analysis = analysis.get("depth_analysis", {})

                        # Calculate max depth from depth_analysis
                        max_depth = file_analyzer._find_max_depth(depth_analysis) if depth_analysis else 0

                        # Generate smart access patterns based on structure
                        access_patterns = self._generate_json_access_patterns(analysis, depth_analysis)

                        # Extract SAFE preview (limit depth and size to prevent context overflow)
                        preview_data = analysis.get("preview", {})
                        safe_preview = self._create_safe_json_preview(preview_data)

                        # Check if null values exist (requires defensive coding)
                        requires_null_check = self._check_for_null_values(preview_data)

                        metadata.update({
                            "type": "json",
                            "structure": analysis.get("structure", "unknown"),
                            "keys": analysis.get("keys", []),
                            "structure_summary": analysis.get("structure_summary", ""),
                            "depth_analysis": depth_analysis,
                            "items_count": analysis.get("items_count", 0),
                            "first_item_type": analysis.get("first_item_type", None),
                            "max_depth": max_depth,
                            "access_patterns": access_patterns,  # NEW: Smart access patterns
                            "safe_preview": safe_preview,  # NEW: Safe preview instead of full data
                            "requires_null_check": requires_null_check  # NEW: Null value warning
                        })
                        logger.info(f"[PythonCoderTool] JSON structure analyzed: {analysis.get('structure')} with {len(analysis.get('keys', []))} top-level keys, depth={max_depth}, {len(access_patterns)} access patterns generated")
                    else:
                        # Malformed JSON - include error info
                        metadata.update({
                            "type": "json",
                            "error": analysis.get("error"),
                            "parsing_note": "JSON file may be malformed or use non-standard format"
                        })
                        logger.warning(f"[PythonCoderTool] JSON analysis error: {analysis.get('error')}")

                except Exception as e:
                    logger.warning(f"[PythonCoderTool] Could not extract JSON metadata: {e}")
                    metadata.update({
                        "type": "json",
                        "error": str(e)[:100]
                    })

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

            # Word documents (.docx)
            elif path.suffix.lower() == '.docx':
                try:
                    from backend.tools.file_analyzer_tool import file_analyzer
                    analysis = file_analyzer._analyze_docx(str(path))

                    if "error" not in analysis:
                        metadata.update({
                            "type": "docx",
                            "total_words": analysis.get("total_words", 0),
                            "total_paragraphs": analysis.get("total_paragraphs", 0),
                            "total_tables": analysis.get("total_tables", 0),
                            "table_details": analysis.get("tables", []),
                            "headings": analysis.get("headings", []),
                            "text_preview": analysis.get("text_preview", "")[:200]
                        })
                        logger.info(f"[PythonCoderTool] DOCX analyzed: {analysis.get('total_words')} words, {analysis.get('total_tables')} tables")
                    else:
                        metadata.update({
                            "type": "docx",
                            "error": analysis.get("error")
                        })
                except Exception as e:
                    logger.warning(f"[PythonCoderTool] Could not extract DOCX metadata: {e}")
                    metadata.update({"type": "docx"})

            # Excel files - use enhanced analyzer
            elif path.suffix.lower() in ['.xlsx', '.xls', '.xlsm']:
                try:
                    from backend.tools.file_analyzer_tool import file_analyzer
                    analysis = file_analyzer._analyze_excel(str(path))

                    if "error" not in analysis:
                        # Extract enhanced Excel metadata
                        metadata.update({
                            "type": "excel",
                            "total_sheets": analysis.get("total_sheets", 0),
                            "sheet_names": analysis.get("sheet_names", []),
                            "sheets_analyzed": analysis.get("sheets_analyzed", []),
                            "has_formulas": analysis.get("has_formulas", False),
                            "has_named_ranges": analysis.get("has_named_ranges", False),
                            "has_merged_cells": analysis.get("has_merged_cells", False)
                        })
                        logger.info(f"[PythonCoderTool] Excel analyzed: {analysis.get('total_sheets')} sheets, formulas={analysis.get('has_formulas')}")
                    else:
                        metadata.update({
                            "type": "excel",
                            "error": analysis.get("error")
                        })
                except Exception as e:
                    logger.warning(f"[PythonCoderTool] Could not extract enhanced Excel metadata: {e}")
                    # Fallback to basic pandas analysis
                    try:
                        import pandas as pd
                        df = pd.read_excel(path, nrows=5)
                        metadata.update({
                            "type": "excel",
                            "columns": df.columns.tolist(),
                            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                            "sample_rows": len(df)
                        })
                    except:
                        metadata.update({"type": "excel"})

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

    def _generate_json_access_patterns(
        self,
        analysis: Dict[str, Any],
        depth_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate smart access pattern examples based on JSON structure.

        Args:
            analysis: JSON analysis from file_analyzer
            depth_analysis: Depth analysis from file_analyzer

        Returns:
            List of code examples showing how to access the JSON data
        """
        patterns = []
        structure = analysis.get("structure", "unknown")

        if structure == "dict":
            keys = analysis.get("keys", [])
            for key in keys[:3]:  # Show first 3 keys
                # Safe access pattern
                patterns.append(f"data.get('{key}', default_value)")
                # Check existence pattern
                patterns.append(f"if '{key}' in data: value = data['{key}']")

            # Nested access from depth_analysis
            if depth_analysis and depth_analysis.get("children"):
                for key, child in list(depth_analysis["children"].items())[:2]:
                    if child.get("type") == "dict":
                        child_keys = child.get("keys", [])
                        if child_keys:
                            patterns.append(
                                f"data.get('{key}', {{}}).get('{child_keys[0]}', default)"
                            )
                    elif child.get("type") == "list":
                        patterns.append(f"if len(data.get('{key}', [])) > 0: item = data['{key}'][0]")

        elif structure == "list":
            items_count = analysis.get("items_count", 0)
            first_item_type = analysis.get("first_item_type")

            patterns.append(f"# Array with {items_count} items")
            patterns.append("if len(data) > 0: first_item = data[0]")

            if first_item_type == "dict":
                keys = analysis.get("keys", [])
                if keys:
                    patterns.append(f"items = [item.get('{keys[0]}') for item in data if '{keys[0]}' in item]")
                    patterns.append(f"for item in data:\n        value = item.get('{keys[0]}', default)")

        return patterns[:8]  # Return max 8 patterns

    def _create_safe_json_preview(self, preview_data: Any, max_depth: int = 2, current_depth: int = 0) -> Any:
        """
        Create a safe, truncated preview of JSON data to avoid context overflow.

        Args:
            preview_data: Original preview data
            max_depth: Maximum nesting depth to show
            current_depth: Current depth (for recursion)

        Returns:
            Truncated preview safe for LLM context
        """
        if current_depth >= max_depth:
            return "... (nested data omitted)"

        if isinstance(preview_data, dict):
            safe_dict = {}
            for i, (key, value) in enumerate(preview_data.items()):
                if i >= 5:  # Max 5 keys at each level
                    safe_dict["..."] = f"({len(preview_data) - 5} more keys)"
                    break
                safe_dict[key] = self._create_safe_json_preview(value, max_depth, current_depth + 1)
            return safe_dict

        elif isinstance(preview_data, list):
            if len(preview_data) == 0:
                return []
            # Show first 3 items only
            safe_list = [
                self._create_safe_json_preview(item, max_depth, current_depth + 1)
                for item in preview_data[:3]
            ]
            if len(preview_data) > 3:
                safe_list.append(f"... ({len(preview_data) - 3} more items)")
            return safe_list

        elif isinstance(preview_data, str):
            # Truncate long strings
            if len(preview_data) > 50:
                return preview_data[:50] + "..."
            return preview_data

        else:
            # Return primitives as-is
            return preview_data

    def _check_for_null_values(self, data: Any) -> bool:
        """
        Check if JSON data contains any None/null values.

        Args:
            data: JSON data to check

        Returns:
            True if null values found, False otherwise
        """
        if data is None:
            return True

        if isinstance(data, dict):
            for value in data.values():
                if self._check_for_null_values(value):
                    return True

        elif isinstance(data, list):
            for item in data[:10]:  # Check first 10 items only
                if self._check_for_null_values(item):
                    return True

        return False

    def _build_file_context(
        self,
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any]
    ) -> str:
        """
        Build file context string for LLM prompts.

        Args:
            validated_files: Dict of validated files
            file_metadata: Metadata for files

        Returns:
            Formatted file context string
        """
        if not validated_files:
            return ""

        file_context = """

🚨 CRITICAL - EXACT FILENAMES REQUIRED 🚨
ALL files are in the current working directory.
YOU MUST use the EXACT filenames shown below - NO generic names like 'file.json' or 'data.csv'!

Available files (USE THESE EXACT NAMES):
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
                file_context += f"   Structure: {metadata['structure']} ({metadata.get('items_count', 0)} items)\n"

                # Add detailed JSON structure information
                if file_type == 'json':
                    # Show top-level keys for objects
                    if 'keys' in metadata and metadata['keys']:
                        keys_display = metadata['keys'][:10]  # Show up to 10 keys
                        file_context += f"   Top-level keys: {', '.join(keys_display)}"
                        if len(metadata['keys']) > 10:
                            file_context += f" ... (+{len(metadata['keys']) - 10} more)"
                        file_context += "\n"

                    # Show depth and item type info
                    if 'max_depth' in metadata and metadata['max_depth'] > 1:
                        file_context += f"   Nesting depth: {metadata['max_depth']} levels\n"

                    if 'first_item_type' in metadata and metadata['first_item_type']:
                        file_context += f"   Array items are: {metadata['first_item_type']}\n"

                    # NEW: Show smart access patterns (most important!)
                    if 'access_patterns' in metadata and metadata['access_patterns']:
                        file_context += f"   📋 Access Patterns (COPY THESE EXACTLY):\n"
                        for pattern in metadata['access_patterns'][:6]:  # Show first 6 patterns
                            file_context += f"      {pattern}\n"

                    # NEW: Show safe preview (truncated to avoid context overflow)
                    if 'safe_preview' in metadata:
                        import json as json_module
                        try:
                            preview_str = json_module.dumps(metadata['safe_preview'], indent=2, ensure_ascii=False)[:500]
                            file_context += f"   Sample Data (first few items):\n"
                            for line in preview_str.split('\n')[:15]:
                                file_context += f"      {line}\n"
                        except:
                            pass  # Skip if can't serialize

                    # NEW: Warnings for special cases
                    if metadata.get('requires_null_check'):
                        file_context += f"   ⚠️  IMPORTANT: Contains null values - use .get() method for safe access\n"
                    if metadata.get('max_depth', 0) > 3:
                        file_context += f"   ⚠️  IMPORTANT: Deep nesting detected - validate each level before accessing\n"

            if 'line_count' in metadata:
                file_context += f"   Lines: {metadata['line_count']}\n"

            if 'preview' in metadata:
                preview = metadata['preview'][:100]
                file_context += f"   Preview: {preview}...\n"

            # Word document metadata
            if file_type == 'docx':
                if 'total_words' in metadata:
                    file_context += f"   Words: {metadata['total_words']}, Paragraphs: {metadata.get('total_paragraphs', 0)}\n"
                if 'total_tables' in metadata and metadata['total_tables'] > 0:
                    file_context += f"   Tables: {metadata['total_tables']} table(s)\n"
                    if 'table_details' in metadata:
                        for table in metadata['table_details'][:2]:  # Show first 2 tables
                            file_context += f"      Table {table['table_number']}: {table['rows']} rows × {table['columns']} cols\n"
                if 'headings' in metadata and metadata['headings']:
                    file_context += f"   Headings: {len(metadata['headings'])} found\n"
                if 'text_preview' in metadata:
                    file_context += f"   Text preview: {metadata['text_preview'][:150]}...\n"

            # Enhanced Excel metadata
            if file_type == 'excel':
                if 'total_sheets' in metadata:
                    sheets = ', '.join(metadata.get('sheet_names', [])[:3])
                    file_context += f"   Sheets ({metadata['total_sheets']} total): {sheets}\n"
                if metadata.get('has_formulas'):
                    file_context += f"   ⚠️  Contains formulas - use data_only=True or read_excel()\n"
                if metadata.get('has_merged_cells'):
                    file_context += f"   ⚠️  Contains merged cells - may affect data reading\n"
                if metadata.get('has_named_ranges'):
                    file_context += f"   Contains named ranges\n"
                # Show sheet details if available
                if 'sheets_analyzed' in metadata:
                    for sheet in metadata['sheets_analyzed'][:2]:  # First 2 sheets
                        file_context += f"      Sheet '{sheet['sheet_name']}': {sheet['rows']} rows × {sheet['columns']} cols\n"
                        if sheet.get('columns'):
                            cols_preview = ', '.join(sheet['columns'][:5])
                            file_context += f"         Columns: {cols_preview}\n"

            # File access example
            if file_type == 'csv':
                file_context += f"   Example: df = pd.read_csv('{original_filename}')\n"
            elif file_type == 'docx':
                file_context += f"   Example loading code:\n"
                file_context += f"      from docx import Document\n"
                file_context += f"      doc = Document('{original_filename}')\n"
                file_context += f"      # Extract text: text = '\\n'.join([p.text for p in doc.paragraphs])\n"
                if metadata.get('total_tables', 0) > 0:
                    file_context += f"      # Extract tables: tables = doc.tables\n"
            elif file_type == 'json':
                # Provide proper JSON loading with encoding and error handling
                file_context += f"   Example loading code:\n"
                file_context += f"      import json\n"
                file_context += f"      with open('{original_filename}', 'r', encoding='utf-8') as f:\n"
                file_context += f"          data = json.load(f)\n"

                # Add structure-specific access example
                if 'access_patterns' in metadata and metadata['access_patterns']:
                    # Use the first access pattern as example
                    first_pattern = metadata['access_patterns'][0]
                    file_context += f"      # Then use the access patterns above, e.g.:\n"
                    file_context += f"      # {first_pattern}\n"
                elif 'keys' in metadata and metadata['keys']:
                    example_key = metadata['keys'][0]
                    file_context += f"      # Access: value = data.get('{example_key}', default)\n"
                elif 'first_item_type' in metadata and metadata['first_item_type']:
                    file_context += f"      # Access: if len(data) > 0: item = data[0]\n"

                # Add error handling note if JSON had issues
                if 'error' in metadata:
                    file_context += f"   ⚠️  CRITICAL: Wrap in try/except json.JSONDecodeError (file has parsing issues)\n"
                elif 'parsing_note' in metadata:
                    file_context += f"   ⚠️  {metadata['parsing_note']}\n"
            elif file_type == 'excel':
                file_context += f"   Example: df = pd.read_excel('{original_filename}')\n"

        file_context += "\n"
        return file_context

    async def _generate_code(
        self,
        query: str,
        context: Optional[str],
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any],
        is_prestep: bool = False
    ) -> str:
        """
        Generate Python code using LLM.

        Args:
            query: User's question
            context: Optional additional context
            validated_files: Dict of validated files
            file_metadata: Metadata for files
            is_prestep: Whether this is pre-step execution (uses specialized prompt)

        Returns:
            Generated code
        """
        # Build file context using helper method
        file_context = self._build_file_context(validated_files, file_metadata)

        # Check if any JSON files are present
        has_json_files = any(
            metadata.get('type') == 'json'
            for metadata in file_metadata.values()
        )

        # Use different prompts for pre-step vs normal execution
        if is_prestep:
            # Build base prompt
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

            if file_context:  # Only add file-related instructions if files exist
                prompt_parts.extend([
                    "🚨 CRITICAL: Use the EXACT filenames shown in the file list above",
                    "🚨 DO NOT use generic names like 'file.json', 'data.csv', 'input.json', etc.",
                    "🚨 COPY the actual filename from the list - character by character",
                    "- NEVER makeup data, ALWAYS use the real files provided"
                ])

            prompt_parts.extend([
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

            # Add JSON-specific instructions ONLY if JSON files are present
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
                    "6. For nested access, validate each level: data.get('parent', {{}}).get('child', default)",
                    "7. For arrays, check length first: if len(data) > 0: item = data[0]",
                    "8. COPY the \"Access Patterns\" shown above - they are structure-validated",
                    "9. Handle None/null values: if value is not None: process(value)",
                    "10. Add debug prints: print(\"Data type:\", type(data), \"Keys:\", list(data.keys()) if isinstance(data, dict) else 'N/A')"
                ])

            prompt_parts.append("\nGenerate ONLY the Python code, no explanations or markdown:")
            prompt = "\n".join(prompt_parts)
        else:
            # Normal mode prompt (for ReAct loop iterations)
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

            if file_context:  # Only add file-related requirements if files exist
                prompt_parts.extend([
                    "🚨 CRITICAL: Use the EXACT filenames shown in the file list above",
                    "🚨 DO NOT use generic names like 'file.json', 'data.csv', 'input.xlsx', 'output.txt', etc.",
                    "🚨 COPY the actual filename from the list - including ALL special characters, numbers, Korean text",
                    "- Never add raw data to the code, always use the actual filenames to read the data",
                    "- Always use the real data. NEVER makeup data and ask user to input data."
                ])

            prompt_parts.extend([
                "- Output results using print() statements",
                "- Include error handling (try/except)",
                "- Add a docstring explaining what the code does",
                "- Keep code clean and readable"
            ])

            # Add JSON-specific requirements ONLY if JSON files are present
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
                    "6. Nested access: Use chained .get(): data.get('parent', {{}}).get('child', default)",
                    "7. Array safety: Check length before indexing: if len(data) > 0: item = data[0]",
                    "8. Copy patterns: The \"📋 Access Patterns\" are pre-validated - copy them exactly",
                    "9. Null handling: Check if value is not None before using",
                    "10. Debugging: Print data structure first: print(\"Type:\", type(data), \"Keys:\", list(data.keys()) if isinstance(data, dict) else len(data))"
                ])

            prompt_parts.append("\nGenerate ONLY the Python code, no explanations or markdown:")
            prompt = "\n".join(prompt_parts)

        try:
            logger.info("\n\n[PythonCoderTool] Generating code...")
            logger.info("=" * 80)
            if file_context:
                logger.info("[PythonCoderTool] File Context:")
                for line in file_context.strip().split('\n'):
                    logger.info(f"  {line}")
            if context:
                logger.info("[PythonCoderTool] Agent Context:")
                for line in context.strip().split('\n')[:20]:  # First 20 lines
                    logger.info(f"  {line}")
            logger.info("=" * 80)
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
            logger.info("[PythonCoderTool] Generated code:")
            logger.info("=" * 80)
            for line in code.split('\n'):
                logger.info(f"  {line}")
            logger.info("=" * 80)
            return code

        except Exception as e:
            logger.error(f"[PythonCoderTool] Failed to generate code: {e}")
            return ""

    async def _verify_code_answers_question(
        self,
        code: str,
        query: str,
        context: Optional[str] = None,
        file_context: str = "",
        file_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Verify code focuses on answering user's question.
        Simplified verification focused on the core goal.

        Args:
            code: Python code to verify
            query: Original user query
            context: Optional additional context
            file_context: Information about available files
            file_metadata: Optional file metadata to check for JSON files

        Returns:
            Tuple of (is_verified, list of issues)
        """
        issues = []

        # Static analysis checks (safety)
        is_valid, static_issues = self.executor.validate_imports(code)
        if not is_valid:
            issues.extend(static_issues)

        # LLM-based semantic check: Does it answer the question?
        semantic_issues = await self._llm_verify_answers_question(code, query, context, file_context, file_metadata)
        issues.extend(semantic_issues)

        is_verified = len(issues) == 0
        return is_verified, issues

    async def _llm_verify_answers_question(self, code: str, query: str, context: Optional[str] = None, file_context: str = "", file_metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Use LLM to verify if code answers the user's question.
        Simplified verification focused on core requirements.

        Args:
            code: Python code to verify
            query: Original user query
            context: Optional additional context
            file_context: Information about available files
            file_metadata: Optional file metadata to check for JSON files

        Returns:
            List of issues found
        """
        # Check if any JSON files are present
        has_json_files = False
        if file_metadata:
            has_json_files = any(
                metadata.get('type') == 'json'
                for metadata in file_metadata.values()
            )

        # Build verification prompt
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

        if file_context:  # Only check for real data if files are present
            prompt_parts.extend([
                "4️⃣ FILE HANDLING:",
                "   - Uses EXACT filenames from the file list?",
                "   - NO generic names like 'file.json', 'data.csv', 'input.xlsx'?",
                "   - File paths are strings, properly quoted?",
                "   - Uses ONLY real data (NO fake/placeholder data)?",
                "   - File reading has error handling (FileNotFoundError)?",
                ""
            ])

        # Add JSON-specific checks ONLY if JSON files are present
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
            "🚨 ERROR DETECTION PRIORITY:",
            "- Your primary goal is to find potential ERRORS (not style issues)",
            "- Focus on issues that will cause EXECUTION FAILURES or WRONG RESULTS",
            "- Be STRICT - even small issues can cause failures",
            "- If uncertain about filename correctness, mark it as an issue",
            "",
            "📋 RESPONSE FORMAT:",
            'Return a JSON object: {"verified": true/false, "issues": ["issue1", "issue2", ...]}',
            "",
            "✅ Return {\"verified\": true, \"issues\": []} ONLY IF:",
            "   - Code is 100% correct and will execute without errors",
            "   - All filenames are exact matches from the file list",
            "   - All required safety checks are present",
            f"{'   - All JSON safety patterns are followed' if has_json_files else ''}",
            "",
            "❌ Return {\"verified\": false, \"issues\": [...]} IF:",
            "   - ANY potential error detected (syntax, runtime, logic)",
            "   - Filenames don't match EXACTLY",
            "   - Missing error handling",
            "   - Unsafe data access patterns",
            f"{'   - JSON access patterns not followed' if has_json_files else ''}",
            "",
            "⚠️  BE THOROUGH: It's better to flag a potential issue than miss a real error.",
            ""
        ])

        prompt = "\n".join(prompt_parts)

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
        error_message: str,
        context: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Fix code based on execution error.

        Args:
            code: Current Python code
            query: Original user query
            error_message: Error from execution
            context: Optional additional context from agent execution history

        Returns:
            Tuple of (fixed_code, list of changes made)
        """
        prompt = f"""Fix the following Python code that failed during execution:

Original request: {query}
{f"Context: {context}" if context else ""}

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
