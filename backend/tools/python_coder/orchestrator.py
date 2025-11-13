"""
Python Coder Tool - Orchestrator

Main orchestrator that coordinates all components of the Python code generation,
verification, and execution system. This module ties together:
- File handling and context building
- Code generation
- Code verification
- Code fixing
- Code execution

This is the main entry point for the Python coder tool.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from backend.config.settings import settings
from backend.config import prompts
from backend.utils.logging_utils import get_logger

# Import all the sub-components
from .executor import CodeExecutor, SUPPORTED_FILE_TYPES
from .code_generator import CodeGenerator
from .code_verifier import CodeVerifier
from .code_fixer import CodeFixer
from .context_builder import FileContextBuilder
from .file_handlers import FileHandlerFactory
from .utils import get_original_filename

logger = get_logger(__name__)


# ============================================================================
# Main Python Coder Tool Orchestrator
# ============================================================================

class PythonCoderTool:
    """
    Python code generator tool with iterative verification and execution with retry logic.

    This orchestrator coordinates the full workflow:
    - Phase 0: File preparation and metadata extraction
    - Phase 1: Code generation
    - Phase 2: Verification loop (max 3 iterations)
    - Phase 3: Execution loop (max 5 attempts)
    - Phase 4: Result formatting

    Features:
    - Code generation using LLM
    - Verification focused on answering user's question (max 3 iterations)
    - Code execution with retry on failure (max 5 attempts)
    - File handling and metadata extraction
    """

    def __init__(self):
        """Initialize the Python coder tool and all sub-components."""
        # Initialize LLM
        self.llm = ChatOllama(
            base_url=settings.ollama_host,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            num_ctx=settings.ollama_num_ctx,
            top_p=settings.ollama_top_p,
            top_k=settings.ollama_top_k
        )

        # Initialize executor
        self.executor = CodeExecutor(
            timeout=settings.python_code_timeout,
            max_memory_mb=settings.python_code_max_memory,
            execution_base_dir=settings.python_code_execution_dir
        )

        # Initialize all sub-components
        self.code_generator = CodeGenerator(self.llm)
        self.code_verifier = CodeVerifier(self.llm, self.executor)
        self.code_fixer = CodeFixer(self.llm)
        self.file_handler_factory = FileHandlerFactory()
        self.context_builder = FileContextBuilder()

        # Configuration
        self.max_verification_iterations = 3
        self.max_execution_attempts = 5
        self.allow_partial_execution = settings.python_code_allow_partial_execution

        logger.info(f"[PythonCoderTool] Initialized with verification_iterations={self.max_verification_iterations}, execution_attempts={self.max_execution_attempts}")

    async def execute_code_task(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        is_prestep: bool = False,
        stage_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for code generation and execution.

        Args:
            query: User's question/task
            context: Optional additional context
            file_paths: Optional list of input file paths
            session_id: Optional session ID
            is_prestep: Whether this is called from ReAct pre-step (uses specialized prompt)
            stage_prefix: Optional stage prefix (e.g., "step5", "stage3") for code file naming

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
        file_context = self.context_builder.build_context(validated_files, file_metadata)

        if file_context:
            logger.info("[PythonCoderTool] File context built successfully")
            logger.debug(f"[PythonCoderTool] File context preview:\n{file_context[:500]}...")

        # Phase 1: Generate initial code
        code = await self.code_generator.generate_code(
            query=query,
            context=context,
            validated_files=validated_files,
            file_metadata=file_metadata,
            is_prestep=is_prestep
        )

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
            verified, issues = await self.code_verifier.verify_code_answers_question(
                code=code,
                query=query,
                context=context,
                file_context=file_context,
                file_metadata=file_metadata
            )

            verification_history.append({
                "iteration": iteration + 1,
                "issues": issues,
                "action": "approved" if verified else "needs_modification"
            })

            # Save verification stage code
            stage_suffix = f"verify{iteration + 1}"
            stage_name = f"{stage_prefix}_{stage_suffix}" if stage_prefix else stage_suffix
            self._save_stage_code(code, validated_files, session_id, stage_name)

            if verified:
                logger.info(f"[PythonCoderTool] ✅ Code verified successfully at iteration {iteration + 1}")
                break

            # AUTO-FIX: Try to automatically fix common issues before calling LLM
            auto_fixed, auto_changes = self.code_fixer.auto_fix_common_issues(
                code=code,
                validated_files=validated_files,
                file_metadata=file_metadata
            )

            if auto_changes:
                logger.info(f"[PythonCoderTool] 🔧 Auto-fixed {len(auto_changes)} issue(s):")
                for change in auto_changes:
                    logger.info(f"  - {change}")
                code = auto_fixed
                modifications.extend(auto_changes)

            # Modify code with LLM
            logger.warning(f"[PythonCoderTool] ⚠️  Verification issues found ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                logger.warning(f"  {i}. {issue}")
            logger.info(f"[PythonCoderTool] Modifying code to address issues...")

            code, changes = await self.code_generator.modify_code(
                code=code,
                issues=issues,
                query=query,
                context=context
            )
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

            # Save execution stage code
            stage_suffix = f"exec{attempt + 1}"
            stage_name = f"{stage_prefix}_{stage_suffix}" if stage_prefix else stage_suffix
            execution_result = self.executor.execute(
                code=code,
                input_files=validated_files,
                session_id=session_id,
                stage_name=stage_name
            )

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
                code, fix_changes = await self.code_fixer.fix_execution_error(
                    code=code,
                    query=query,
                    error_message=error_message,
                    context=context
                )
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
        logger.header("PYTHON CODER EXECUTION COMPLETE", "heavy")

        summary = {
            "Status": "SUCCESS" if result['success'] else "FAILED",
            "Verification Iterations": result['verification_iterations'],
            "Execution Attempts": result['execution_attempts'],
            "Code Modifications": len(modifications),
            "Execution Time": f"{result['execution_time']:.2f}s"
        }

        if result['success']:
            summary["Output Length"] = f"{len(result['output'])} chars"
            logger.key_values(summary, title="Execution Summary")
            logger.success("Code generation and execution completed successfully")
        else:
            summary["Error"] = result.get('error', 'Unknown')[:150]
            logger.key_values(summary, title="Execution Summary")
            logger.failure("Code execution failed", result.get('error', 'Unknown')[:100])

        return result

    def _save_stage_code(
        self,
        code: str,
        validated_files: Dict[str, str],
        session_id: Optional[str],
        stage_name: str
    ) -> None:
        """
        Save code to a stage-specific file for debugging/tracking.

        Args:
            code: Python code to save
            validated_files: Dict of validated files
            session_id: Session ID for directory path
            stage_name: Stage name (e.g., "verify1", "exec2", "stage5")
        """
        if not session_id:
            # No session ID, skip saving
            return

        try:
            execution_dir = Path(settings.python_code_execution_dir) / session_id
            execution_dir.mkdir(parents=True, exist_ok=True)

            stage_script_path = execution_dir / f"script_{stage_name}.py"
            stage_script_path.write_text(code, encoding='utf-8')
            logger.info(f"[PythonCoderTool] 💾 Saved {stage_name} code to {stage_script_path.name}")
        except Exception as e:
            logger.warning(f"[PythonCoderTool] Failed to save stage code: {e}")

    def get_previous_code_history(
        self,
        session_id: Optional[str],
        max_versions: int = 3
    ) -> List[Dict[str, str]]:
        """
        Load previous code versions from session directory.

        Args:
            session_id: Session ID for directory path
            max_versions: Maximum number of previous versions to return

        Returns:
            List of dicts with keys: stage_name, code, timestamp
        """
        if not session_id:
            return []

        try:
            execution_dir = Path(settings.python_code_execution_dir) / session_id
            if not execution_dir.exists():
                return []

            # Find all script_*.py files (excluding main script.py)
            script_files = list(execution_dir.glob("script_*.py"))
            if not script_files:
                return []

            # Sort by modification time (most recent first)
            script_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Load up to max_versions files
            code_history = []
            for script_path in script_files[:max_versions]:
                try:
                    code = script_path.read_text(encoding='utf-8')
                    stage_name = script_path.stem.replace('script_', '')  # Extract stage name
                    timestamp = script_path.stat().st_mtime

                    code_history.append({
                        "stage_name": stage_name,
                        "code": code,
                        "timestamp": timestamp,
                        "filename": script_path.name
                    })
                except Exception as e:
                    logger.warning(f"[PythonCoderTool] Failed to read {script_path.name}: {e}")

            if code_history:
                logger.info(f"[PythonCoderTool] Loaded {len(code_history)} previous code version(s)")

            return code_history

        except Exception as e:
            logger.warning(f"[PythonCoderTool] Failed to load code history: {e}")
            return []

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
            original_filename = get_original_filename(path.name)

            # Extract metadata using FileHandlerFactory
            metadata = self.file_handler_factory.extract_metadata(path)
            metadata['original_filename'] = original_filename  # Add to metadata
            file_metadata[str(path)] = metadata

            # Store mapping (temp path -> ORIGINAL filename for execution)
            validated_files[str(path)] = original_filename

            logger.info(f"[PythonCoderTool] Validated file: {original_filename} (temp: {path.name}, {size_mb:.2f}MB)")

        return validated_files, file_metadata


# ============================================================================
# Global Instance
# ============================================================================

python_coder_tool = PythonCoderTool()


# ============================================================================
# Backward Compatibility Exports
# ============================================================================

# For backward compatibility with existing imports
PythonExecutor = CodeExecutor
