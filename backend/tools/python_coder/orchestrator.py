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

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.config.settings import settings
from backend.utils.logging_utils import get_logger
from backend.utils.llm_factory import LLMFactory

# Import all the sub-components
from .executor import CodeExecutor, SUPPORTED_FILE_TYPES
from .code_generator import CodeGenerator
from .code_verifier import CodeVerifier
from .code_fixer import CodeFixer
from .auto_fixer import AutoFixer
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
    - Phase 2: Verification loop (max 2 iterations)
    - Phase 3: Execution loop (max 3 attempts)
    - Phase 4: Result formatting

    Features:
    - Code generation using LLM
    - Verification focused on answering user's question (max 2 iterations)
    - Code execution with retry on failure (max 3 attempts)
    - File handling and metadata extraction
    """

    def __init__(self):
        """Initialize the Python coder tool and all sub-components."""
        # Initialize LLM using factory
        self.llm = LLMFactory.create_coder_llm()

        # Initialize executor
        self.executor = CodeExecutor(
            timeout=settings.python_code_timeout,
            max_memory_mb=settings.python_code_max_memory,
            execution_base_dir=settings.python_code_execution_dir,
            use_persistent_repl=settings.python_code_use_persistent_repl
        )

        # Initialize all sub-components
        self.code_generator = CodeGenerator(self.llm)
        self.code_verifier = CodeVerifier(self.llm, self.executor)
        self.code_fixer = CodeFixer(self.llm)
        self.auto_fixer = AutoFixer()
        self.file_handler_factory = FileHandlerFactory()
        self.context_builder = FileContextBuilder()

        # Configuration - OPTIMIZED for minimal LLM calls
        self.max_retry_attempts = 3  # Max attempts for execution + adequacy check
        self.allow_partial_execution = settings.python_code_allow_partial_execution

        logger.info(f"[PythonCoderTool] Initialized with OPTIMIZED workflow (max_retry_attempts={self.max_retry_attempts})")

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
        OPTIMIZED: Main entry point for code generation and execution.

        New Workflow (Reduced LLM calls):
        0. Prepare files (non-LLM)
        1. Generate code WITH self-verification (1 LLM call - combined!)
        2. Execute code (non-LLM)
        3. Check output adequacy (1 LLM call only if needed)
        4. Retry if needed (max 3 total attempts)

        Args:
            query: User's question/task
            context: Optional additional context
            file_paths: Optional list of input file paths
            session_id: Optional session ID
            is_prestep: Whether this is called from ReAct pre-step
            stage_prefix: Optional stage prefix for code file naming

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

        logger.info(f"[PythonCoderTool OPTIMIZED] Executing task: {query[:100]}...")

        # Phase 0: Prepare input files (non-LLM)
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

        # Build file context
        file_context = self.context_builder.build_context(validated_files, file_metadata)
        has_json_files = any(m.get('type') == 'json' for m in file_metadata.values())

        # Tracking
        attempt_history = []
        modifications = []

        # Main retry loop (max 3 attempts)
        for attempt in range(self.max_retry_attempts):
            logger.info(f"\n[PythonCoderTool] === ATTEMPT {attempt + 1}/{self.max_retry_attempts} ===")

            # Phase 1: Generate code WITH self-verification (1 LLM call!)
            code, self_verified, issues = await self._generate_code_with_self_verification(
                query=query,
                context=context,
                file_context=file_context,
                validated_files=validated_files,
                file_metadata=file_metadata,
                is_prestep=is_prestep,
                has_json_files=has_json_files,
                attempt_num=attempt + 1
            )

            if not code:
                return {
                    "success": False,
                    "error": "Failed to generate code",
                    "code": "",
                    "output": ""
                }

            # Apply automatic fixes
            code, auto_changes = self.auto_fixer.apply_all_fixes(code, validated_files, file_metadata)
            if auto_changes:
                modifications.extend(auto_changes)
                logger.info(f"[PythonCoderTool] Applied {len(auto_changes)} automatic fix(es)")

            # Phase 2: Execute code (non-LLM)
            stage_name = f"{stage_prefix}_attempt{attempt + 1}" if stage_prefix else f"attempt{attempt + 1}"
            execution_result = self.executor.execute(
                code=code,
                input_files=validated_files,
                session_id=session_id,
                stage_name=stage_name
            )

            attempt_history.append({
                "attempt": attempt + 1,
                "self_verified": self_verified,
                "issues_found": issues,
                "execution_success": execution_result["success"],
                "execution_error": execution_result.get("error"),
                "execution_time": execution_result["execution_time"]
            })

            # If execution failed, prepare for retry
            if not execution_result["success"]:
                logger.error(f"[PythonCoderTool] ❌ Execution failed: {execution_result.get('error', 'Unknown')[:100]}")
                if attempt < self.max_retry_attempts - 1:
                    logger.info("[PythonCoderTool] Will retry with error feedback...")
                    context = self._build_retry_context(context, execution_result.get("error"), issues)
                    continue
                else:
                    logger.error(f"[PythonCoderTool] Max attempts ({self.max_retry_attempts}) reached")
                    break

            # Phase 3: Check output adequacy (1 LLM call, only if execution succeeded)
            logger.info("[PythonCoderTool] ✅ Execution succeeded, checking output adequacy...")
            is_adequate, adequacy_reason, suggestion = await self._check_output_adequacy(
                query=query,
                code=code,
                output=execution_result["output"],
                context=context
            )

            if is_adequate:
                logger.info(f"[PythonCoderTool] ✅ Output is adequate: {adequacy_reason}")
                # SUCCESS - return result
                return self._build_success_result(
                    code=code,
                    execution_result=execution_result,
                    attempt_history=attempt_history,
                    modifications=modifications,
                    validated_files=validated_files,
                    file_metadata=file_metadata
                )
            else:
                # Output not adequate - retry if attempts remain
                logger.warning(f"[PythonCoderTool] ⚠️  Output not adequate: {adequacy_reason}")
                if attempt < self.max_retry_attempts - 1:
                    logger.info(f"[PythonCoderTool] Suggestion: {suggestion}")
                    context = self._build_retry_context(context, None, [suggestion])
                    continue
                else:
                    logger.warning(f"[PythonCoderTool] Max attempts reached, returning current result")
                    break

        # If we get here, all attempts exhausted - return last result
        return self._build_final_result(
            code=code,
            execution_result=execution_result,
            attempt_history=attempt_history,
            modifications=modifications,
            validated_files=validated_files,
            file_metadata=file_metadata
        )

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

    async def _generate_code_with_self_verification(
        self,
        query: str,
        context: Optional[str],
        file_context: str,
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any],
        is_prestep: bool,
        has_json_files: bool,
        attempt_num: int
    ) -> Tuple[str, bool, List[str]]:
        """
        OPTIMIZED: Generate code WITH self-verification in single LLM call.

        Returns:
            Tuple of (code, self_verified, issues_list)
        """
        from backend.config.prompts.python_coder import get_code_generation_with_self_verification_prompt
        from langchain_core.messages import HumanMessage
        import json

        prompt = get_code_generation_with_self_verification_prompt(
            query=query,
            context=context,
            file_context=file_context,
            is_prestep=is_prestep,
            has_json_files=has_json_files
        )

        logger.info(f"[PythonCoderTool] Generating code with self-verification (attempt {attempt_num})...")

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            result = json.loads(response_text.strip())

            code = result.get("code", "")
            self_verified = result.get("self_check_passed", False)
            issues = result.get("issues", [])

            logger.info(f"[PythonCoderTool] Generated code (self-verified: {self_verified}, issues: {len(issues)})")
            if issues:
                for i, issue in enumerate(issues, 1):
                    logger.warning(f"  Issue {i}: {issue}")

            return code, self_verified, issues

        except Exception as e:
            logger.error(f"[PythonCoderTool] Failed to generate code with self-verification: {e}")
            # Fallback: try old method
            logger.warning("[PythonCoderTool] Falling back to old generation method...")
            code = await self.code_generator.generate_code(
                query=query,
                context=context,
                validated_files=validated_files,
                file_metadata=file_metadata,
                file_context=file_context,
                is_prestep=is_prestep
            )
            return code, False, []

    async def _check_output_adequacy(
        self,
        query: str,
        code: str,
        output: str,
        context: Optional[str]
    ) -> Tuple[bool, str, str]:
        """
        OPTIMIZED: Check if code output adequately answers the question.

        Returns:
            Tuple of (is_adequate, reason, suggestion)
        """
        from backend.config.prompts.python_coder import get_output_adequacy_check_prompt
        from langchain_core.messages import HumanMessage
        import json

        prompt = get_output_adequacy_check_prompt(
            query=query,
            code=code,
            output=output,
            context=context
        )

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            result = json.loads(response_text.strip())

            is_adequate = result.get("adequate", True)  # Default to adequate if parsing fails
            reason = result.get("reason", "")
            suggestion = result.get("suggestion", "")

            return is_adequate, reason, suggestion

        except Exception as e:
            logger.warning(f"[PythonCoderTool] Failed to check output adequacy: {e}")
            # Default to adequate if check fails
            return True, "Adequacy check failed, assuming adequate", ""

    def _build_retry_context(
        self,
        original_context: Optional[str],
        error: Optional[str],
        suggestions: List[str]
    ) -> str:
        """Build context for retry attempt with error/suggestion feedback."""
        parts = []

        if original_context:
            parts.append(original_context)

        if error:
            parts.append(f"\n\nPrevious execution error:\n{error}")

        if suggestions:
            parts.append(f"\n\nSuggestions for improvement:\n" + "\n".join(f"- {s}" for s in suggestions))

        return "\n".join(parts)

    def _build_success_result(
        self,
        code: str,
        execution_result: Dict[str, Any],
        attempt_history: List[Dict],
        modifications: List[str],
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build success result dictionary."""
        return {
            "success": True,
            "code": code,
            "output": execution_result["output"],
            "error": None,
            "execution_time": execution_result["execution_time"],
            "total_attempts": len(attempt_history),
            "modifications": modifications,
            "input_files": list(validated_files.keys()),
            "file_metadata": file_metadata,
            "attempt_history": attempt_history
        }

    def _build_final_result(
        self,
        code: str,
        execution_result: Dict[str, Any],
        attempt_history: List[Dict],
        modifications: List[str],
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build final result dictionary (may be success or failure)."""
        return {
            "success": execution_result["success"],
            "code": code,
            "output": execution_result.get("output", ""),
            "error": execution_result.get("error"),
            "execution_time": execution_result["execution_time"],
            "total_attempts": len(attempt_history),
            "modifications": modifications,
            "input_files": list(validated_files.keys()),
            "file_metadata": file_metadata,
            "attempt_history": attempt_history
        }


# ============================================================================
# Global Instance
# ============================================================================

python_coder_tool = PythonCoderTool()


# ============================================================================
# Backward Compatibility Exports
# ============================================================================

# For backward compatibility with existing imports
PythonExecutor = CodeExecutor
