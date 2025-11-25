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
from .code_patcher import CodePatcher
from .context_builder import FileContextBuilder
from .file_handlers import FileHandlerFactory
from .utils import get_original_filename
from .file_context_storage import FileContextStorage

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
    - Phase 2: Verification loop (configurable via settings.python_code_max_iterations)
    - Phase 3: Execution loop (configurable via settings.python_code_max_iterations)
    - Phase 4: Result formatting

    Features:
    - Code generation using LLM
    - Verification focused on answering user's question
    - Code execution with retry on failure (max attempts from settings)
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
        self.code_patcher = CodePatcher(
            llm=self.llm,
            code_generator=self.code_generator,
            code_verifier=self.code_verifier,
            max_verification_iterations=settings.python_code_max_iterations
        )  # New: Incremental code building with verification
        self.file_handler_factory = FileHandlerFactory()
        self.context_builder = FileContextBuilder()

        # Configuration - OPTIMIZED for minimal LLM calls
        self.max_retry_attempts = settings.python_code_max_iterations  # Max attempts for execution + adequacy check
        self.allow_partial_execution = settings.python_code_allow_partial_execution

        logger.info(f"[PythonCoderTool] Initialized with OPTIMIZED workflow (max_retry_attempts={self.max_retry_attempts})")

    async def execute_code_task(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        is_prestep: bool = False,
        stage_prefix: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
        plan_context: Optional[dict] = None,
        react_context: Optional[dict] = None,
        react_step: Optional[int] = None,  # NEW: explicit react step number
        plan_step: Optional[int] = None   # NEW: explicit plan step number
    ) -> Dict[str, Any]:
        """
        OPTIMIZED: Main entry point for code generation and execution.

        New Workflow (Reduced LLM calls):
        0. Prepare files (non-LLM)
        1. Generate code WITH self-verification (1 LLM call - combined!)
        2. Execute code (non-LLM)
        3. Check output adequacy (1 LLM call only if needed)
        4. Retry if needed (max attempts from settings.python_code_max_iterations)

        Args:
            query: User's question/task
            context: Optional additional context
            file_paths: Optional list of input file paths
            session_id: Optional session ID
            is_prestep: Whether this is called from ReAct pre-step
            stage_prefix: Optional stage prefix for code file naming (legacy, prefer react_step/plan_step)
            conversation_history: Past conversation turns
            plan_context: Plan-Execute workflow context
            react_context: ReAct iteration context with failed attempts
            react_step: ReAct iteration/step number (for prompt organization)
            plan_step: Plan-Execute step number (for prompt organization)

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

        # Save file context to session directory for multi-phase workflows
        if session_id and validated_files:
            FileContextStorage.save_file_context(
                session_id=session_id,
                validated_files=validated_files,
                file_metadata=file_metadata,
                file_context_text=file_context
            )

        # Tracking
        attempt_history = []
        modifications = []

        # Main retry loop (max attempts from settings)
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
                attempt_num=attempt + 1,
                session_id=session_id,
                stage_prefix=stage_prefix,
                conversation_history=conversation_history,
                plan_context=plan_context,
                react_context=react_context,
                react_step=react_step,  # NEW: pass through to save
                plan_step=plan_step     # NEW: pass through to save
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
                logger.error(f"[PythonCoderTool] [X] Execution failed: {execution_result.get('error', 'Unknown')[:100]}")
                if attempt < self.max_retry_attempts - 1:
                    # Use INCREMENTAL PATCHING for attempts 2+ (preserve working sections)
                    if attempt > 0 and code:
                        logger.info("[PythonCoderTool] Using INCREMENTAL patching (preserving working sections)...")

                        # Analyze error to identify failed section
                        error_analysis = self.code_patcher.analyze_execution_error(
                            code=code,
                            error_message=execution_result.get("error", ""),
                            execution_output=execution_result.get("output", "")
                        )

                        # Generate patched code (only fixes failed section)
                        code = await self.code_patcher.patch_code(
                            original_code=code,
                            error_analysis=error_analysis,
                            query=query,
                            file_context=file_context,
                            attempt_num=attempt + 2  # Next attempt number
                        )

                        logger.info(f"[PythonCoderTool] Patched {error_analysis['error_location']} section")

                        # Skip LLM code generation - go directly to execution with patched code
                        continue
                    else:
                        # First attempt - use traditional retry with error feedback
                        logger.info("[PythonCoderTool] First attempt failed - will retry with full regeneration...")
                        context = self._build_retry_context(context, execution_result.get("error"), issues)
                        continue
                else:
                    logger.error(f"[PythonCoderTool] Max attempts ({self.max_retry_attempts}) reached")
                    break

            # Phase 3: Check output adequacy (1 LLM call, only if execution succeeded)
            logger.info("[PythonCoderTool] [OK] Execution succeeded, checking output adequacy...")
            is_adequate, adequacy_reason, suggestion = await self._check_output_adequacy(
                query=query,
                code=code,
                output=execution_result["output"],
                context=context
            )

            if is_adequate:
                logger.info(f"[PythonCoderTool] [OK] Output is adequate: {adequacy_reason}")
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
                logger.warning(f"[PythonCoderTool] [WARNING] Output not adequate: {adequacy_reason}")
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

    def _save_llm_prompt(
        self,
        messages: List[Any],
        prompt: str,
        query: str,
        file_context: str,
        validated_files: Dict[str, str],
        attempt_num: int,
        session_id: Optional[str] = None,
        react_step: Optional[int] = None,
        plan_step: Optional[int] = None,
        stage_prefix: Optional[str] = None
    ) -> None:
        """
        Save COMPLETE LLM input to scratch directory with hierarchical organization.

        NOW SAVES: System messages, conversation history, current prompt, message structure,
        and LLM configuration - EVERYTHING the LLM actually sees.

        Directory structure:
        /data/scratch/
        ├── {session_id}/              # Session-specific directory
        │   └── prompts/                # All prompts for this session
        │       ├── python_coder/       # Python coder tool prompts
        │       │   ├── attempt_1.txt
        │       │   ├── attempt_2.txt
        │       │   └── attempt_3.txt
        │       ├── react/              # ReAct agent prompts
        │       │   ├── step_1_thought_action.txt
        │       │   ├── step_2_thought_action.txt
        │       │   └── step_3_thought_action.txt
        │       └── plan_execute/       # Plan-Execute prompts
        │           ├── step_1_plan.txt
        │           ├── step_2_execute.txt
        │           └── step_3_verify.txt
        └── global_prompts/             # Prompts without session_id (fallback)
            └── {timestamp}_{tool}.txt

        Args:
            messages: COMPLETE list of messages sent to LLM (system, history, current)
            prompt: Full LLM prompt text (current message only)
            query: User's original query
            file_context: File context string
            validated_files: Dict of validated files
            attempt_num: Current attempt number
            session_id: Optional session ID for organization
            react_step: Optional ReAct iteration number
            plan_step: Optional plan-execute step number
            stage_prefix: Optional stage prefix (e.g., "plan", "execute", "verify")
        """
        try:
            from datetime import datetime
            from .llm_input_formatter import format_llm_input

            # Determine directory structure
            if session_id:
                # Session-specific organization
                session_dir = Path(settings.python_code_execution_dir) / session_id
                prompts_base = session_dir / "prompts"
            else:
                # Global fallback for prompts without session
                prompts_base = Path(settings.python_code_execution_dir) / "global_prompts"

            # Create subdirectory based on context
            if react_step is not None:
                # ReAct agent prompts
                prompts_dir = prompts_base / "react"
                filename = f"step_{react_step:02d}_attempt_{attempt_num}.txt"
                description = f"ReAct Step {react_step}, Attempt {attempt_num}"
            elif plan_step is not None and stage_prefix:
                # Plan-Execute prompts
                prompts_dir = prompts_base / "plan_execute"
                filename = f"step_{plan_step:02d}_{stage_prefix}_attempt_{attempt_num}.txt"
                description = f"Plan-Execute Step {plan_step} ({stage_prefix}), Attempt {attempt_num}"
            elif stage_prefix:
                # Stage-specific (e.g., verification, execution)
                prompts_dir = prompts_base / "python_coder" / stage_prefix
                filename = f"attempt_{attempt_num}.txt"
                description = f"Python Coder ({stage_prefix}), Attempt {attempt_num}"
            else:
                # Default python coder prompts
                prompts_dir = prompts_base / "python_coder"
                filename = f"attempt_{attempt_num}.txt"
                description = f"Python Coder, Attempt {attempt_num}"

            # Create directory structure
            prompts_dir.mkdir(parents=True, exist_ok=True)

            # Build file path
            prompt_file = prompts_dir / filename

            # Use the new formatter to get COMPLETE LLM input
            complete_input = format_llm_input(
                messages=messages,
                llm=self.llm,
                description=description
            )

            # Add metadata header
            metadata_lines = []
            metadata_lines.append("=" * 80)
            metadata_lines.append("METADATA")
            metadata_lines.append("=" * 80)
            metadata_lines.append(f"Timestamp: {datetime.now().isoformat()}")
            metadata_lines.append(f"Session ID: {session_id or 'N/A'}")
            metadata_lines.append(f"Attempt: {attempt_num}")
            if react_step is not None:
                metadata_lines.append(f"ReAct Step: {react_step}")
            if plan_step is not None:
                metadata_lines.append(f"Plan Step: {plan_step}")
            if stage_prefix:
                metadata_lines.append(f"Stage: {stage_prefix}")
            metadata_lines.append(f"Query: {query[:200]}{'...' if len(query) > 200 else ''}")
            metadata_lines.append(f"Files: {len(validated_files)}")
            metadata_lines.append("")

            # Combine metadata + complete LLM input
            final_content = "\n".join(metadata_lines) + "\n" + complete_input

            # Write to file
            prompt_file.write_text(final_content, encoding='utf-8')

            # Log with relative path for readability
            relative_path = prompt_file.relative_to(Path(settings.python_code_execution_dir))
            logger.info(f"[PythonCoderTool] [SAVED] COMPLETE LLM input → {relative_path}")

        except Exception as e:
            logger.warning(f"[PythonCoderTool] Failed to save LLM prompt: {e}")

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
            logger.info(f"[PythonCoderTool] [SAVED] Saved {stage_name} code to {stage_script_path.name}")
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

    def get_saved_file_context(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Load previously saved file context from session directory.

        This enables multi-phase workflows where file analysis is done once
        and reused across multiple phases without re-processing files.

        Args:
            session_id: Session ID for directory path

        Returns:
            Dict with file context data, or None if not found
        """
        return FileContextStorage.load_file_context(session_id)

    def has_saved_file_context(self, session_id: Optional[str]) -> bool:
        """
        Check if saved file context exists for session.

        Args:
            session_id: Session ID to check

        Returns:
            True if context file exists, False otherwise
        """
        return FileContextStorage.has_file_context(session_id)

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
        attempt_num: int,
        session_id: Optional[str] = None,
        stage_prefix: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
        plan_context: Optional[dict] = None,
        react_context: Optional[dict] = None,
        react_step: Optional[int] = None,  # NEW
        plan_step: Optional[int] = None    # NEW
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
            has_json_files=has_json_files,
            conversation_history=conversation_history,
            plan_context=plan_context,
            react_context=react_context
        )

        logger.info(f"[PythonCoderTool] Generating code with self-verification (attempt {attempt_num})...")

        # Prepare the EXACT messages that will be sent to LLM
        messages = [HumanMessage(content=prompt)]

        # Save COMPLETE LLM input (including any system messages or history)
        self._save_llm_prompt(
            messages=messages,
            prompt=prompt,
            query=query,
            file_context=file_context,
            validated_files=validated_files,
            attempt_num=attempt_num,
            session_id=session_id,
            stage_prefix=stage_prefix,
            react_step=react_step,  # NEW: proper hierarchy
            plan_step=plan_step     # NEW: proper hierarchy
        )

        try:
            # Send the EXACT same messages to LLM
            response = await self.llm.ainvoke(messages)
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
            raise

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
            logger.error(f"[PythonCoderTool] Failed to check output adequacy: {e}")
            raise

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
        # Extract namespace from execution result
        namespace = execution_result.get("namespace", {})
        
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
            "attempt_history": attempt_history,
            "namespace": namespace  # Add namespace to result
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
        # Extract namespace from execution result
        namespace = execution_result.get("namespace", {})
        
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
            "attempt_history": attempt_history,
            "namespace": namespace  # Add namespace to result
        }


# ============================================================================
# Global Instance
# ============================================================================

python_coder_tool = PythonCoderTool()


# ============================================================================
# Exports
# ============================================================================
