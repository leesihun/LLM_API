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

            # Classify error for better retry guidance (with context)
            error_type = self._classify_error(
                error_message=execution_result.get("error", ""),
                return_code=execution_result.get("return_code", -1),
                stdout=execution_result.get("output", ""),
                namespace=execution_result.get("namespace", {})
            ) if not execution_result["success"] else ("Success", "")
            
            attempt_history.append({
                "attempt": attempt + 1,
                "code": code,  # Store full code for retry context
                "self_verified": self_verified,
                "issues_found": issues,
                "execution_success": execution_result["success"],
                "execution_error": execution_result.get("error"),
                "error_type": error_type,  # Classified error type + guidance
                "execution_time": execution_result["execution_time"],
                "namespace": execution_result.get("namespace", {})  # Variable state at failure
            })

            # If execution failed, prepare for retry
            if not execution_result["success"]:
                logger.error(f"[PythonCoderTool] [X] Execution failed: {execution_result.get('error', 'Unknown')[:100]}")
                logger.info(f"[PythonCoderTool] Error type: {error_type[0]} - {error_type[1][:80]}")
                
                if attempt < self.max_retry_attempts - 1:
                    # Check if same error type is repeating
                    prev_error_types = [h.get("error_type", ("Unknown", ""))[0] for h in attempt_history[:-1]]  # Exclude current
                    current_error_type = error_type[0]
                    same_error_count = prev_error_types.count(current_error_type)
                    force_different = same_error_count >= 1
                    
                    if force_different:
                        logger.warning(f"[PythonCoderTool] Same error type '{current_error_type}' repeated {same_error_count + 1} times - forcing different approach")
                    
                    # Use INCREMENTAL PATCHING for attempts 2+ (preserve working sections)
                    # BUT if same error keeps repeating, skip patching and do full regeneration
                    if attempt > 0 and code and not force_different:
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
                        # Full regeneration with attempt history context
                        if force_different:
                            logger.info("[PythonCoderTool] Forcing FULL REGENERATION with different approach...")
                        else:
                            logger.info("[PythonCoderTool] First attempt failed - will retry with full regeneration...")
                        
                        context = self._build_retry_context(
                            original_context=context,
                            error=execution_result.get("error"),
                            suggestions=issues,
                            attempt_history=attempt_history,
                            force_different_approach=force_different,
                            repeated_error_type=current_error_type if force_different else None
                        )
                        continue
                else:
                    logger.error(f"[PythonCoderTool] Max attempts ({self.max_retry_attempts}) reached")
                    break

            # Phase 3: Check output adequacy (1 LLM call, only if execution succeeded)
            logger.info("[PythonCoderTool] [OK] Execution succeeded, checking output adequacy...")

            # COMMENTED OUT: File content injection (now using stdout only)
            # Load files created by the code execution
            # execution_dir = Path(settings.python_code_execution_dir) / session_id if session_id else None
            # result_content = None
            # created_files = execution_result.get("created_files", [])

            # if execution_dir and execution_dir.exists() and created_files:
            #     result_content = self._load_result_files(
            #         execution_dir,
            #         created_files=created_files
            #     )

            # Use stdout output directly (no file concatenation)
            output_for_llm = execution_result.get("output", "")

            # Optional: Limit stdout to reasonable size for LLM
            max_output_chars = settings.python_code_output_max_llm_chars
            if len(output_for_llm) > max_output_chars:
                logger.info(f"[PythonCoderTool] Truncating stdout from {len(output_for_llm)} to {max_output_chars} chars")
                output_for_llm = output_for_llm[:max_output_chars] + f"\n... (truncated, showing {max_output_chars} of {len(output_for_llm)} chars)"

            is_adequate, adequacy_reason, suggestion = await self._check_output_adequacy(
                query=query,
                code=code,
                output=output_for_llm,
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
                    context = self._build_retry_context(
                        original_context=context,
                        error=None,
                        suggestions=[suggestion],
                        attempt_history=attempt_history
                    )
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

    def _load_result_files(
        self,
        execution_dir: Path,
        created_files: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Load contents of files created by generated code with size limit.

        Respects settings.python_code_output_max_llm_chars to prevent token overflow.

        Args:
            execution_dir: Path to execution directory
            created_files: List of filenames created during execution (from CodeExecutor)

        Returns:
            Combined content from all created files (truncated if needed), or None if no files found
        """
        if not created_files:
            logger.debug("[PythonCoderTool] No created files to load")
            return None

        content_parts = []
        total_chars = 0
        max_chars = settings.python_code_output_max_llm_chars

        for filename in created_files:
            file_path = execution_dir / filename
            if not file_path.exists():
                logger.warning(f"[PythonCoderTool] Created file not found: {filename}")
                continue

            # Check if we've reached the limit
            if total_chars >= max_chars:
                remaining_files = len(created_files) - len(content_parts)
                content_parts.append(f"\n... and {remaining_files} more file(s) (truncated due to size limit)")
                logger.info(f"[PythonCoderTool] Truncated output - reached {max_chars} char limit")
                break

            try:
                # Get file extension for type-specific handling
                file_ext = file_path.suffix.lower()
                file_size = file_path.stat().st_size

                # Handle different file types
                if file_ext in {'.txt', '.csv', '.json', '.md', '.log', '.tsv'}:
                    # Text-based files: Load content (with truncation)
                    content = file_path.read_text(encoding='utf-8')
                    file_header = f"=== {filename} ({file_size} bytes) ===\n"

                    # Calculate remaining space
                    remaining_chars = max_chars - total_chars - len(file_header)
                    if remaining_chars <= 0:
                        break

                    # Truncate content if needed
                    if len(content) > remaining_chars:
                        content = content[:remaining_chars] + f"\n... (truncated, showing {remaining_chars} of {len(content)} chars)"
                        logger.info(f"[PythonCoderTool] Truncated {filename} content")

                    content_parts.append(file_header + content)
                    total_chars += len(file_header) + len(content)
                    logger.info(f"[PythonCoderTool] Loaded text file: {filename} ({file_size} bytes, {len(content)} chars)")

                elif file_ext in {'.xlsx', '.xls', '.parquet', '.feather'}:
                    # Structured data files: Load with pandas (with truncation)
                    import pandas as pd

                    if file_ext in {'.xlsx', '.xls'}:
                        # Excel: Load all sheets
                        excel_file = pd.ExcelFile(file_path)
                        sheet_contents = []
                        for sheet_name in excel_file.sheet_names:
                            df = pd.read_excel(file_path, sheet_name=sheet_name)
                            sheet_contents.append(f"Sheet '{sheet_name}':\n{df.to_string()}")
                        content = "\n\n".join(sheet_contents)
                    elif file_ext == '.parquet':
                        df = pd.read_parquet(file_path)
                        content = df.to_string()
                    elif file_ext == '.feather':
                        df = pd.read_feather(file_path)
                        content = df.to_string()

                    file_header = f"=== {filename} ({file_size} bytes) ===\n"

                    # Calculate remaining space
                    remaining_chars = max_chars - total_chars - len(file_header)
                    if remaining_chars <= 0:
                        break

                    # Truncate content if needed
                    if len(content) > remaining_chars:
                        content = content[:remaining_chars] + f"\n... (truncated, showing {remaining_chars} of {len(content)} chars)"
                        logger.info(f"[PythonCoderTool] Truncated {filename} content")

                    content_parts.append(file_header + content)
                    total_chars += len(file_header) + len(content)
                    logger.info(f"[PythonCoderTool] Loaded structured data file: {filename} ({len(content)} chars)")

                elif file_ext in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}:
                    # Image files: Report metadata
                    try:
                        from PIL import Image
                        with Image.open(file_path) as img:
                            width, height = img.size
                            mode = img.mode
                            format_name = img.format
                        content = f"Image file: {format_name}, {width}x{height} pixels, mode={mode}, size={file_size} bytes"
                    except:
                        content = f"Image file: size={file_size} bytes (metadata unavailable)"

                    file_entry = f"=== {filename} ===\n{content}"
                    content_parts.append(file_entry)
                    total_chars += len(file_entry)
                    logger.info(f"[PythonCoderTool] Loaded image metadata: {filename}")

                elif file_ext == '.pdf':
                    # PDF files: Report basic info
                    content = f"PDF file: size={file_size} bytes"
                    file_entry = f"=== {filename} ===\n{content}"
                    content_parts.append(file_entry)
                    total_chars += len(file_entry)
                    logger.info(f"[PythonCoderTool] Loaded PDF metadata: {filename}")

                else:
                    # Other files: Report size only
                    content = f"Binary file: size={file_size} bytes"
                    file_entry = f"=== {filename} ===\n{content}"
                    content_parts.append(file_entry)
                    total_chars += len(file_entry)
                    logger.info(f"[PythonCoderTool] Loaded file metadata: {filename}")

            except Exception as e:
                logger.warning(f"[PythonCoderTool] Failed to load {filename}: {e}")
                error_entry = f"=== {filename} ===\nError loading file: {e}"
                content_parts.append(error_entry)
                total_chars += len(error_entry)

        if content_parts:
            result = "\n\n".join(content_parts)
            logger.info(f"[PythonCoderTool] Total output for LLM: {total_chars} chars (limit: {max_chars})")
            return result
        return None

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

    def _classify_error(
        self,
        error_message: str,
        return_code: int = -1,
        stdout: str = "",
        namespace: Dict = None
    ) -> Tuple[str, str]:
        """
        Classify error and return (error_type, specific_guidance) with enhanced context.

        This helps the LLM understand the nature of the error and provides
        actionable guidance for fixing it.

        Args:
            error_message: The error message from execution
            return_code: Process return code
            stdout: Standard output from execution
            namespace: Variable namespace (if available)

        Returns:
            Tuple of (error_type, guidance_text)
        """
        if not error_message or error_message.strip() == "":
            # Build context-aware guidance for silent failures
            clues = []

            if return_code == 1:
                clues.append("Return code 1 strongly suggests ImportError or ModuleNotFoundError")
            elif return_code == 2:
                clues.append("Return code 2 suggests invalid syntax or command invocation issue")
            elif return_code > 128:
                clues.append(f"Return code {return_code} indicates process was killed by signal")

            if stdout and len(stdout.strip()) < 50:
                clues.append(f"Minimal output ('{stdout.strip()[:30]}...') suggests code crashed very early")
            elif not stdout:
                clues.append("No output at all - code likely failed before any print statements")

            if namespace is not None and len(namespace) == 0:
                clues.append("No variables captured - code execution didn't reach variable assignments")

            guidance_parts = ["Code execution failed silently."]
            if clues:
                guidance_parts.append(" Context: " + "; ".join(clues) + ".")

            guidance_parts.append(" Common fixes: 1) Check all imports are installed (pip install <package>), 2) Verify code syntax, 3) Add try/except with print() to catch silent errors, 4) Check file paths exist.")

            return ("SilentFailure", "".join(guidance_parts))

        error_lower = error_message.lower()
        
        if "indexerror" in error_lower or "list index out of range" in error_lower:
            return ("IndexError", "Data structure is empty or smaller than expected. Check data loading first. Use len() to verify size before accessing indices.")
        elif "keyerror" in error_lower:
            return ("KeyError", "Dictionary key doesn't exist. Use .get('key', default) for safe access or print available keys first: print(data.keys())")
        elif "typeerror" in error_lower and "nonetype" in error_lower:
            return ("NoneType", "A variable is None when it shouldn't be. Check if data loading or a function call returned None.")
        elif "typeerror" in error_lower:
            return ("TypeError", "Wrong data type used. Check types with type() and convert if needed (str(), int(), list(), etc.)")
        elif "filenotfounderror" in error_lower or "no such file" in error_lower:
            return ("FileNotFound", "Check exact filename spelling and path. Use os.listdir('.') to see available files.")
        elif "json" in error_lower and ("decode" in error_lower or "expecting" in error_lower):
            return ("JSONDecode", "File is not valid JSON or has unexpected format. Check file content, encoding, or use try/except.")
        elif "valueerror" in error_lower:
            return ("ValueError", "Wrong value type or format. Check data types and content before operations.")
        elif "attributeerror" in error_lower:
            return ("AttributeError", "Object doesn't have this attribute/method. Check the object type and available methods.")
        elif "nameerror" in error_lower:
            return ("NameError", "Variable not defined. Check for typos or ensure variable is created before use.")
        elif "importerror" in error_lower or "modulenotfounderror" in error_lower:
            # Extract module name if possible for better guidance
            import re
            if "no module named" in error_lower:
                match = re.search(r"no module named ['\"]?(\w+)['\"]?", error_lower)
                if match:
                    module = match.group(1)
                    return ("ImportError", f"Module '{module}' not installed. Install with: pip install {module}. If it's a standard library, check Python version compatibility.")
            return ("ImportError", "Required module not available. Check module name, install missing dependency with pip, or use alternative approach.")
        elif "zerodivisionerror" in error_lower:
            return ("ZeroDivision", "Division by zero. Add check: if denominator != 0 before dividing.")
        elif "unicodedecodeerror" in error_lower or "codec" in error_lower:
            return ("EncodingError", "File encoding issue. Try: open(file, 'r', encoding='utf-8') or encoding='latin-1'")
        elif "permissionerror" in error_lower:
            return ("PermissionError", "No permission to access file. Check file path and permissions.")
        elif "memoryerror" in error_lower:
            return ("MemoryError", "Data too large. Process in chunks or use more efficient data structures.")
        elif "timeout" in error_lower:
            return ("Timeout", "Execution took too long. Optimize the algorithm or reduce data size.")
        else:
            return ("RuntimeError", "Unexpected error. Add print statements to debug intermediate values.")
    
    def _format_namespace_for_prompt(self, namespace: Dict[str, Any], max_items: int = 10) -> str:
        """
        Format namespace dict for inclusion in LLM prompt.
        
        Args:
            namespace: Variable namespace from execution
            max_items: Maximum number of variables to include
            
        Returns:
            Formatted string showing variable states
        """
        if not namespace:
            return "No variables captured."
        
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
                    cols_str = ", ".join(cols) + ("..." if len(info.get("columns", [])) > 5 else "")
                    lines.append(f"  - {name}: {var_type} (shape={shape}, columns=[{cols_str}])")
                else:
                    lines.append(f"  - {name}: {var_type} (shape={shape})")
            elif "length" in info:
                # List
                length = info.get("length", 0)
                if length == 0:
                    lines.append(f"  - {name}: {var_type} (EMPTY - length=0)")
                else:
                    lines.append(f"  - {name}: {var_type} (length={length})")
            elif "keys" in info:
                # Dict
                keys = info.get("keys", [])[:5]
                keys_str = ", ".join(f"'{k}'" for k in keys) + ("..." if len(info.get("keys", [])) > 5 else "")
                lines.append(f"  - {name}: {var_type} (keys=[{keys_str}])")
            elif "value" in info:
                # Simple type
                val = info.get("value", "")[:50]
                lines.append(f"  - {name}: {var_type} = {val}")
            else:
                lines.append(f"  - {name}: {var_type}")
        
        return "\n".join(lines) if lines else "No variables captured."

    def _build_retry_context(
        self,
        original_context: Optional[str],
        error: Optional[str],
        suggestions: List[str],
        attempt_history: Optional[List[Dict]] = None,
        force_different_approach: bool = False,
        repeated_error_type: Optional[str] = None
    ) -> str:
        """
        Build context for retry attempt with error/suggestion feedback.
        
        Enhanced to include previous attempt history and escalating guidance.
        
        Args:
            original_context: Original context string
            error: Current error message
            suggestions: List of suggestions for improvement
            attempt_history: List of previous failed attempts with code and errors
            force_different_approach: If True, strongly encourage different approach
            repeated_error_type: Error type that keeps repeating (if any)
        """
        parts = []

        if original_context:
            parts.append(original_context)

        # Add previous attempt history if available
        if attempt_history and len(attempt_history) > 0:
            parts.append("\n\n" + "="*60)
            parts.append("PREVIOUS FAILED ATTEMPTS (DO NOT REPEAT THESE MISTAKES)")
            parts.append("="*60)
            
            for prev in attempt_history:
                attempt_num = prev.get("attempt", "?")
                error_type = prev.get("error_type", ("Unknown", ""))[0]

                # Smart truncation: preserve error type at start and actual error at end
                error_msg = prev.get("error", "")
                if len(error_msg) > 800:
                    # Keep first 300 chars (error type + traceback start)
                    # Keep last 500 chars (actual error line and context)
                    error_msg = error_msg[:300] + "\n\n... [middle of traceback truncated] ...\n\n" + error_msg[-500:]

                prev_code = prev.get("code", "")
                namespace = prev.get("namespace", {})
                
                parts.append(f"\n--- Attempt {attempt_num} FAILED ---")
                parts.append(f"Error Type: {error_type}")
                parts.append(f"Error: {error_msg}")
                
                if prev_code:
                    # Show first 800 chars of code to avoid token explosion
                    code_preview = prev_code[:800]
                    if len(prev_code) > 800:
                        code_preview += "\n... (code truncated)"
                    parts.append(f"Code that failed:\n```python\n{code_preview}\n```")
                
                if namespace:
                    ns_formatted = self._format_namespace_for_prompt(namespace, max_items=5)
                    parts.append(f"Variables at failure:\n{ns_formatted}")
        
        # Add current error
        if error:
            parts.append(f"\n\nCurrent execution error:\n{error}")
        
        # Add suggestions
        if suggestions:
            parts.append(f"\n\nSuggestions for improvement:\n" + "\n".join(f"- {s}" for s in suggestions))
        
        # Add escalating guidance based on attempt count
        if attempt_history:
            attempt_count = len(attempt_history) + 1
            
            if force_different_approach or (repeated_error_type and attempt_count >= 2):
                parts.append(f"\n\n{'!'*60}")
                parts.append(f"CRITICAL: {repeated_error_type} error keeps repeating!")
                parts.append("You MUST try a FUNDAMENTALLY DIFFERENT approach:")
                parts.append("- If using pandas, try pure Python")
                parts.append("- If accessing data one way, try a completely different access pattern")
                parts.append("- Add debugging prints to understand the actual data structure")
                parts.append("- Consider the data might be in a different format than expected")
                parts.append("!"*60)
            elif attempt_count == 2:
                parts.append("\n\nNOTE: This is attempt 2. Try a DIFFERENT approach than before.")
            elif attempt_count >= 3:
                parts.append("\n\nWARNING: This is attempt 3+. Previous methods don't work. RETHINK the approach entirely.")

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
