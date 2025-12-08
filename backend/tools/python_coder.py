"""
Python Coder Tool
=================
LLM-powered Python code generation and execution tool.

Provides intelligent code generation and execution in the sandbox with a
single-shot flow; retries and error recovery are handled by the agent loop.

Features:
- LLM-based code generation from task descriptions
- Secure execution in sandbox environment
- Observation generation for agent integration
- File context awareness (CSV, Excel, JSON, etc.)
- Session persistence for variables and files

Version: 1.1.0
Created: 2025-12-05
"""

import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from langchain_core.messages import HumanMessage

from backend.core.base_tool import BaseTool
from backend.core.result_types import ToolResult
from backend.config.settings import settings
from backend.utils.logging_utils import get_logger
from backend.tools.code_sandbox import CodeSandbox, SandboxManager, ExecutionResult
from backend.runtime import extract_original_filename
from backend.tools.file_analyzer import file_analyzer

logger = get_logger(__name__)


# ============================================================================
# Prompt Templates
# ============================================================================

CODE_GENERATION_PROMPT = """You are an expert Python programmer. Generate Python code to accomplish the following task.

## Task
{task}

## Available Files
{file_context}

## Context
{additional_context}

## Requirements
1. Write a clean, efficient Python code that accomplishes the given Task.
2. When asked, use matplotlib for visualizations (save to file, don't use plt.show())
3. Print results that should be visible to the user, otherwise, you may write a file.
4. Always use the exact filenames of the files provided to you. Do not create new files.

## CRITICAL INSTRUCTIONS
- Output ONLY Python code, absolutely NO explanations before or after
- Do NOT use markdown code blocks (no ```python ... ```)
- Start immediately with import statements or code
- Use print() for all outputs that should be visible if not viable, you may write a file.
- Save visualizations with plt.savefig('output.png') and print the filename

Code:
"""

CODE_FIX_PROMPT = """You are an expert Python programmer. The following code failed with an error. Fix the code.

## Original Task
{task}

## Failed Code
```python
{code}
```

## Error
{error}

## Available Files
{file_context}

## CRITICAL INSTRUCTIONS
1. Analyze the error and fix the root cause
2. Fix the error while maintaining the original intent
3. Make the code more robust and efficient
4. Output ONLY the fixed Python code, absolutely NO explanations before or after
5. Do NOT use markdown code blocks (no ```python ... ```)
6. Start directly with the fixed code
7. Always use the exact filenames of the files provided to you. Do not create new files.

Fixed Code:"""

OBSERVATION_PROMPT = """Based on the code execution results, generate a concise observation for the agent.

## Task
{task}

## Executed Code
```python
{code}
```

## Execution Output
{output}

## Created Files
{created_files}

## Instructions
Write a comprehensive observation that includes:
1. What the code did, and what the results are.
2. Key results or findings
3. Any files created

Keep it factual and concise. This will be used by the agent to decide next steps.

## Observation:
"""


# ============================================================================
# Python Coder Tool
# ============================================================================

class PythonCoderTool(BaseTool):
    """
    LLM-powered Python code generation and execution tool.
    
    This tool:
    1. Takes a task description from the agent
    2. Generates Python code using LLM
    3. Executes code in a secure sandbox
    4. If errors occur, uses LLM to fix and retry
    5. Generates observations for the agent
    
    Example:
        >>> tool = PythonCoderTool()
        >>> result = await tool.execute(
        ...     query="Load the CSV file and show basic statistics",
        ...     file_paths=["data/sales.csv"],
        ...     session_id="test-session"
        ... )
        >>> print(result.output)
    """
    
    def __init__(self):
        """Initialize the Python coder tool."""
        super().__init__()
        self.max_iterations = settings.python_code_max_iterations
        self._coder_llm = None
        
    def _get_coder_llm(self, user_id: str = "default"):
        """
        Get or create coder LLM instance.
        
        Args:
            user_id: User ID for LLM configuration
            
        Returns:
            Configured LLM for code generation
        """
        if self._coder_llm is None:
            from backend.utils.llm_factory import LLMFactory
            self._coder_llm = LLMFactory.create_coder_llm(user_id=user_id)
            logger.debug("[PythonCoder] Created coder LLM instance")
        return self._coder_llm
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate tool inputs.
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            True if valid, False otherwise
        """
        query = kwargs.get("query", "")
        return bool(query and query.strip())
    
    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        user_id: str = "default",
        stage_prefix: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        plan_context: Optional[Dict] = None,
        react_context: Optional[Dict] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute the Python coder tool.
        
        Args:
            query: Task description (what the code should do)
            context: Optional additional context
            file_paths: List of available file paths
            session_id: Session ID for sandbox persistence
            user_id: User ID for LLM configuration
            stage_prefix: Optional prefix for file naming
            conversation_history: Optional conversation history
            plan_context: Optional plan execution context
            react_context: Optional ReAct agent context
            
        Returns:
            ToolResult with execution results
        """
        self._start_timer()
        self._log_execution_start(
            query=query[:100],
            file_count=len(file_paths) if file_paths else 0,
            session_id=session_id
        )
        
        # Validate inputs
        if not self.validate_inputs(query=query):
            return self._handle_validation_error(
                "Task description (query) is required",
                parameter="query"
            )
        
        try:
            # Setup session
            session_id = session_id or f"session_{int(self._start_time)}"
            sandbox = SandboxManager.get_sandbox(session_id)
            
            # Build file context
            file_context = self._build_file_context(file_paths)
            
            # Build additional context
            additional_context = self._build_additional_context(
                context, conversation_history, plan_context, react_context
            )
            
            # Copy files to sandbox working directory
            if file_paths:
                self._copy_files_to_sandbox(file_paths, sandbox)
            
            # Generate and execute code with retry loop
            result = await self._generate_and_execute(
                task=query,
                file_context=file_context,
                additional_context=additional_context,
                sandbox=sandbox,
                user_id=user_id
            )
            
            self._log_execution_end(result)
            return result
            
        except Exception as e:
            return self._handle_error(e, "execute")
    
    async def _generate_and_execute(
        self,
        task: str,
        file_context: str,
        additional_context: str,
        sandbox: CodeSandbox,
        user_id: str
    ) -> ToolResult:
        """
        Generate code, execute it, and fix errors iteratively.
        
        Args:
            task: Task description
            file_context: Information about available files
            additional_context: Additional context for code generation
            sandbox: Sandbox instance for execution
            user_id: User ID for LLM
            
        Returns:
            ToolResult with execution results
        """
        llm = self._get_coder_llm(user_id)
        attempt_history = []
        code = None
        last_error = None
        
        for attempt in range(1, self.max_iterations + 1):
            logger.info(f"[PythonCoder] Attempt {attempt}/{self.max_iterations}")
            
            # Generate or fix code
            if attempt == 1:
                code = await self._generate_code(
                    llm, task, file_context, additional_context
                )
            else:
                code = await self._fix_code(
                    llm, task, code, last_error, file_context
                )
            
            logger.debug(f"[PythonCoder] Generated code ({len(code)} chars):\n{code[:500]}...")

            # Validate code length
            if len(code) > 50000:  # ~1000 lines at 50 chars/line
                logger.warning(f"[PythonCoder] Code is very long ({len(code)} chars), truncating...")
                code = code[:50000]

            if not code or len(code.strip()) < 5:
                logger.error(f"[PythonCoder] Generated code is empty or too short")
                last_error = "Generated code is empty or too short"
                continue

            # Execute code in sandbox
            exec_result = sandbox.execute(code)
            
            attempt_history.append({
                "attempt": attempt,
                "code": code[:1000],  # Truncate for history
                "success": exec_result.success,
                "output": exec_result.output[:500] if exec_result.output else "",
                "error": exec_result.error[:500] if exec_result.error else ""
            })
            
            if exec_result.success:
                # Generate observation
                observation = await self._generate_observation(
                    llm, task, code, exec_result
                )
                
                return ToolResult.success_result(
                    output={
                        "output": exec_result.output,
                        "observation": observation,
                        "code": code,
                        "created_files": exec_result.created_files,
                        "attempts": attempt,
                        "variables": list(exec_result.variables.keys())
                    },
                    metadata={
                        "session_id": sandbox.session_id,
                        "working_dir": str(sandbox.working_dir),
                        "attempt_history": attempt_history
                    },
                    execution_time=self._elapsed_time()
                )
            else:
                last_error = exec_result.error
                logger.warning(f"[PythonCoder] Attempt {attempt} failed: {last_error[:200]}")
        
        # All attempts failed
        return ToolResult.failure_result(
            error=f"Code execution failed after {self.max_iterations} attempts. Last error: {last_error}",
            error_type="CodeExecutionError",
            metadata={
                "attempt_history": attempt_history,
                "last_code": code,
                "last_error": last_error
            },
            execution_time=self._elapsed_time()
        )
    
    async def _generate_code(
        self,
        llm,
        task: str,
        file_context: str,
        additional_context: str
    ) -> str:
        """
        Generate Python code from task description.
        
        Args:
            llm: LLM instance
            task: Task description
            file_context: File information
            additional_context: Additional context
            
        Returns:
            Generated Python code
        """
        prompt = CODE_GENERATION_PROMPT.format(
            task=task,
            file_context=file_context or "No files provided",
            additional_context=additional_context or "No additional context"
        )
        
        logger.debug(f"[PythonCoder] Code generation prompt ({len(prompt)} chars)")
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        code = self._extract_code(response.content)
        
        return code
    
    async def _fix_code(
        self,
        llm,
        task: str,
        code: str,
        error: str,
        file_context: str
    ) -> str:
        """
        Fix code that failed execution.
        
        Args:
            llm: LLM instance
            task: Original task description
            code: Failed code
            error: Error message
            file_context: File information
            
        Returns:
            Fixed Python code
        """
        prompt = CODE_FIX_PROMPT.format(
            task=task,
            code=code,
            error=error,
            file_context=file_context or "No files provided"
        )
        
        logger.debug(f"[PythonCoder] Code fix prompt ({len(prompt)} chars)")
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        fixed_code = self._extract_code(response.content)
        
        return fixed_code
    
    async def _generate_observation(
        self,
        llm,
        task: str,
        code: str,
        exec_result: ExecutionResult
    ) -> str:
        """
        Generate observation from execution results.
        
        Args:
            llm: LLM instance
            task: Original task
            code: Executed code
            exec_result: Execution result
            
        Returns:
            Observation string
        """
        # Format created files
        created_files_str = ", ".join(exec_result.created_files) if exec_result.created_files else "None"
        
        prompt = OBSERVATION_PROMPT.format(
            task=task,
            code=code[:2000],  # Truncate long code
            output=exec_result.output[:5000] if exec_result.output else "No output",
            created_files=created_files_str
        )
        
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            observation = response.content.strip()
            
            # Fallback if LLM returns empty
            if not observation:
                observation = self._generate_fallback_observation(exec_result)
                
            return observation
            
        except Exception as e:
            logger.warning(f"[PythonCoder] Observation generation failed: {e}")
            return self._generate_fallback_observation(exec_result)
    
    def _generate_fallback_observation(self, exec_result: ExecutionResult) -> str:
        """
        Generate fallback observation when LLM fails.
        
        Args:
            exec_result: Execution result
            
        Returns:
            Fallback observation string
        """
        parts = ["Code executed successfully."]
        
        if exec_result.output:
            # Extract first meaningful line
            lines = [l for l in exec_result.output.split('\n') if l.strip()]
            if lines:
                parts.append(f"Output: {lines[0][:100]}")
                
        if exec_result.created_files:
            parts.append(f"Created files: {', '.join(exec_result.created_files)}")
            
        if exec_result.variables:
            vars_str = ", ".join(list(exec_result.variables.keys())[:5])
            parts.append(f"Variables: {vars_str}")
            
        return " ".join(parts)
    
    def _extract_code(self, response: str) -> str:
        """
        Extract Python code from LLM response.

        Handles various formats:
        - Raw code
        - Markdown code blocks (```python ... ```)
        - Code with explanations

        Args:
            response: LLM response text

        Returns:
            Extracted Python code
        """
        if not response:
            return ""

        # Try to extract from markdown code block (prefer this)
        code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            # Return the first code block
            code = matches[0].strip()
            logger.debug(f"[PythonCoder] Extracted code from markdown block ({len(code)} chars)")
            return code

        # Check if response starts with common code patterns
        code_starters = ['import ', 'from ', 'def ', 'class ', '#', 'print(', 'with ', 'for ', 'if ', 'while ']
        lines = response.split('\n')

        # Find where code starts
        code_start = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(stripped.startswith(s) for s in code_starters):
                code_start = i
                break

        # If we found code start, extract from there
        if code_start >= 0:
            # Find where code ends (look for explanatory text after code)
            code_end = len(lines)
            for i in range(code_start, len(lines)):
                line = lines[i].strip()
                # Stop if we hit explanatory text (sentences, not code)
                if line and not line.startswith('#') and ': ' in line and not any(c in line for c in ['=', '(', '[', '{']):
                    # Might be explanation like "This code does..."
                    if i > code_start + 3:  # Give it at least 3 lines
                        code_end = i
                        break

            code = '\n'.join(lines[code_start:code_end]).strip()
            logger.debug(f"[PythonCoder] Extracted code from detected start ({len(code)} chars)")
            return code

        # Fallback: return entire response and let validation catch errors
        logger.warning(f"[PythonCoder] Could not detect code pattern, returning full response")
        return response.strip()
    
    def _build_file_context(self, file_paths: Optional[List[str]]) -> str:
        """
        Build file context string from file paths.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Formatted file context string
        """
        if not file_paths:
            return "No files provided"
        
        try:
            # Use file_analyzer for detailed context
            analysis = file_analyzer.analyze(file_paths, quick_mode=True)
            
            if analysis.get('success'):
                return analysis.get('llm_context', analysis.get('summary', ''))
            else:
                # Fallback to simple listing
                return f"Files: {', '.join(file_paths)}"
                
        except Exception as e:
            logger.warning(f"[PythonCoder] File analysis failed: {e}")
            return f"Files: {', '.join(file_paths)}"
    
    def _build_additional_context(
        self,
        context: Optional[str],
        conversation_history: Optional[List[Dict]],
        plan_context: Optional[Dict],
        react_context: Optional[Dict]
    ) -> str:
        """
        Build additional context from various sources.
        
        Args:
            context: Direct context string
            conversation_history: Past conversation messages
            plan_context: Plan execution context
            react_context: ReAct agent context
            
        Returns:
            Combined context string
        """
        parts = []
        
        if context:
            parts.append(f"Context: {context}")
        
        if conversation_history:
            # Extract recent relevant messages
            recent = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            history_str = "\n".join(
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:200]}"
                for msg in recent
            )
            parts.append(f"Recent conversation:\n{history_str}")
        
        if plan_context:
            step = plan_context.get('current_step', '?')
            total = plan_context.get('total_steps', '?')
            parts.append(f"Plan: Step {step}/{total}")
        
        if react_context:
            prev_steps = react_context.get('previous_steps', [])
            if prev_steps:
                steps_str = "\n".join(
                    f"- Step {s.get('step')}: {s.get('action')} -> {s.get('observation', '')[:100]}"
                    for s in prev_steps[-3:]
                )
                parts.append(f"Previous steps:\n{steps_str}")
        
        return "\n\n".join(parts) if parts else ""
    
    def _copy_files_to_sandbox(
        self,
        file_paths: List[str],
        sandbox: CodeSandbox
    ) -> None:
        """
        Copy input files to sandbox working directory.
        
        Args:
            file_paths: Source file paths
            sandbox: Target sandbox
        """
        import shutil
        
        for file_path in file_paths:
            src = Path(file_path)
            if src.exists():
                try:
                    original_name = extract_original_filename(src.name)
                except Exception:
                    original_name = src.name

                dst = sandbox.working_dir / original_name
                try:
                    shutil.copy2(src, dst)
                    logger.debug(f"[PythonCoder] Copied {src.name} to sandbox as {original_name}")
                except Exception as e:
                    logger.warning(f"[PythonCoder] Failed to copy {src.name}: {e}")
    
    async def execute_code_task(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        user_id: str = "default",
        stage_prefix: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        plan_context: Optional[Dict] = None,
        react_context: Optional[Dict] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute a code generation task (alias for execute() for backwards compatibility).

        This method is called by react_agent and other components that expect
        the execute_code_task interface.

        Args:
            query: Task description (what the code should do)
            context: Optional additional context
            file_paths: List of available file paths
            session_id: Session ID for sandbox persistence
            user_id: User ID for LLM configuration
            stage_prefix: Optional prefix for file naming
            conversation_history: Optional conversation history
            plan_context: Optional plan execution context
            react_context: Optional ReAct agent context

        Returns:
            ToolResult with execution results
        """
        return await self.execute(
            query=query,
            context=context,
            file_paths=file_paths,
            session_id=session_id,
            user_id=user_id,
            stage_prefix=stage_prefix,
            conversation_history=conversation_history,
            plan_context=plan_context,
            react_context=react_context,
            **kwargs
        )

    async def execute_code(
        self,
        code: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute pre-written code directly (without LLM generation).

        This is a simpler interface for testing or direct code execution.

        Args:
            code: Python code to execute
            session_id: Optional session ID

        Returns:
            Dict with execution results
        """
        session_id = session_id or "direct_exec"
        sandbox = SandboxManager.get_sandbox(session_id)

        result = sandbox.execute(code)

        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "created_files": result.created_files,
            "variables": list(result.variables.keys())
        }


# ============================================================================
# Global Tool Instance
# ============================================================================

# Create singleton instance
python_coder_tool = PythonCoderTool()

