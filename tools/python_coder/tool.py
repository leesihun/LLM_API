"""
Python Code Executor Tool
Executes Python code in session scratch directory
"""
import os
import sys
import time
import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import config


def log_to_prompts_file(message: str):
    """Write message to prompts.log"""
    try:
        with open(config.PROMPTS_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"[WARNING] Failed to write to prompts.log: {e}")


class PythonCoderTool:
    """
    Execute Python code in session directory
    No sandboxing - raw execution with natural error handling
    """

    def __init__(self, session_id: str):
        """
        Initialize Python executor for a session

        Args:
            session_id: Session ID for workspace isolation
        """
        self.session_id = session_id
        self.workspace = config.PYTHON_WORKSPACE_DIR / session_id
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.timeout = config.PYTHON_EXECUTOR_TIMEOUT
        self.max_output_size = config.PYTHON_EXECUTOR_MAX_OUTPUT_SIZE

    def _generate_script_name(self, code: str) -> str:
        """
        Generate human-readable script name from code content

        Args:
            code: Python code

        Returns:
            Descriptive script name (e.g., "calc_factorial.py", "fibonacci.py")
        """
        # Try to extract meaningful name from code
        name_parts = []

        # Check for function/class definitions (first priority - more descriptive)
        func_matches = re.findall(r'^\s*def\s+(\w+)', code, re.MULTILINE)
        class_matches = re.findall(r'^\s*class\s+(\w+)', code, re.MULTILINE)

        if func_matches:
            name_parts.append(func_matches[0])
        elif class_matches:
            name_parts.append(class_matches[0])

        # Check for imports (secondary - add context)
        import_matches = re.findall(r'^\s*import\s+(\w+)', code, re.MULTILINE)
        from_matches = re.findall(r'^\s*from\s+(\w+)', code, re.MULTILINE)
        imports = import_matches + from_matches

        if imports and not name_parts:
            # Only use import name if no function/class found
            # Add "_script" suffix to avoid module name collision
            name_parts.append(imports[0] + "_script")

        # Generate name
        if name_parts:
            # Join with underscore, limit length
            name = "_".join(name_parts[:2])  # Max 2 parts
            name = name[:50]  # Max 50 chars
            # Sanitize
            name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            return f"{name}.py"

        # Fallback to timestamp
        return f"script_{int(time.time() * 1000)}.py"

    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code (raw, no sandboxing)

        Args:
            code: Python code to execute
            timeout: Execution timeout (optional)
            context: Additional context (for logging/tracking)

        Returns:
            Execution result dictionary
        """
        exec_timeout = timeout or self.timeout
        start_time = time.time()

        # Smart edit: Check if we should merge with existing .py files
        existing_py_files = [f for f in self.list_files() if f.endswith('.py')]

        if existing_py_files and config.PYTHON_CODER_SMART_EDIT:
            # Let LLM decide how to handle code with workspace context
            final_code, script_name = self._smart_code_generation(code, existing_py_files)
        else:
            # No existing .py files or smart edit disabled - use code as-is
            final_code = code
            script_name = self._generate_script_name(code)

        script_path = self.workspace / script_name

        # Log to prompts.log
        log_to_prompts_file("\n\n")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"TOOL EXECUTION: python_coder")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_to_prompts_file(f"Session ID: {self.session_id}")
        log_to_prompts_file(f"Workspace: {self.workspace}")
        log_to_prompts_file(f"Script: {script_name}")
        log_to_prompts_file(f"Code Length: {len(final_code)} chars")
        log_to_prompts_file(f"Timeout: {exec_timeout}s")
        if existing_py_files and config.PYTHON_CODER_SMART_EDIT:
            log_to_prompts_file(f"Smart Edit: Enabled (existing .py files: {existing_py_files})")
        log_to_prompts_file(f"")
        log_to_prompts_file(f"CODE:")
        for line in final_code.split('\n'):
            log_to_prompts_file(f"  {line}")

        # Console logging
        print("\n" + "=" * 80)
        print("[PYTHON TOOL] execute() called")
        print("=" * 80)
        print(f"Session ID: {self.session_id}")
        print(f"Workspace: {self.workspace}")
        print(f"Script: {script_name}")
        print(f"Code length: {len(final_code)} chars")
        print(f"Timeout: {exec_timeout}s")
        if existing_py_files and config.PYTHON_CODER_SMART_EDIT:
            print(f"Smart Edit: Enabled (merged with existing files)")

        # Write code to file
        print(f"\n[PYTHON] Writing script to disk...")
        script_path.write_text(final_code, encoding='utf-8')
        print(f"[PYTHON] [OK] Script written: {script_path}")

        try:
            # Execute with cwd set to workspace
            print(f"\n[PYTHON] Executing...")
            print(f"  Python: {sys.executable}")
            print(f"  Script: {script_name}")
            print(f"  Working dir: {self.workspace}")

            result = subprocess.run(
                [sys.executable, script_name],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=exec_timeout
            )

            print(f"[PYTHON] [OK] Subprocess completed")

            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode

            # Limit output size
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n... (output truncated)"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n... (output truncated)"

            execution_time = time.time() - start_time
            files = self._get_workspace_files()
            success = returncode == 0

            # Console logging
            print(f"\n[PYTHON] Execution completed in {execution_time:.2f}s")
            print(f"[PYTHON] Return code: {returncode}")
            if stdout:
                stdout_preview = stdout[:300] + "..." if len(stdout) > 300 else stdout
                print(f"[PYTHON] STDOUT:\n{stdout_preview}")
            if stderr:
                stderr_preview = stderr[:300] + "..." if len(stderr) > 300 else stderr
                print(f"[PYTHON] STDERR:\n{stderr_preview}")
            if files:
                print(f"[PYTHON] Files in workspace: {list(files.keys())}")

            # Log to prompts.log
            log_to_prompts_file(f"")
            log_to_prompts_file(f"OUTPUT:")
            log_to_prompts_file(f"  Status: {'SUCCESS' if success else 'FAILED'}")
            log_to_prompts_file(f"  Return Code: {returncode}")
            log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")
            if stdout:
                log_to_prompts_file(f"")
                log_to_prompts_file(f"STDOUT:")
                for line in stdout.split('\n'):
                    log_to_prompts_file(f"  {line}")
            if stderr:
                log_to_prompts_file(f"")
                log_to_prompts_file(f"STDERR:")
                for line in stderr.split('\n'):
                    log_to_prompts_file(f"  {line}")
            if files:
                log_to_prompts_file(f"")
                log_to_prompts_file(f"FILES:")
                for filename, meta in files.items():
                    log_to_prompts_file(f"  {filename} ({meta['size']} bytes)")
            log_to_prompts_file(f"")
            log_to_prompts_file("=" * 80)

            return {
                "success": success,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": returncode,
                "execution_time": execution_time,
                "files": files,
                "workspace": str(self.workspace),
                "error": None if success else stderr
            }

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            error_msg = f"Execution timeout after {exec_timeout} seconds"

            print(f"\n[PYTHON] ERROR: {error_msg}")

            log_to_prompts_file(f"")
            log_to_prompts_file(f"OUTPUT:")
            log_to_prompts_file(f"  Status: TIMEOUT")
            log_to_prompts_file(f"  Error: {error_msg}")
            log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")
            log_to_prompts_file(f"")
            log_to_prompts_file("=" * 80)

            return {
                "success": False,
                "stdout": "",
                "stderr": error_msg,
                "returncode": -1,
                "execution_time": execution_time,
                "files": self._get_workspace_files(),
                "workspace": str(self.workspace),
                "error": error_msg
            }

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            print(f"\n[PYTHON] ERROR: {error_msg}")

            log_to_prompts_file(f"")
            log_to_prompts_file(f"OUTPUT:")
            log_to_prompts_file(f"  Status: ERROR")
            log_to_prompts_file(f"  Error: {error_msg}")
            log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")
            log_to_prompts_file(f"")
            log_to_prompts_file("=" * 80)

            return {
                "success": False,
                "stdout": "",
                "stderr": error_msg,
                "returncode": -1,
                "execution_time": execution_time,
                "files": self._get_workspace_files(),
                "workspace": str(self.workspace),
                "error": error_msg
            }

    def _get_workspace_files(self) -> Dict[str, Any]:
        """
        Get list of files in workspace

        Returns:
            Dictionary of files with metadata
        """
        files = {}

        for file_path in self.workspace.iterdir():
            if file_path.is_file():
                rel_path = file_path.name
                files[rel_path] = {
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                    "path": str(file_path)
                }

        return files

    def read_file(self, filename: str) -> Optional[str]:
        """
        Read a file from workspace

        Args:
            filename: File name

        Returns:
            File contents or None
        """
        file_path = self.workspace / filename

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

    def list_files(self) -> List[str]:
        """
        List all files in workspace

        Returns:
            List of file names
        """
        return [f.name for f in self.workspace.iterdir() if f.is_file()]

    def clear_workspace(self):
        """Clear all files in workspace"""
        import shutil
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _smart_code_generation(self, code: str, existing_py_files: List[str]) -> Tuple[str, str]:
        """
        Use LLM to decide whether to merge with existing code or create new file

        Args:
            code: New Python code to execute
            existing_py_files: List of existing .py files in workspace

        Returns:
            Tuple of (final_code, script_name)
        """
        # Build context of existing files
        files_context = ""
        for f in existing_py_files:
            content = self.read_file(f)
            if content:
                files_context += f"\n### {f}\n```python\n{content}\n```\n"

        # Prompt LLM to generate context-aware code
        prompt = f"""Python workspace has existing files. Generate executable code.

**Existing files:**{files_context}

**New code to execute:**
```python
{code}
```

**Instructions:**
- If new code builds on existing variables/dataframes → edit/merge with existing file
- If new code is independent task → output as-is (new file will be created)
- Maintain all necessary functionality and variable continuity
- Remove duplicate imports
- Keep the code clean and well-organized

**Output ONLY the complete Python code to execute:**
```python"""

        print(f"\n[PYTHON SMART EDIT] Calling LLM for context-aware code generation...")
        print(f"[PYTHON SMART EDIT] Existing files: {existing_py_files}")

        try:
            final_code = self._llm_call(prompt)

            # Generate script name from the final code
            script_name = self._generate_script_name(final_code)

            print(f"[PYTHON SMART EDIT] Target file: {script_name}")

            return final_code, script_name

        except Exception as e:
            # LLM call failed - fall back to using original code without merging
            print(f"[PYTHON SMART EDIT] Smart edit failed, using original code: {e}")
            script_name = self._generate_script_name(code)
            return code, script_name

    def _llm_call(self, prompt: str) -> str:
        """
        Call LLM backend for code generation

        Args:
            prompt: Prompt for code generation

        Returns:
            Generated Python code
        """
        try:
            from backend.core.llm_backend import LLMBackend

            llm = LLMBackend()

            # Use python_coder tool configuration
            model = config.TOOL_MODELS.get('python_coder', config.OLLAMA_MODEL)
            temperature = config.TOOL_PARAMETERS.get('python_coder', {}).get('temperature', config.DEFAULT_TEMPERATURE)
            max_tokens = config.TOOL_PARAMETERS.get('python_coder', {}).get('max_tokens', config.DEFAULT_MAX_TOKENS)

            print(f"[PYTHON SMART EDIT] LLM model: {model}, temperature: {temperature}, max_tokens: {max_tokens}")

            response = llm.generate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract Python code from response
            code = self._extract_python_code(response)

            print(f"[PYTHON SMART EDIT] Generated code length: {len(code)} chars")

            return code

        except Exception as e:
            print(f"[PYTHON SMART EDIT] ERROR: LLM call failed: {e}")
            print(f"[PYTHON SMART EDIT] Falling back to original code (no merge)")
            # Return empty string to signal failure
            raise

    def _extract_python_code(self, response: str) -> str:
        """
        Extract Python code from LLM response

        Args:
            response: LLM response text

        Returns:
            Extracted Python code
        """
        # Try to extract from markdown code blocks
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
            else:
                code = response.strip()
        elif "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                code = parts[1].strip()
            else:
                code = response.strip()
        else:
            code = response.strip()

        # Validate it looks like Python code (basic sanity check)
        if not code or len(code) < 10:
            print(f"[PYTHON SMART EDIT] WARNING: Extracted code is very short ({len(code)} chars)")

        return code
