"""
Code Sandbox
============
Secure execution environment for Python code.
Combines sandbox configuration, import validation, and execution logic.
"""

import os
import sys
import time
import uuid
import ast
import subprocess
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Security: Blocked imports
BLOCKED_IMPORTS = {
    "socket", "subprocess", "os.system", "eval", "exec",
    "__import__", "importlib", "shutil.rmtree", "pickle",
    "pty", "platform", "webbrowser"
}

# Supported file types
SUPPORTED_FILE_TYPES = {
    ".txt", ".md", ".log", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",
    ".xlsx", ".xls", ".pdf", ".parquet", ".feather", ".png", ".jpg", ".jpeg"
}

class CodeSandbox:
    """
    Executes Python code in a secured subprocess environment.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 512,
        execution_base_dir: str = "./data/scratch"
    ):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.execution_base_dir = Path(execution_base_dir).resolve()
        self.execution_base_dir.mkdir(parents=True, exist_ok=True)

    def _validate_characters(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate that code only contains valid Python characters.
        Blocks problematic Unicode characters that cause parsing issues.

        This is a simplified check that blocks known problematic characters.
        Unicode in strings/comments is generally safe and allowed by Python's parser.
        """
        issues = []

        # Problematic characters that cause parsing/security issues
        # These should NEVER appear in code, even in strings
        problematic = {
            0x00B7: 'MIDDLE DOT',            # Often mistaken for multiplication
            0x00D7: 'MULTIPLICATION SIGN',   # Often mistaken for 'x'
            0x2022: 'BULLET',                # Can cause issues
            0x2024: 'ONE DOT LEADER',        # Can cause issues
            0x2027: 'HYPHENATION POINT',     # Can cause issues
            0x00A0: 'NON-BREAKING SPACE',    # Invisible, causes weird errors
            0x2019: 'RIGHT SINGLE QUOTATION MARK',  # Looks like ' but isn't
            0x201C: 'LEFT DOUBLE QUOTATION MARK',   # Looks like " but isn't
            0x201D: 'RIGHT DOUBLE QUOTATION MARK',  # Looks like " but isn't
            0x2013: 'EN DASH',               # Can be mistaken for minus
            0x2014: 'EM DASH',               # Can be mistaken for minus
            0x2212: 'MINUS SIGN',            # Looks like - but isn't
        }

        for i, char in enumerate(code):
            char_code = ord(char)

            # Check for problematic characters
            if char_code in problematic:
                line_num = code[:i].count('\n') + 1
                col_num = i - code[:i].rfind('\n')
                char_name = problematic[char_code]
                issues.append(
                    f"Invalid character '{char}' (U+{char_code:04X} {char_name}) "
                    f"at line {line_num}, column {col_num}. "
                    f"Use standard ASCII equivalent instead."
                )

        return len(issues) == 0, issues

    def validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """Check code for unsafe imports and invalid characters using AST."""
        issues = []

        # First validate characters
        chars_valid, char_issues = self._validate_characters(code)
        if not chars_valid:
            issues.extend(char_issues)
            return False, issues

        # Then validate AST and imports
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] in BLOCKED_IMPORTS:
                            issues.append(f"Import '{alias.name}' is blocked")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] in BLOCKED_IMPORTS:
                        issues.append(f"Import from '{node.module}' is blocked")
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")

        return len(issues) == 0, issues

    def execute(
        self,
        code: str,
        input_files: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
        stage_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute code in isolated directory.
        """
        execution_id = session_id if session_id else uuid.uuid4().hex
        exec_dir = self.execution_base_dir / execution_id
        exec_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Check imports
            is_valid, issues = self.validate_imports(code)
            if not is_valid:
                return {"success": False, "error": f"Security violation: {'; '.join(issues)}"}

            # Setup files
            self._prepare_files(exec_dir, input_files)
            
            # Save script
            script_path = exec_dir / f"script_{stage_name or 'run'}.py"
            script_path.write_text(code, encoding='utf-8')
            
            # Snapshot files
            files_before = set(os.listdir(exec_dir))

            # Execute
            start_time = time.time()
            env = os.environ.copy()
            env["PYTHONPATH"] = os.getcwd() # Allow local imports if needed

            # Use 'python' to run the script
            process = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(exec_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env
            )
            duration = time.time() - start_time

            # Check created files
            files_after = set(os.listdir(exec_dir))
            created_files = list(files_after - files_before)
            created_files = [f for f in created_files if not f.startswith("script_") and f != "__pycache__"]

            success = process.returncode == 0
            error = process.stderr
            
            # Fallback: if failed but no stderr, check return code
            if not success and not error:
                error = f"Process exited with code {process.returncode}. Output: {process.stdout}"

            # Attempt variable capture on failure
            namespace = {}
            if not success:
                namespace = self._capture_variables(exec_dir, code)

            return {
                "success": success,
                "output": process.stdout,
                "error": error,
                "execution_time": duration,
                "created_files": created_files,
                "namespace": namespace
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Execution timed out after {self.timeout}s"}
        except Exception as e:
            return {"success": False, "error": f"Execution error: {str(e)}"}

    def _prepare_files(self, exec_dir: Path, input_files: Optional[Dict[str, str]]):
        """
        Copy input files to execution directory.

        Args:
            exec_dir: Execution directory
            input_files: Dict mapping temp file paths to original filenames
        """
        if not input_files:
            return
        for temp_path_str, original_filename in input_files.items():
            try:
                src = Path(temp_path_str)
                # Use original filename as destination
                dst = exec_dir / original_filename
                if src.exists():
                    # Copy file if not already there or source is newer
                    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(src, dst)
                        logger.debug(f"Copied {src.name} -> {original_filename}")
            except Exception as e:
                logger.warning(f"Failed to copy input file {original_filename}: {e}")

    def _capture_variables(self, exec_dir: Path, code: str) -> Dict[str, Any]:
        """Run introspection script to capture variables."""
        try:
            introspect_code = f"""
import json
import pandas as pd
import numpy as np

try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except:
    pass

data = {{}}
for k, v in list(locals().items()):
    if k.startswith('_') or k in ['pd', 'np', 'json']: continue
    try:
        t = type(v).__name__
        if t == 'DataFrame':
            data[k] = {{'type': 'DataFrame', 'shape': v.shape, 'columns': list(v.columns)}}
        elif t == 'ndarray':
            data[k] = {{'type': 'ndarray', 'shape': v.shape}}
        elif t in ['list', 'dict', 'str', 'int', 'float', 'bool']:
            data[k] = {{'type': t, 'value': str(v)[:100]}}
    except:
        pass
print('__VARS__')
print(json.dumps(data))
"""
            script_path = exec_dir / "introspect.py"
            script_path.write_text(introspect_code, encoding='utf-8')
            
            res = subprocess.run(
                [sys.executable, str(script_path)], cwd=str(exec_dir), 
                capture_output=True, text=True, timeout=5
            )
            
            if "__VARS__" in res.stdout:
                json_str = res.stdout.split("__VARS__")[1].strip()
                return json.loads(json_str)
        except Exception:
            pass
        return {}

code_sandbox = CodeSandbox()

