"""
Code Fixer Module - Automatic and LLM-based Code Fixing

This module provides code fixing capabilities for Python code generation:
- Automatic fixes for common issues (generic filenames, imports, encoding, etc.)
- LLM-based fixes for execution errors

Extracted from python_coder_tool.py for better modularity.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage

from backend.config import prompts
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CodeFixer:
    """
    Handles automatic and LLM-based code fixing for Python code generation.

    Features:
    - Auto-fix common issues (generic filenames, missing imports, encoding, sys.argv)
    - LLM-based execution error fixing
    """

    def __init__(self, llm):
        """
        Initialize the code fixer.

        Args:
            llm: LangChain LLM instance for LLM-based fixing
        """
        self.llm = llm
        logger.info("[CodeFixer] Initialized")

    def auto_fix_common_issues(
        self,
        code: str,
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """
        Automatically fix common issues without LLM intervention.

        Args:
            code: Current Python code
            validated_files: Dict of validated files {path: filename}
            file_metadata: Metadata for files

        Returns:
            Tuple of (fixed_code, list of changes made)
        """
        fixed_code = code
        changes = []

        if not validated_files:
            return fixed_code, changes

        # Get list of correct filenames
        correct_filenames = list(validated_files.values())

        # Common wrong filenames to check
        wrong_patterns = [
            "file.json", "data.json", "input.json", "output.json",
            "file.csv", "data.csv", "input.csv", "output.csv",
            "file.xlsx", "data.xlsx", "input.xlsx", "output.xlsx",
            "file.txt", "data.txt", "input.txt", "output.txt"
        ]

        # Fix 1: Replace generic filenames with correct ones
        for wrong_name in wrong_patterns:
            if wrong_name in fixed_code:
                # Determine correct filename based on extension
                wrong_ext = Path(wrong_name).suffix
                matching_files = [f for f in correct_filenames if f.endswith(wrong_ext)]

                if matching_files:
                    correct_name = matching_files[0]  # Use first match
                    # Replace with correct filename
                    fixed_code = fixed_code.replace(f"'{wrong_name}'", f"'{correct_name}'")
                    fixed_code = fixed_code.replace(f'"{wrong_name}"', f'"{correct_name}"')
                    changes.append(f"Auto-fixed filename: '{wrong_name}' → '{correct_name}'")
                    logger.info(f"[AutoFix] Replaced '{wrong_name}' with '{correct_name}'")

        # Fix 2: Replace pd.read_json() with json.load() for complex nested JSON
        json_files = [
            (path, filename) for path, filename in validated_files.items()
            if file_metadata.get(path, {}).get('type') == 'json'
        ]

        for path, filename in json_files:
            metadata = file_metadata.get(path, {})
            max_depth = metadata.get('max_depth', 0)

            # If JSON has depth > 2, it's likely nested and not suitable for pd.read_json
            if max_depth > 2:
                # Check if code uses pd.read_json
                if f"pd.read_json('{filename}')" in fixed_code or f'pd.read_json("{filename}")' in fixed_code:
                    # Replace with json.load()
                    old_pattern_single = f"pd.read_json('{filename}')"
                    old_pattern_double = f'pd.read_json("{filename}")'
                    new_pattern = f"json.load(open('{filename}', 'r', encoding='utf-8'))"

                    if old_pattern_single in fixed_code:
                        fixed_code = fixed_code.replace(old_pattern_single, new_pattern)
                        changes.append(f"Auto-fixed: pd.read_json → json.load for nested JSON '{filename}'")
                        logger.info(f"[AutoFix] Replaced pd.read_json with json.load for '{filename}' (depth={max_depth})")
                    elif old_pattern_double in fixed_code:
                        fixed_code = fixed_code.replace(old_pattern_double, new_pattern)
                        changes.append(f"Auto-fixed: pd.read_json → json.load for nested JSON '{filename}'")
                        logger.info(f"[AutoFix] Replaced pd.read_json with json.load for '{filename}' (depth={max_depth})")

                    # Ensure json import is present
                    if "import json" not in fixed_code:
                        # Add import at the beginning
                        lines = fixed_code.split('\n')
                        # Find first non-comment, non-docstring line
                        insert_index = 0
                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                                insert_index = i
                                break
                        lines.insert(insert_index, "import json")
                        fixed_code = '\n'.join(lines)
                        changes.append("Auto-added: import json")
                        logger.info("[AutoFix] Added 'import json' statement")

        # Fix 3: Add missing imports
        import_checks = {
            'pandas': ['pd.', 'pandas.'],
            'numpy': ['np.', 'numpy.'],
            'json': ['json.load', 'json.dump', 'json.loads', 'json.dumps'],
            'openpyxl': ['.read_excel(', '.to_excel('],
            'docx': ['Document(', 'from docx'],
        }

        for module, patterns in import_checks.items():
            needs_import = any(pattern in fixed_code for pattern in patterns)
            has_import = f'import {module}' in fixed_code or f'from {module}' in fixed_code

            if needs_import and not has_import:
                # Add import at the beginning
                lines = fixed_code.split('\n')
                insert_index = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                        insert_index = i
                        break

                if module == 'pandas':
                    lines.insert(insert_index, 'import pandas as pd')
                    changes.append("Auto-added: import pandas as pd")
                elif module == 'numpy':
                    lines.insert(insert_index, 'import numpy as np')
                    changes.append("Auto-added: import numpy as np")
                else:
                    lines.insert(insert_index, f'import {module}')
                    changes.append(f"Auto-added: import {module}")

                fixed_code = '\n'.join(lines)
                logger.info(f"[AutoFix] Added missing import: {module}")

        # Fix 4: Add encoding to open() calls
        # Pattern: open('filename', 'r') or open("filename", "r")
        # Replace with: open('filename', 'r', encoding='utf-8')
        open_patterns = [
            (r"open\((['\"][^'\"]+['\"])\s*,\s*'r'\s*\)", r"open(\1, 'r', encoding='utf-8')"),
            (r'open\((["\'][^"\']+["\']),\s*"r"\s*\)', r'open(\1, "r", encoding="utf-8")'),
        ]

        for pattern, replacement in open_patterns:
            matches = re.findall(pattern, fixed_code)
            if matches:
                fixed_code = re.sub(pattern, replacement, fixed_code)
                changes.append(f"Auto-added: encoding='utf-8' to {len(matches)} open() call(s)")
                logger.info(f"[AutoFix] Added encoding='utf-8' to {len(matches)} open() call(s)")
                break  # Avoid duplicate logging

        # Fix 5: Remove sys.argv usage and replace with hardcoded filenames
        if 'sys.argv' in fixed_code:
            logger.info("[AutoFix] Detected sys.argv usage - attempting to fix with hardcoded filename")

            # Pattern 1: if len(sys.argv) > 1: main(sys.argv[1])
            # Replace with: main('filename.ext') where filename.ext is from validated_files
            if correct_filenames:
                # Use the first filename from validated files
                hardcoded_filename = correct_filenames[0]

                # Pattern: if len(sys.argv) > 1: followed by main(sys.argv[1]) or similar
                # Strategy: Replace the entire sys.argv check block with direct main() call

                # Check for common patterns
                patterns_to_fix = [
                    # Pattern 1: if len(sys.argv) > 1:\n    main(sys.argv[1])
                    (r'if\s+len\(sys\.argv\)\s*>\s*1\s*:\s*\n\s+(\w+)\(sys\.argv\[1\]\)',
                     rf'\1(\'{hardcoded_filename}\')'),

                    # Pattern 2: main(sys.argv[1]) without if check
                    (r'(\w+)\(sys\.argv\[1\]\)',
                     rf'\1(\'{hardcoded_filename}\')'),

                    # Pattern 3: filename = sys.argv[1]
                    (r'(\w+)\s*=\s*sys\.argv\[1\]',
                     rf'\1 = \'{hardcoded_filename}\''),

                    # Pattern 4: else: print("Usage: ...")
                    (r'else\s*:\s*\n\s+print\(["\']Usage:.*?\)',
                     ''),
                ]

                for pattern, replacement in patterns_to_fix:
                    if re.search(pattern, fixed_code):
                        fixed_code = re.sub(pattern, replacement, fixed_code, flags=re.MULTILINE)
                        changes.append(f"Auto-fixed: Replaced sys.argv usage with hardcoded filename '{hardcoded_filename}'")
                        logger.info(f"[AutoFix] Replaced sys.argv with hardcoded filename '{hardcoded_filename}'")
                        break

                # Remove import sys if no longer needed (after fixing sys.argv)
                if 'sys.argv' not in fixed_code and 'sys.executable' not in fixed_code:
                    # Check if sys is still used elsewhere
                    if not re.search(r'sys\.\w+', fixed_code):
                        fixed_code = re.sub(r'^\s*import\s+sys\s*$', '', fixed_code, flags=re.MULTILINE)
                        changes.append("Auto-removed: 'import sys' (no longer needed)")
                        logger.info("[AutoFix] Removed 'import sys' statement")

        return fixed_code, changes

    async def fix_execution_error(
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
        # Use centralized prompt from prompts module
        prompt = prompts.get_python_code_execution_fix_prompt(
            query=query,
            context=context,
            code=code,
            error_message=error_message
        )

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
            logger.info(f"[CodeFixer] Fixed code after execution error")

            return fixed_code, changes

        except Exception as e:
            logger.error(f"[CodeFixer] Failed to fix execution error: {e}")
            return code, []  # Return original code if fix fails
