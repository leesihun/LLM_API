"""
Auto-Fixer Pre-processor - Automatic Code Fixes

This module provides automatic fixes for common issues without LLM intervention.
Runs BEFORE verification to catch and fix common errors automatically.

Features:
- Fix generic filenames (file.json -> actual_file.json)
- Add missing imports (pandas, numpy, json, etc.)
- Fix encoding issues in open() calls
- Replace pd.read_json() with json.load() for nested JSON
- Remove sys.argv usage and replace with hardcoded filenames
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AutoFixer:
    """
    Automatic code fixer for common issues.

    Runs BEFORE verification to automatically fix:
    - Generic filenames -> actual filenames
    - Missing imports
    - Encoding issues
    - pd.read_json() for nested JSON -> json.load()
    - sys.argv usage -> hardcoded filenames
    """

    def __init__(self):
        """Initialize the auto-fixer."""
        logger.info("[AutoFixer] Initialized")

    def fix_filename_references(
        self,
        code: str,
        validated_files: Dict[str, str]
    ) -> Tuple[str, List[str]]:
        """
        Fix generic filename references with actual filenames.

        Args:
            code: Python code
            validated_files: Dict mapping paths to actual filenames

        Returns:
            Tuple of (fixed_code, list of changes)
        """
        if not validated_files:
            return code, []

        fixed_code = code
        changes = []
        correct_filenames = list(validated_files.values())

        # Common wrong filename patterns
        wrong_patterns = [
            "file.json", "data.json", "input.json", "output.json",
            "file.csv", "data.csv", "input.csv", "output.csv",
            "file.xlsx", "data.xlsx", "input.xlsx", "output.xlsx",
            "file.txt", "data.txt", "input.txt", "output.txt"
        ]

        for wrong_name in wrong_patterns:
            if wrong_name in fixed_code:
                # Find correct filename by extension
                wrong_ext = Path(wrong_name).suffix
                matching = [f for f in correct_filenames if f.endswith(wrong_ext)]

                if matching:
                    correct_name = matching[0]
                    fixed_code = fixed_code.replace(f"'{wrong_name}'", f"'{correct_name}'")
                    fixed_code = fixed_code.replace(f'"{wrong_name}"', f'"{correct_name}"')
                    changes.append(f"Fixed filename: '{wrong_name}' → '{correct_name}'")
                    logger.info(f"[AutoFix] Replaced '{wrong_name}' with '{correct_name}'")

        return fixed_code, changes

    def add_missing_imports(self, code: str) -> Tuple[str, List[str]]:
        """
        Add missing import statements.

        Args:
            code: Python code

        Returns:
            Tuple of (fixed_code, list of changes)
        """
        fixed_code = code
        changes = []

        # Define import checks
        import_checks = {
            'pandas': ['pd.', 'pandas.'],
            'numpy': ['np.', 'numpy.'],
            'json': ['json.load', 'json.dump', 'json.loads', 'json.dumps'],
            'openpyxl': ['.read_excel(', '.to_excel('],
            'docx': ['Document(', 'from docx'],
        }

        for module, patterns in import_checks.items():
            needs_import = any(p in fixed_code for p in patterns)
            has_import = f'import {module}' in fixed_code or f'from {module}' in fixed_code

            if needs_import and not has_import:
                # Find insertion point (after docstring, before first code line)
                lines = fixed_code.split('\n')
                insert_index = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                        insert_index = i
                        break

                # Add import
                if module == 'pandas':
                    lines.insert(insert_index, 'import pandas as pd')
                    changes.append("Added: import pandas as pd")
                elif module == 'numpy':
                    lines.insert(insert_index, 'import numpy as np')
                    changes.append("Added: import numpy as np")
                else:
                    lines.insert(insert_index, f'import {module}')
                    changes.append(f"Added: import {module}")

                fixed_code = '\n'.join(lines)
                logger.info(f"[AutoFix] Added missing import: {module}")

        return fixed_code, changes

    def fix_encoding_issues(self, code: str) -> Tuple[str, List[str]]:
        """
        Add encoding='utf-8' to open() calls.

        Args:
            code: Python code

        Returns:
            Tuple of (fixed_code, list of changes)
        """
        fixed_code = code
        changes = []

        # Pattern: open('filename', 'r') -> open('filename', 'r', encoding='utf-8')
        patterns = [
            (r"open\((['\"][^'\"]+['\"])\s*,\s*'r'\s*\)", r"open(\1, 'r', encoding='utf-8')"),
            (r'open\((["\'][^"\']+["\']),\s*"r"\s*\)', r'open(\1, "r", encoding="utf-8")'),
        ]

        for pattern, replacement in patterns:
            matches = re.findall(pattern, fixed_code)
            if matches:
                fixed_code = re.sub(pattern, replacement, fixed_code)
                changes.append(f"Added encoding='utf-8' to {len(matches)} open() call(s)")
                logger.info(f"[AutoFix] Added encoding to {len(matches)} open() call(s)")
                break

        return fixed_code, changes

    def fix_nested_json_loading(
        self,
        code: str,
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """
        Replace pd.read_json() with json.load() for nested JSON files.

        Args:
            code: Python code
            validated_files: Dict mapping paths to filenames
            file_metadata: File metadata

        Returns:
            Tuple of (fixed_code, list of changes)
        """
        fixed_code = code
        changes = []

        # Find JSON files with depth > 2
        json_files = [
            (path, filename) for path, filename in validated_files.items()
            if file_metadata.get(path, {}).get('type') == 'json'
        ]

        for path, filename in json_files:
            metadata = file_metadata.get(path, {})
            max_depth = metadata.get('max_depth', 0)

            if max_depth > 2:
                # Replace pd.read_json with json.load
                old_single = f"pd.read_json('{filename}')"
                old_double = f'pd.read_json("{filename}")'
                new = f"json.load(open('{filename}', 'r', encoding='utf-8'))"

                if old_single in fixed_code:
                    fixed_code = fixed_code.replace(old_single, new)
                    changes.append(f"Fixed: pd.read_json → json.load for nested '{filename}'")
                    logger.info(f"[AutoFix] Replaced pd.read_json with json.load (depth={max_depth})")
                elif old_double in fixed_code:
                    fixed_code = fixed_code.replace(old_double, new)
                    changes.append(f"Fixed: pd.read_json → json.load for nested '{filename}'")
                    logger.info(f"[AutoFix] Replaced pd.read_json with json.load (depth={max_depth})")

                # Add json import if needed
                if "import json" not in fixed_code and ('json.load' in fixed_code or 'json.dump' in fixed_code):
                    lines = fixed_code.split('\n')
                    insert_index = 0
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        if stripped and not stripped.startswith('#'):
                            insert_index = i
                            break
                    lines.insert(insert_index, "import json")
                    fixed_code = '\n'.join(lines)
                    changes.append("Added: import json")

        return fixed_code, changes

    def fix_sys_argv_usage(
        self,
        code: str,
        validated_files: Dict[str, str]
    ) -> Tuple[str, List[str]]:
        """
        Remove sys.argv usage and replace with hardcoded filenames.

        Args:
            code: Python code
            validated_files: Dict mapping paths to filenames

        Returns:
            Tuple of (fixed_code, list of changes)
        """
        if 'sys.argv' not in code or not validated_files:
            return code, []

        fixed_code = code
        changes = []
        hardcoded_filename = list(validated_files.values())[0]

        # Fix patterns
        patterns = [
            # if len(sys.argv) > 1: main(sys.argv[1])
            (r'if\s+len\(sys\.argv\)\s*>\s*1\s*:\s*\n\s+(\w+)\(sys\.argv\[1\]\)',
             rf'\1(\'{hardcoded_filename}\')'),
            # main(sys.argv[1])
            (r'(\w+)\(sys\.argv\[1\]\)',
             rf'\1(\'{hardcoded_filename}\')'),
            # filename = sys.argv[1]
            (r'(\w+)\s*=\s*sys\.argv\[1\]',
             rf'\1 = \'{hardcoded_filename}\''),
            # else: print("Usage: ...")
            (r'else\s*:\s*\n\s+print\(["\']Usage:.*?\)',
             ''),
        ]

        for pattern, replacement in patterns:
            if re.search(pattern, fixed_code):
                fixed_code = re.sub(pattern, replacement, fixed_code, flags=re.MULTILINE)
                changes.append(f"Fixed: sys.argv → hardcoded '{hardcoded_filename}'")
                logger.info(f"[AutoFix] Replaced sys.argv with '{hardcoded_filename}'")
                break

        # Remove import sys if no longer needed
        if 'sys.argv' not in fixed_code and 'sys.executable' not in fixed_code:
            if not re.search(r'sys\.\w+', fixed_code):
                fixed_code = re.sub(r'^\s*import\s+sys\s*$', '', fixed_code, flags=re.MULTILINE)
                changes.append("Removed: 'import sys' (no longer needed)")

        return fixed_code, changes

    def apply_all_fixes(
        self,
        code: str,
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """
        Apply all automatic fixes to code.

        Args:
            code: Python code
            validated_files: Dict mapping paths to filenames
            file_metadata: File metadata

        Returns:
            Tuple of (fixed_code, list of all changes)
        """
        all_changes = []
        fixed_code = code

        # Fix 1: Filename references
        fixed_code, changes = self.fix_filename_references(fixed_code, validated_files)
        all_changes.extend(changes)

        # Fix 2: Missing imports
        fixed_code, changes = self.add_missing_imports(fixed_code)
        all_changes.extend(changes)

        # Fix 3: Encoding issues
        fixed_code, changes = self.fix_encoding_issues(fixed_code)
        all_changes.extend(changes)

        # Fix 4: Nested JSON loading
        fixed_code, changes = self.fix_nested_json_loading(fixed_code, validated_files, file_metadata)
        all_changes.extend(changes)

        # Fix 5: sys.argv usage
        fixed_code, changes = self.fix_sys_argv_usage(fixed_code, validated_files)
        all_changes.extend(changes)

        if all_changes:
            logger.info(f"[AutoFixer] Applied {len(all_changes)} automatic fix(es)")

        return fixed_code, all_changes
