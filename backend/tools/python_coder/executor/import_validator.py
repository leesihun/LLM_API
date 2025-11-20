"""Import validation for secure code execution."""

import ast
from typing import List, Tuple

from backend.utils.logging_utils import get_logger
from .sandbox import BLOCKED_IMPORTS

logger = get_logger(__name__)


class ImportValidator:
    """
    Validates Python code imports for security.
    Blocks dangerous imports and function calls.
    """

    def __init__(self, blocked_imports: List[str] = None):
        """
        Initialize import validator.

        Args:
            blocked_imports: Optional list of blocked imports (defaults to BLOCKED_IMPORTS)
        """
        self.blocked_imports = blocked_imports or BLOCKED_IMPORTS

    def validate(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate that code only imports safe packages.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Parse code into AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]

        # Walk through AST nodes
        for node in ast.walk(tree):
            # Check import statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module in self.blocked_imports:
                        issues.append(f"Blocked import detected: {module}")
                        logger.warning(f"[ImportValidator] Blocked import: {module}")

            # Check from ... import statements
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    if module in self.blocked_imports:
                        issues.append(f"Blocked import detected: {module}")
                        logger.warning(f"[ImportValidator] Blocked from-import: {module}")

            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ['eval', 'exec', '__import__']:
                        issues.append(f"Dangerous function call detected: {func_name}")
                        logger.warning(f"[ImportValidator] Dangerous call: {func_name}")

        is_valid = len(issues) == 0

        if is_valid:
            logger.debug("[ImportValidator] Code validation passed")
        else:
            logger.error(f"[ImportValidator] Validation failed: {len(issues)} issues")

        return is_valid, issues

    def add_blocked_import(self, module: str):
        """
        Add a module to the blocked list.

        Args:
            module: Module name to block
        """
        if module not in self.blocked_imports:
            self.blocked_imports.append(module)
            logger.info(f"[ImportValidator] Added blocked import: {module}")

    def remove_blocked_import(self, module: str):
        """
        Remove a module from the blocked list.

        Args:
            module: Module name to unblock
        """
        if module in self.blocked_imports:
            self.blocked_imports.remove(module)
            logger.info(f"[ImportValidator] Removed blocked import: {module}")

    def is_blocked(self, module: str) -> bool:
        """
        Check if a module is blocked.

        Args:
            module: Module name to check

        Returns:
            True if module is blocked
        """
        return module in self.blocked_imports
