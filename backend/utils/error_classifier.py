"""
Error Classifier Utility
========================
Unified error classification for code execution errors.

Extracted from duplicate implementations in:
- backend/tools/python_coder/orchestrator.py
- backend/tools/python_coder/code_patcher.py

Version: 1.0.0
Created: 2025-01-20
"""

from typing import Dict, Tuple, Optional, Any


class ErrorClassifier:
    """
    Unified error classification utility.
    
    Classifies Python execution errors into categories for better
    error handling and user guidance.
    """

    @staticmethod
    def classify_error(
        error_message: str,
        return_code: int = -1,
        stdout: str = "",
        namespace: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Classify error and return (error_type, specific_guidance).

        Args:
            error_message: The error message from execution
            return_code: Process return code
            stdout: Standard output from execution
            namespace: Variable namespace (if available)

        Returns:
            Tuple of (error_type, guidance_text)
        """
        if not error_message or error_message.strip() == "":
            return ErrorClassifier._classify_silent_failure(return_code, stdout, namespace)

        error_lower = error_message.lower()
        
        # Import errors
        if "importerror" in error_lower or "modulenotfounderror" in error_lower:
            return ("import_error", "Missing import. Install package: pip install <package_name>")
        
        # File errors
        if "filenotfounderror" in error_lower or "no such file" in error_lower:
            return ("file_not_found", "File not found. Check file path and ensure file exists.")
        
        # Key errors
        if "keyerror" in error_lower:
            return ("key_error", "Dictionary key not found. Check key name and dictionary contents.")
        
        # Attribute errors
        if "attributeerror" in error_lower:
            return ("attribute_error", "Object attribute not found. Check object type and available attributes.")
        
        # Type errors
        if "typeerror" in error_lower:
            return ("type_error", "Type mismatch. Check variable types and function signatures.")
        
        # Value errors
        if "valueerror" in error_lower:
            return ("value_error", "Invalid value. Check input format and constraints.")
        
        # Index errors
        if "indexerror" in error_lower:
            return ("index_error", "Index out of range. Check list/array length before accessing.")
        
        # Zero division
        if "zerodivisionerror" in error_lower:
            return ("zero_division_error", "Division by zero. Add check for zero denominator.")
        
        # Syntax errors
        if "syntaxerror" in error_lower:
            return ("syntax_error", "Syntax error in code. Check Python syntax and indentation.")
        
        # Name errors
        if "nameerror" in error_lower:
            return ("name_error", "Variable not defined. Check variable name spelling and scope.")
        
        # Runtime errors (catch-all)
        return ("runtime_error", f"Runtime error: {error_message[:200]}")

    @staticmethod
    def _classify_silent_failure(
        return_code: int,
        stdout: str,
        namespace: Optional[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Classify silent failures (no error message)."""
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

        guidance_parts.append(
            " Common fixes: 1) Check all imports are installed (pip install <package>), "
            "2) Verify code syntax, 3) Add try/except with print() to catch silent errors, "
            "4) Check file paths exist."
        )

        return ("SilentFailure", "".join(guidance_parts))

    @staticmethod
    def extract_error_line(error_message: str) -> int:
        """
        Extract the line number where the error occurred from traceback.

        Args:
            error_message: Error message with traceback

        Returns:
            Line number where error occurred (1-indexed), or 50 as fallback
        """
        import re
        match = re.search(r'line (\d+)', error_message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 50  # Fallback: assume error is in the middle

