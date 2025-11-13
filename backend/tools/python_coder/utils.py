"""Utility functions for python_coder module."""

from pathlib import Path


def get_original_filename(temp_filename: str) -> str:
    """
    Extract original filename from temp filename.

    Temp files are named: temp_XXXXXXXX_originalname.ext
    This method strips the temp_ prefix to get: originalname.ext

    Args:
        temp_filename: Filename possibly with temp_ prefix

    Returns:
        Original filename without temp_ prefix
    """
    if temp_filename.startswith('temp_'):
        # Split on underscore: ['temp', 'XXXXXXXX', 'originalname.ext']
        parts = temp_filename.split('_', 2)
        if len(parts) >= 3:
            return parts[2]  # Return the original filename

    # If no temp_ prefix, return as-is
    return temp_filename


def extract_code_from_markdown(response_text: str) -> str:
    """
    Extract Python code from markdown code blocks.

    Args:
        response_text: Response text possibly containing markdown

    Returns:
        Extracted code
    """
    code = response_text.strip()

    if code.startswith("```python"):
        code = code.split("```python", 1)[1]
        code = code.split("```", 1)[0]
    elif code.startswith("```"):
        code = code.split("```", 1)[1]
        code = code.split("```", 1)[0]

    return code.strip()
