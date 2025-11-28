"""Utility functions for code execution."""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


def prepare_input_files(
    input_files: Dict[str, str],
    execution_dir: Path,
    file_cache: Dict[str, Dict[str, str]],
    session_id: Optional[str] = None
) -> None:
    """
    Copy input files to execution directory with caching.

    Args:
        input_files: Dict mapping original file paths to basenames
        execution_dir: Target directory for files
        file_cache: File cache dictionary for session-based caching
        session_id: Optional session ID for caching
    """
    # Check if files already cached for this session
    if session_id and session_id in file_cache:
        logger.info(f"[FileUtils] Using cached input files ({len(input_files)} files)")
        return

    # Copy files
    for original_path, basename in input_files.items():
        target_path = execution_dir / basename
        if not target_path.exists():
            shutil.copy2(original_path, target_path)
            logger.debug(f"[FileUtils] Copied {original_path} -> {target_path}")

    # Cache file list for this session
    if session_id:
        file_cache[session_id] = input_files
        logger.info(f"[FileUtils] Copied and cached {len(input_files)} input files")
    else:
        logger.debug(f"[FileUtils] Copied {len(input_files)} input files")


def cleanup_execution_dir(execution_dir: Path, session_id: Optional[str] = None) -> None:
    """
    Cleanup execution directory.

    Args:
        execution_dir: Directory to cleanup
        session_id: Optional session ID (if provided, directory is kept)
    """
    # Only cleanup temporary directories (no session_id)
    if not session_id:
        try:
            if execution_dir.exists():
                shutil.rmtree(execution_dir)
                logger.debug("[FileUtils] Cleaned up temporary directory")
        except Exception as e:
            logger.warning(f"[FileUtils] Failed to cleanup: {e}")
    else:
        logger.debug("[FileUtils] Keeping session directory")


def log_execution_result(result: Dict[str, Any]) -> None:
    """
    Log execution result with appropriate formatting.

    Args:
        result: Execution result dictionary
    """
    execution_time = result.get("execution_time", 0)

    if result["success"]:
        logger.success("Code execution succeeded", f"{execution_time:.2f}s")
    else:
        logger.failure("Code execution failed", f"Return code: {result.get('return_code', -1)}")

    # Log stdout
    output = result.get("output", "")
    if output:
        logger.multiline(output, title="STDOUT", max_lines=50)
    else:
        logger.info("STDOUT: (empty)")

    # Log stderr
    error = result.get("error", "")
    if error:
        if not result["success"]:
            logger.multiline(error, title="STDERR - ERROR", max_lines=30)
        else:
            logger.multiline(error, title="STDERR - WARNING", max_lines=20)


def enhance_error_detection(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced error detection: check for error patterns in stdout for BOTH success and failure cases.

    For success (return_code 0): Detects silent errors that exit with code 0
    For failure (return_code != 0): Enriches empty stderr with stdout error patterns

    Args:
        result: Execution result dict

    Returns:
        Updated result dict with enhanced error detection
    """
    output = result.get("output", "")
    error = result.get("error", "")
    return_code = result.get("return_code", -1)

    if not output:
        return result

    # Define error patterns to check
    error_patterns = [
        "Error:", "error:", "ERROR:",
        "Failed:", "failed:", "FAILED:",
        "Exception:", "exception:",
        "Traceback (most recent call last):",
        "not found", "Not found", "NOT FOUND",
        "does not contain", "does not exist",
        "No valid", "no valid",
        "Invalid", "invalid"
    ]

    output_lower = output.lower()
    pattern_found = None

    for pattern in error_patterns:
        if pattern.lower() in output_lower:
            pattern_found = pattern
            logger.warning(f"[ErrorDetection] Error pattern in stdout: '{pattern}'")
            break

    if pattern_found:
        if return_code == 0:
            # Success case: code printed error but exited with 0
            result["success"] = False
            result["error"] = f"Error detected in output (code exited with 0 but printed error):\n{output}"
            logger.error("[ErrorDetection] Code printed error messages despite return code 0")
        elif not error or error.strip() == "":
            # Failure case: empty stderr but stdout has error patterns
            result["error"] = f"Error detected in output (stderr was empty):\n{output}"
            logger.info("[ErrorDetection] Enriched empty stderr with stdout error patterns")

    return result


def save_code_to_file(
    code: str,
    execution_dir: Path,
    stage_name: Optional[str] = None
) -> None:
    """
    Save code to execution directory.

    Args:
        code: Python code to save
        execution_dir: Directory to save code in
        stage_name: Optional stage name for versioning (e.g., "verify1", "exec2")
    """
    # Always save to script.py (main execution file)
    script_path = execution_dir / "script.py"
    script_path.write_text(code, encoding='utf-8')
    logger.debug(f"[FileUtils] Wrote code to {script_path}")

    # Also save to stage-specific file if stage_name provided
    if stage_name:
        stage_script_path = execution_dir / f"script_{stage_name}.py"
        stage_script_path.write_text(code, encoding='utf-8')
        logger.info(f"[FileUtils] [SAVED] Saved stage code to {stage_script_path.name}")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def validate_file_path(file_path: str) -> bool:
    """
    Validate that file path exists and is accessible.

    Args:
        file_path: Path to validate

    Returns:
        True if valid
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception as e:
        logger.warning(f"[FileUtils] Invalid file path '{file_path}': {e}")
        return False


def get_execution_env() -> Dict[str, str]:
    """
    Get environment variables for code execution.

    Returns:
        Dictionary of environment variables
    """
    import os

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'  # Force UTF-8 encoding

    return env
