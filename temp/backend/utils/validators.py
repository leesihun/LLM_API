"""
Input Validation Utilities
Provides validation functions for files, sessions, and user inputs
"""

import re
from pathlib import Path
from typing import Optional, List, Tuple
from fastapi import HTTPException, status

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# File Validation
# ============================================================================

# Allowed file extensions by category
ALLOWED_EXTENSIONS = {
    "document": {".pdf", ".docx", ".doc", ".txt", ".md"},
    "spreadsheet": {".csv", ".xlsx", ".xls"},
    "data": {".json", ".xml", ".yaml", ".yml"},
    "image": {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"},
    "code": {".py", ".js", ".java", ".cpp", ".c", ".h"},
}

# Flatten all allowed extensions
ALL_ALLOWED_EXTENSIONS = set()
for extensions in ALLOWED_EXTENSIONS.values():
    ALL_ALLOWED_EXTENSIONS.update(extensions)

# Maximum file sizes (in bytes)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB default
MAX_FILE_SIZE_BY_TYPE = {
    "image": 10 * 1024 * 1024,      # 10 MB for images
    "document": 50 * 1024 * 1024,   # 50 MB for documents
    "spreadsheet": 100 * 1024 * 1024,  # 100 MB for spreadsheets
    "data": 50 * 1024 * 1024,       # 50 MB for data files
    "code": 5 * 1024 * 1024,        # 5 MB for code files
}


def validate_file_path(file_path: str, check_exists: bool = True) -> Path:
    """
    Validate file path for security and existence

    Args:
        file_path: Path to validate
        check_exists: If True, verify file exists

    Returns:
        Path object if valid

    Raises:
        HTTPException: If path is invalid or file doesn't exist

    Security checks:
        - Path traversal prevention (no ..)
        - Absolute path validation
        - File existence check
    """
    try:
        path = Path(file_path).resolve()

        # Prevent path traversal attacks
        if ".." in str(file_path):
            logger.warning(f"Path traversal attempt detected: {file_path}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file path: path traversal not allowed"
            )

        # Check if file exists
        if check_exists and not path.exists():
            logger.warning(f"File not found: {file_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {file_path}"
            )

        return path

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating file path: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file path: {str(e)}"
        )


def validate_file_extension(file_path: str, allowed_categories: Optional[List[str]] = None) -> bool:
    """
    Validate file extension against allowed types

    Args:
        file_path: Path to file
        allowed_categories: List of allowed categories (document, spreadsheet, data, image, code)
                          If None, all categories are allowed

    Returns:
        True if extension is allowed

    Raises:
        HTTPException: If extension is not allowed
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if not extension:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File has no extension"
        )

    # Determine allowed extensions
    if allowed_categories:
        allowed_exts = set()
        for category in allowed_categories:
            if category in ALLOWED_EXTENSIONS:
                allowed_exts.update(ALLOWED_EXTENSIONS[category])
    else:
        allowed_exts = ALL_ALLOWED_EXTENSIONS

    if extension not in allowed_exts:
        logger.warning(f"Invalid file extension: {extension} for file {file_path}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed: {extension}. Allowed types: {', '.join(sorted(allowed_exts))}"
        )

    return True


def validate_file_size(file_path: str, max_size: Optional[int] = None) -> bool:
    """
    Validate file size

    Args:
        file_path: Path to file
        max_size: Maximum size in bytes (if None, uses type-based limits)

    Returns:
        True if size is within limits

    Raises:
        HTTPException: If file is too large
    """
    path = Path(file_path)

    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )

    file_size = path.stat().st_size

    # Determine max size
    if max_size is None:
        # Use type-based limit
        extension = path.suffix.lower()
        for category, extensions in ALLOWED_EXTENSIONS.items():
            if extension in extensions:
                max_size = MAX_FILE_SIZE_BY_TYPE.get(category, MAX_FILE_SIZE)
                break
        else:
            max_size = MAX_FILE_SIZE

    if file_size > max_size:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        logger.warning(f"File too large: {file_path} ({size_mb:.2f} MB > {max_mb:.2f} MB)")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {size_mb:.2f} MB (max: {max_mb:.2f} MB)"
        )

    return True


def validate_file(
    file_path: str,
    allowed_categories: Optional[List[str]] = None,
    max_size: Optional[int] = None,
    check_exists: bool = True
) -> Path:
    """
    Comprehensive file validation

    Args:
        file_path: Path to validate
        allowed_categories: Allowed file categories
        max_size: Maximum file size in bytes
        check_exists: Whether to check file existence

    Returns:
        Validated Path object

    Raises:
        HTTPException: If validation fails
    """
    path = validate_file_path(file_path, check_exists=check_exists)
    validate_file_extension(file_path, allowed_categories=allowed_categories)

    if check_exists:
        validate_file_size(file_path, max_size=max_size)

    return path


# ============================================================================
# Session ID Validation
# ============================================================================

SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{8,64}$')


def validate_session_id(session_id: Optional[str]) -> Optional[str]:
    """
    Validate session ID format

    Args:
        session_id: Session ID to validate

    Returns:
        Valid session ID or None if input is None

    Raises:
        HTTPException: If session ID format is invalid

    Valid format: 8-64 alphanumeric characters, hyphens, and underscores
    """
    if session_id is None:
        return None

    if not SESSION_ID_PATTERN.match(session_id):
        logger.warning(f"Invalid session ID format: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format (must be 8-64 alphanumeric characters)"
        )

    return session_id


# ============================================================================
# Input Sanitization
# ============================================================================

def sanitize_text(text: str, max_length: int = 10000) -> str:
    """
    Sanitize text input

    Args:
        text: Text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text

    Raises:
        HTTPException: If text exceeds max length
    """
    if not text:
        return ""

    # Remove null bytes and other control characters
    sanitized = text.replace('\x00', '').strip()

    # Check length
    if len(sanitized) > max_length:
        logger.warning(f"Text input too long: {len(sanitized)} > {max_length}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input too long: {len(sanitized)} characters (max: {max_length})"
        )

    return sanitized


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage

    Args:
        filename: Original filename

    Returns:
        Sanitized filename

    Removes:
        - Path separators (/, \\)
        - Null bytes
        - Control characters
        - Leading/trailing spaces and dots
    """
    # Remove path separators and null bytes
    sanitized = filename.replace('/', '').replace('\\', '').replace('\x00', '')

    # Remove control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32)

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')

    # Ensure not empty
    if not sanitized:
        return "unnamed_file"

    return sanitized


# ============================================================================
# Username and Password Validation
# ============================================================================

USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,32}$')
PASSWORD_MIN_LENGTH = 8


def validate_username(username: str) -> Tuple[bool, Optional[str]]:
    """
    Validate username format

    Args:
        username: Username to validate

    Returns:
        Tuple of (is_valid, error_message)

    Rules:
        - 3-32 characters
        - Alphanumeric, hyphens, and underscores only
        - No spaces
    """
    if not username:
        return False, "Username is required"

    if len(username) < 3:
        return False, "Username must be at least 3 characters"

    if len(username) > 32:
        return False, "Username must be at most 32 characters"

    if not USERNAME_PATTERN.match(username):
        return False, "Username must contain only alphanumeric characters, hyphens, and underscores"

    return True, None


def validate_password(password: str) -> Tuple[bool, Optional[str]]:
    """
    Validate password strength

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, error_message)

    Rules:
        - At least 8 characters
        - Contains at least one letter
        - Contains at least one digit (recommended)
    """
    if not password:
        return False, "Password is required"

    if len(password) < PASSWORD_MIN_LENGTH:
        return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters"

    # Check for at least one letter
    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter"

    # Recommend (but don't require) at least one digit
    if not any(c.isdigit() for c in password):
        logger.info("Password validation: No digit found (recommended but not required)")

    return True, None
