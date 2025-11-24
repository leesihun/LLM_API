"""
FastAPI Dependencies
Shared dependencies for authentication, authorization, and validation
"""

from .auth import get_current_user, security
from .role_checker import require_role, require_any_role, require_admin

__all__ = [
    "get_current_user",
    "security",
    "require_role",
    "require_any_role",
    "require_admin",
]
