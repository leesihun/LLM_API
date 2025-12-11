"""
Role-Based Access Control (RBAC) Dependencies
Provides role checking for protected endpoints
"""

from typing import Dict, Any, Callable
from fastapi import Depends, HTTPException, status

from .auth import get_current_user
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


def require_role(required_role: str) -> Callable:
    """
    Create a FastAPI dependency that requires a specific user role

    Args:
        required_role: The role required to access the endpoint (e.g., "admin", "user", "guest")

    Returns:
        A FastAPI dependency function that checks user role

    Raises:
        HTTPException: 403 Forbidden if user doesn't have required role

    Usage:
        # Require admin role for endpoint
        @router.post("/admin/settings")
        async def update_settings(
            settings: Settings,
            current_user: Dict = Depends(require_role("admin"))
        ):
            return {"message": "Settings updated"}

        # Alternative shorter syntax
        @router.delete("/admin/users/{user_id}")
        async def delete_user(
            user_id: str,
            _: Dict = Depends(require_role("admin"))
        ):
            return {"message": "User deleted"}
    """
    async def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        """Check if current user has required role"""
        user_role = current_user.get("role")

        if user_role != required_role:
            logger.warning(
                f"Access denied: User '{current_user.get('username')}' "
                f"with role '{user_role}' attempted to access endpoint requiring '{required_role}'"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )

        logger.debug(f"Access granted: User '{current_user.get('username')}' has required role '{required_role}'")
        return current_user

    return role_checker


def require_any_role(*roles: str) -> Callable:
    """
    Create a FastAPI dependency that requires any of the specified roles

    Args:
        *roles: Variable number of acceptable roles

    Returns:
        A FastAPI dependency function that checks if user has any of the specified roles

    Raises:
        HTTPException: 403 Forbidden if user doesn't have any of the required roles

    Usage:
        @router.get("/content")
        async def get_content(
            current_user: Dict = Depends(require_any_role("admin", "user"))
        ):
            return {"content": "Protected content"}
    """
    async def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        """Check if current user has any of the required roles"""
        user_role = current_user.get("role")

        if user_role not in roles:
            logger.warning(
                f"Access denied: User '{current_user.get('username')}' "
                f"with role '{user_role}' attempted to access endpoint requiring one of {roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {', '.join(roles)}"
            )

        logger.debug(f"Access granted: User '{current_user.get('username')}' has role '{user_role}'")
        return current_user

    return role_checker


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Convenience dependency for requiring admin role

    This is a shorthand for require_role("admin") for common use cases

    Usage:
        @router.post("/admin/model")
        async def change_model(
            model: str,
            current_user: Dict = Depends(require_admin)
        ):
            return {"message": "Model changed"}
    """
    if current_user.get("role") != "admin":
        logger.warning(
            f"Access denied: User '{current_user.get('username')}' "
            f"with role '{current_user.get('role')}' attempted to access admin endpoint"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    return current_user
