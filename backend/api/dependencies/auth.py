"""
Authentication Dependencies
Handles user authentication and current user retrieval
"""

from typing import Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError

from backend.utils.auth import decode_access_token, load_users
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# HTTP Bearer token security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user from JWT token

    Args:
        credentials: HTTP Bearer credentials containing JWT token

    Returns:
        Dict containing user information (username, role)

    Raises:
        HTTPException: If token is invalid or user not found (401 Unauthorized)

    Usage:
        @router.get("/protected")
        async def protected_route(current_user: Dict = Depends(get_current_user)):
            return {"user": current_user["username"]}
    """
    token = credentials.credentials

    try:
        # Decode and verify JWT token
        payload = decode_access_token(token)
        username: str = payload.get("sub")

        if username is None:
            logger.warning("Token missing 'sub' claim")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify user still exists in the system
        users_data = load_users()
        user = next(
            (u for u in users_data.get("users", []) if u["username"] == username),
            None
        )

        if user is None:
            logger.warning(f"User not found in system: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Return user information
        return {
            "username": user["username"],
            "role": user["role"]
        }

    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_current_user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error",
            headers={"WWW-Authenticate": "Bearer"},
        )
