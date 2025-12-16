"""
Authentication Utilities
Handles JWT tokens, password hashing, and user verification
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
from pathlib import Path

from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

from backend.config.settings import settings


# ============================================================================
# Password Hashing with bcrypt
# ============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt

    Args:
        password: Plain text password

    Returns:
        Hashed password string
        
    Raises:
        ValueError: If password exceeds 72 bytes (bcrypt limitation)
    """
    # Bcrypt has a 72 byte limit for passwords
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        raise ValueError(
            f"Password cannot exceed 72 bytes. Current password is {len(password_bytes)} bytes. "
            "Please use a shorter password."
        )
    
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hashed password

    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password to compare against

    Returns:
        True if password matches, False otherwise

    Note:
        Also handles backward compatibility with plaintext passwords
        for migration purposes. If hashed_password doesn't start with
        bcrypt prefix ($2b$), falls back to plaintext comparison.
    """
    # Backward compatibility: check if password is hashed
    if hashed_password.startswith("$2b$") or hashed_password.startswith("$2a$"):
        # Bcrypt hashed password
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            return False
    else:
        # Legacy plaintext password - allow for migration
        return plain_password == hashed_password


# ============================================================================
# JWT Token Management
# ============================================================================

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)

    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and verify JWT token"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================================
# User Management
# ============================================================================

def load_users() -> Dict[str, Any]:
    """Load users from JSON file"""
    users_path = Path(settings.users_path)

    if not users_path.exists():
        # Create default users file
        default_users = {
            "users": [
                {
                    "username": "guest",
                    "password": "guest_test1",
                    "role": "guest"
                },
                {
                    "username": "admin",
                    "password": "administrator",
                    "role": "admin"
                }
            ]
        }
        users_path.parent.mkdir(parents=True, exist_ok=True)
        with open(users_path, "w") as f:
            json.dump(default_users, f, indent=2)

    with open(users_path, "r") as f:
        return json.load(f)


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate user with username and password

    Args:
        username: Username to authenticate
        password: Plain text password

    Returns:
        User dict with username and role if authentication succeeds, None otherwise

    Note:
        Supports both 'password' (legacy) and 'password_hash' fields for migration
    """
    users_data = load_users()

    for user in users_data.get("users", []):
        if user["username"] == username:
            # Support both 'password_hash' (new) and 'password' (legacy) fields
            stored_password = user.get("password_hash", user.get("password", ""))
            if verify_password(password, stored_password):
                return {
                    "username": user["username"],
                    "role": user["role"]
                }

    return None


# ============================================================================
# FastAPI Dependencies (DEPRECATED - Use backend.api.dependencies instead)
# ============================================================================

security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user

    DEPRECATED: This function is kept for backward compatibility.
    New code should import from: backend.api.dependencies.get_current_user

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        Dict containing user information (username, role)

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials

    try:
        payload = decode_access_token(token)
        username: str = payload.get("sub")

        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )

        # Verify user still exists
        users_data = load_users()
        user = next((u for u in users_data.get("users", []) if u["username"] == username), None)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        return {
            "username": user["username"],
            "role": user["role"]
        }

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
