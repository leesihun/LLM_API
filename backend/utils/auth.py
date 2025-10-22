"""
Authentication Utilities
Handles JWT tokens, password hashing, and user verification
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
from pathlib import Path

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.config.settings import settings


# ============================================================================
# Password Hashing
# ============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


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
                    "password_hash": get_password_hash("guest_test1"),
                    "role": "guest"
                },
                {
                    "username": "admin",
                    "password_hash": get_password_hash("administrator"),
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
    """Authenticate user with username and password"""
    users_data = load_users()

    for user in users_data.get("users", []):
        if user["username"] == username:
            if verify_password(password, user["password_hash"]):
                return {
                    "username": user["username"],
                    "role": user["role"]
                }

    return None


# ============================================================================
# FastAPI Dependencies
# ============================================================================

security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """FastAPI dependency to get current authenticated user"""
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
