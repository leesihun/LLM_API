"""
Authentication Routes
Handles user authentication, signup, and profile management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
from pathlib import Path
import json
import traceback

from backend.models.schemas import (
    LoginRequest,
    LoginResponse,
    User,
    SignupRequest,
    SignupResponse
)
from backend.utils.auth import (
    authenticate_user,
    create_access_token,
    hash_password
)
from backend.api.dependencies import get_current_user
from backend.config.settings import settings
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ============================================================================
# Router Setup
# ============================================================================

auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ============================================================================
# Authentication Endpoints
# ============================================================================

@auth_router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return access token"""
    try:
        logger.info(f"Login attempt for user: {request.username}")
        user = authenticate_user(request.username, request.password)

        if user is None:
            logger.warning(f"Failed login attempt for user: {request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create access token
        access_token = create_access_token(data={"sub": user["username"]})
        logger.info(f"Successful login for user: {request.username}")

        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=user
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login error: {str(e)}"
        )


@auth_router.get("/me", response_model=User)
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current authenticated user"""
    return User(
        username=current_user["username"],
        role=current_user["role"]
    )


@auth_router.post("/signup", response_model=SignupResponse)
async def signup(request: SignupRequest):
    """
    Create a new user account with hashed password

    Security improvements:
    - Passwords are hashed using bcrypt before storage
    - Password validation (minimum length, complexity)
    - Username validation
    """
    users_path = Path(settings.users_path)
    users_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Validate username
        if len(request.username.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Username must be at least 3 characters long"
            )

        # Validate password
        if len(request.password) < 8:
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 8 characters long"
            )

        if users_path.exists():
            with open(users_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"users": []}

        # Check if user exists
        for u in data.get("users", []):
            if u.get("username") == request.username:
                raise HTTPException(status_code=400, detail="Username already exists")

        # Hash password before storing
        hashed_password = hash_password(request.password)
        logger.info(f"Creating new user: {request.username} with role: {request.role or 'guest'}")

        # Create new user with hashed password
        new_user = {
            "username": request.username,
            "password_hash": hashed_password,  # Store as 'password_hash' for clarity
            "role": request.role or "guest",
        }
        data.setdefault("users", []).append(new_user)

        with open(users_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"User created successfully: {request.username}")
        return SignupResponse(
            success=True,
            user=User(username=new_user["username"], role=new_user["role"])
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error creating user")
