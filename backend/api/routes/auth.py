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
from backend.utils.auth import authenticate_user, create_access_token, get_current_user
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
    """Create a new user account (stores plaintext password for simplicity)"""
    users_path = Path(settings.users_path)
    users_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if users_path.exists():
            with open(users_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"users": []}

        # Check if user exists
        for u in data.get("users", []):
            if u.get("username") == request.username:
                raise HTTPException(status_code=400, detail="Username already exists")

        # Append new user (plaintext password for dev simplicity)
        new_user = {
            "username": request.username,
            "password": request.password,
            "role": request.role or "guest",
        }
        data.setdefault("users", []).append(new_user)

        with open(users_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return SignupResponse(success=True, user=User(username=new_user["username"], role=new_user["role"]))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Error creating user")
