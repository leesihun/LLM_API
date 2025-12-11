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
from backend.api.dependencies import get_current_user
from backend.runtime import (
    authenticate_credentials,
    create_user_account,
)
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
        result = authenticate_credentials(request.username, request.password)
        return LoginResponse(**result)
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
    """Create a new user account with hashed password"""
    try:
        return create_user_account(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error creating user")
