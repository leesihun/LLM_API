"""
Admin Routes
Handles administrative operations like model changes and system configuration
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from backend.models.schemas import (
    ModelChangeRequest,
    ModelChangeResponse
)
from backend.api.dependencies import require_admin
from backend.config.settings import settings
from backend.agents.agent_system import agent_system
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ============================================================================
# Router Setup
# ============================================================================

admin_router = APIRouter(prefix="/api/admin", tags=["Admin"])


# ============================================================================
# Admin Endpoints
# ============================================================================

@admin_router.post("/model", response_model=ModelChangeResponse)
async def change_model(
    request: ModelChangeRequest,
    current_user: Dict[str, Any] = Depends(require_admin)
):
    """
    Change the active Ollama model (admin only)

    Requires admin role for access.
    """
    try:
        # Update settings in-memory
        settings.ollama_model = request.model

        return ModelChangeResponse(success=True, model=settings.ollama_model)
    except Exception as e:
        logger.error(f"Model change error: {e}")
        raise HTTPException(status_code=500, detail="Failed to change model")
