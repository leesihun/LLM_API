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
from backend.utils.auth import get_current_user
from backend.config.settings import settings
from backend.tasks.chat_task import chat_task
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
async def change_model(request: ModelChangeRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Change the active Ollama model (admin only)"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")

    try:
        # Update settings in-memory
        settings.ollama_model = request.model

        # Reinitialize simple chat LLM to take effect immediately
        from backend.utils.llm_factory import LLMFactory
        chat_task.llm = LLMFactory.create_llm()

        return ModelChangeResponse(success=True, model=settings.ollama_model)
    except Exception as e:
        logger.error(f"Model change error: {e}")
        raise HTTPException(status_code=500, detail="Failed to change model")
