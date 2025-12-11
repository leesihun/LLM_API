"""
Admin endpoints
/api/admin/model - Change default model
"""
from fastapi import APIRouter, Depends

from backend.models.schemas import ChangeModelRequest
from backend.utils.auth import require_admin
import config

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/model")
def change_model(
    request: ChangeModelRequest,
    admin: dict = Depends(require_admin)
):
    """
    Change the default model (admin only)
    Note: This updates the runtime config, not persistent storage
    """
    # Update the global config
    config.OLLAMA_MODEL = request.model

    return {
        "status": "success",
        "message": f"Model changed to {request.model}",
        "model": request.model
    }
