"""
OpenAI-compatible models endpoint
/v1/models
"""
from fastapi import APIRouter
from typing import Optional
import time

from backend.models.schemas import ModelsListResponse, ModelObject
from backend.core.llm_backend import llm_backend
import config

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=ModelsListResponse)
def list_models():
    """
    List available models (OpenAI-compatible)
    """
    try:
        # Get models from LLM backend
        model_names = llm_backend.list_models()

        # If no models found, use default from config
        if not model_names:
            model_names = [config.OLLAMA_MODEL]

        # Convert to OpenAI format
        models = [
            ModelObject(
                id=model_name,
                object="model",
                created=int(time.time()),
                owned_by="system"
            )
            for model_name in model_names
        ]

        return ModelsListResponse(data=models)

    except Exception as e:
        # Fallback to config default if backend unavailable
        return ModelsListResponse(
            data=[
                ModelObject(
                    id=config.OLLAMA_MODEL,
                    object="model",
                    created=int(time.time()),
                    owned_by="system"
                )
            ]
        )
