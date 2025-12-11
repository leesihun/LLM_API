"""
Vision Analyzer Tool
====================
Uses vision-enabled LLM to understand and analyze images.
"""

from typing import Dict, Any, List
from pathlib import Path

from backend.utils.llm_factory import LLMFactory
from backend.utils.multimodal_messages import (
    create_multimodal_message,
    create_vision_prompt,
    extract_images_from_files
)
from backend.utils.logging_utils import get_logger
from backend.config.settings import settings

logger = get_logger(__name__)


async def vision_analyzer_tool(
    query: str,
    file_paths: List[str] = None,
    user_id: str = "default",
    additional_context: str = None
) -> Dict[str, Any]:
    """
    Analyze images using vision-enabled LLM.
    """
    logger.info(f"[VisionAnalyzer] Starting vision analysis for user '{user_id}'")

    if not settings.vision_enabled:
        return {"success": False, "error": "Vision support disabled in settings", "analysis": None}

    if not file_paths:
        return {"success": False, "error": "No file paths provided", "analysis": None}

    image_paths, _ = extract_images_from_files(file_paths)

    if not image_paths:
        return {"success": False, "error": "No image files found", "analysis": None}

    try:
        vision_llm = LLMFactory.create_vision_llm(user_id=user_id)
        
        vision_prompt = create_vision_prompt(
            query=query,
            image_count=len(image_paths),
            additional_context=additional_context
        )

        multimodal_message = create_multimodal_message(
            text=vision_prompt,
            image_paths=image_paths,
            role="user"
        )

        response = await vision_llm.ainvoke([multimodal_message])
        analysis_text = response.content if hasattr(response, 'content') else str(response)

        return {
            "success": True,
            "analysis": analysis_text,
            "image_count": len(image_paths),
            "images_analyzed": [Path(p).name for p in image_paths],
            "error": None
        }

    except Exception as e:
        logger.error(f"[VisionAnalyzer] Error: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Vision analysis failed: {str(e)}",
            "analysis": None,
            "image_count": len(image_paths)
        }

