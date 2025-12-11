"""
Vision Analyzer Tool Implementation
====================================
Uses vision-enabled LLM to understand and analyze images.

Features:
- Image description and analysis
- Visual question answering
- Multi-image comparison
- OCR-like text extraction
- Chart and diagram interpretation

Version: 1.0.0
Created: 2025-01-27
"""

from typing import Dict, Any, List, Optional
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

    Args:
        query: Question or task about the images
        file_paths: List of file paths (will filter for images)
        user_id: User ID for logging
        additional_context: Additional context or instructions

    Returns:
        Dictionary with analysis results:
        - success: True if analysis succeeded
        - analysis: The vision model's analysis/response
        - image_count: Number of images analyzed
        - error: Error message if failed

    Example:
        >>> result = await vision_analyzer_tool(
        ...     query="What's in this image?",
        ...     file_paths=["photo.jpg"]
        ... )
        >>> print(result['analysis'])
    """
    logger.info(f"[VisionAnalyzer] Starting vision analysis for user '{user_id}'")

    # Check if vision is enabled
    if not settings.vision_enabled:
        logger.warning("[VisionAnalyzer] Vision support is disabled in settings")
        return {
            "success": False,
            "error": "Vision support is disabled. Set vision_enabled=True in settings.",
            "analysis": None,
            "image_count": 0
        }

    # Extract image files
    if not file_paths:
        logger.warning("[VisionAnalyzer] No file paths provided")
        return {
            "success": False,
            "error": "No files provided for vision analysis",
            "analysis": None,
            "image_count": 0
        }

    image_paths, other_paths = extract_images_from_files(file_paths)

    if not image_paths:
        logger.warning("[VisionAnalyzer] No image files found in provided paths")
        return {
            "success": False,
            "error": "No image files found. Provided files are not images.",
            "analysis": None,
            "image_count": 0
        }

    logger.info(f"[VisionAnalyzer] Analyzing {len(image_paths)} images")

    try:
        # Create vision LLM
        vision_llm = LLMFactory.create_vision_llm(user_id=user_id)
        logger.debug(f"[VisionAnalyzer] Created vision LLM: {settings.ollama_vision_model}")

        # Create optimized vision prompt
        vision_prompt = create_vision_prompt(
            query=query,
            image_count=len(image_paths),
            additional_context=additional_context
        )

        # Construct multimodal message
        multimodal_message = create_multimodal_message(
            text=vision_prompt,
            image_paths=image_paths,
            role="user"
        )

        logger.debug("[VisionAnalyzer] Invoking vision model...")

        # Invoke vision LLM
        response = await vision_llm.ainvoke([multimodal_message])

        analysis_text = response.content if hasattr(response, 'content') else str(response)

        logger.info(
            f"[VisionAnalyzer] Analysis complete. "
            f"Response length: {len(analysis_text)} chars"
        )

        return {
            "success": True,
            "analysis": analysis_text,
            "image_count": len(image_paths),
            "images_analyzed": [Path(p).name for p in image_paths],
            "error": None
        }

    except Exception as e:
        logger.error(f"[VisionAnalyzer] Error during analysis: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Vision analysis failed: {str(e)}",
            "analysis": None,
            "image_count": len(image_paths) if image_paths else 0
        }


def vision_analyzer_tool_sync(
    query: str,
    file_paths: List[str] = None,
    user_id: str = "default",
    additional_context: str = None
) -> Dict[str, Any]:
    """
    Synchronous version of vision analyzer tool.

    Args:
        query: Question or task about the images
        file_paths: List of file paths (will filter for images)
        user_id: User ID for logging
        additional_context: Additional context or instructions

    Returns:
        Dictionary with analysis results

    Example:
        >>> result = vision_analyzer_tool_sync(
        ...     query="Describe this chart",
        ...     file_paths=["chart.png"]
        ... )
    """
    logger.info(f"[VisionAnalyzer] Starting sync vision analysis for user '{user_id}'")

    # Check if vision is enabled
    if not settings.vision_enabled:
        logger.warning("[VisionAnalyzer] Vision support is disabled in settings")
        return {
            "success": False,
            "error": "Vision support is disabled. Set vision_enabled=True in settings.",
            "analysis": None,
            "image_count": 0
        }

    # Extract image files
    if not file_paths:
        logger.warning("[VisionAnalyzer] No file paths provided")
        return {
            "success": False,
            "error": "No files provided for vision analysis",
            "analysis": None,
            "image_count": 0
        }

    image_paths, other_paths = extract_images_from_files(file_paths)

    if not image_paths:
        logger.warning("[VisionAnalyzer] No image files found in provided paths")
        return {
            "success": False,
            "error": "No image files found. Provided files are not images.",
            "analysis": None,
            "image_count": 0
        }

    logger.info(f"[VisionAnalyzer] Analyzing {len(image_paths)} images")

    try:
        # Create vision LLM
        vision_llm = LLMFactory.create_vision_llm(user_id=user_id)
        logger.debug(f"[VisionAnalyzer] Created vision LLM: {settings.ollama_vision_model}")

        # Create optimized vision prompt
        vision_prompt = create_vision_prompt(
            query=query,
            image_count=len(image_paths),
            additional_context=additional_context
        )

        # Construct multimodal message
        multimodal_message = create_multimodal_message(
            text=vision_prompt,
            image_paths=image_paths,
            role="user"
        )

        logger.debug("[VisionAnalyzer] Invoking vision model (sync)...")

        # Invoke vision LLM synchronously
        response = vision_llm.invoke([multimodal_message])

        analysis_text = response.content if hasattr(response, 'content') else str(response)

        logger.info(
            f"[VisionAnalyzer] Analysis complete. "
            f"Response length: {len(analysis_text)} chars"
        )

        return {
            "success": True,
            "analysis": analysis_text,
            "image_count": len(image_paths),
            "images_analyzed": [Path(p).name for p in image_paths],
            "error": None
        }

    except Exception as e:
        logger.error(f"[VisionAnalyzer] Error during sync analysis: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Vision analysis failed: {str(e)}",
            "analysis": None,
            "image_count": len(image_paths) if image_paths else 0
        }
