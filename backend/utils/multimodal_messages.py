"""
Multimodal Message Construction for Vision-Enabled LLMs
========================================================
Handles text + image content in LangChain message format.

Supports:
- Text-only messages (backward compatible)
- Single image + text
- Multiple images + text
- Base64 encoding integration

Version: 1.0.0
Created: 2025-01-27
"""

from typing import List, Dict, Any, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pathlib import Path

from backend.utils.image_encoder import encode_image_to_base64, is_image_file
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_multimodal_message(
    text: str,
    image_paths: List[str] = None,
    role: str = "user"
) -> Union[HumanMessage, SystemMessage, AIMessage]:
    """
    Create a multimodal message with text and images.

    Args:
        text: Text content
        image_paths: List of image file paths (optional)
        role: Message role - "user", "system", or "assistant"

    Returns:
        LangChain message with multimodal content

    Example:
        >>> # Text-only message
        >>> msg = create_multimodal_message("Hello!")

        >>> # Text + single image
        >>> msg = create_multimodal_message(
        ...     "What's in this image?",
        ...     image_paths=["photo.jpg"]
        ... )

        >>> # Text + multiple images
        >>> msg = create_multimodal_message(
        ...     "Compare these charts",
        ...     image_paths=["chart1.png", "chart2.png"]
        ... )
    """
    # Filter out non-existent or non-image files
    valid_image_paths = []
    if image_paths:
        for img_path in image_paths:
            if not Path(img_path).exists():
                logger.warning(f"[MultimodalMessage] Image not found: {img_path}")
                continue
            if not is_image_file(img_path):
                logger.warning(f"[MultimodalMessage] Not an image file: {img_path}")
                continue
            valid_image_paths.append(img_path)

    # If no valid images, create text-only message
    if not valid_image_paths:
        logger.debug(f"[MultimodalMessage] Creating text-only message (role: {role})")
        return _create_message_by_role(role, text)

    # Create multimodal content structure
    logger.info(
        f"[MultimodalMessage] Creating multimodal message with "
        f"{len(valid_image_paths)} images (role: {role})"
    )

    content = [{"type": "text", "text": text}]

    # Add images
    for img_path in valid_image_paths:
        try:
            base64_image = encode_image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            })
            logger.debug(f"[MultimodalMessage] Added image: {Path(img_path).name}")
        except Exception as e:
            logger.error(f"[MultimodalMessage] Failed to encode {img_path}: {e}")
            # Continue with other images

    return _create_message_by_role(role, content)


def _create_message_by_role(
    role: str,
    content: Union[str, List[Dict[str, Any]]]
) -> Union[HumanMessage, SystemMessage, AIMessage]:
    """
    Create message object based on role.

    Args:
        role: Message role
        content: Message content (text or multimodal structure)

    Returns:
        Appropriate LangChain message object
    """
    if role == "user":
        return HumanMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    else:
        logger.warning(f"[MultimodalMessage] Unknown role '{role}', defaulting to 'user'")
        return HumanMessage(content=content)


def create_vision_prompt(
    query: str,
    image_count: int,
    additional_context: str = None
) -> str:
    """
    Create an optimized prompt for vision tasks.

    Args:
        query: User's question or task
        image_count: Number of images attached
        additional_context: Additional context or instructions

    Returns:
        Formatted prompt for vision model

    Example:
        >>> prompt = create_vision_prompt(
        ...     "What's the trend?",
        ...     image_count=1,
        ...     additional_context="Focus on the sales data"
        ... )
    """
    parts = []

    if image_count > 1:
        parts.append(f"I've provided {image_count} images for your analysis.")
    elif image_count == 1:
        parts.append("I've provided an image for your analysis.")

    if additional_context:
        parts.append(additional_context)

    parts.append(query)

    return "\n\n".join(parts)


def extract_images_from_files(file_paths: List[str]) -> tuple[List[str], List[str]]:
    """
    Separate image files from other file types.

    Args:
        file_paths: List of file paths

    Returns:
        Tuple of (image_paths, other_file_paths)

    Example:
        >>> images, others = extract_images_from_files([
        ...     "photo.jpg", "data.csv", "chart.png"
        ... ])
        >>> # images = ["photo.jpg", "chart.png"]
        >>> # others = ["data.csv"]
    """
    image_paths = []
    other_paths = []

    for file_path in file_paths:
        if is_image_file(file_path):
            image_paths.append(file_path)
        else:
            other_paths.append(file_path)

    logger.debug(
        f"[MultimodalMessage] Separated {len(image_paths)} images "
        f"from {len(other_paths)} other files"
    )

    return image_paths, other_paths


def has_vision_keywords(text: str) -> bool:
    """
    Check if text contains keywords suggesting vision/image analysis.

    Args:
        text: User message text

    Returns:
        True if vision-related keywords found

    Example:
        >>> has_vision_keywords("What's in this image?")
        True
        >>> has_vision_keywords("Calculate the total")
        False
    """
    vision_keywords = [
        'image', 'picture', 'photo', 'screenshot', 'chart', 'graph',
        'diagram', 'visual', 'see', 'look', 'show', 'display',
        'what', 'describe', 'identify', 'detect', 'recognize',
        'ocr', 'text in', 'read from'
    ]

    text_lower = text.lower()
    return any(keyword in text_lower for keyword in vision_keywords)
