"""
Image Encoding Utilities for Multimodal LLM Inputs
===================================================
Handles base64 encoding and image preprocessing for vision models.

Features:
- Base64 encoding for Ollama vision models
- Automatic image resizing to reduce token usage
- Format conversion (RGBA → RGB)
- Memory-efficient processing

Version: 1.0.0
Created: 2025-01-27
"""

import base64
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import io

from backend.utils.logging_utils import get_logger
from backend.config.settings import settings

logger = get_logger(__name__)


def encode_image_to_base64(
    image_path: str,
    max_size: Optional[int] = None
) -> str:
    """
    Encode image to base64 string with optional resizing.

    Args:
        image_path: Path to image file
        max_size: Maximum dimension (width/height) for resizing.
                 If None, uses settings.vision_max_image_size

    Returns:
        Base64 encoded string

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is unsupported

    Example:
        >>> base64_str = encode_image_to_base64("chart.png")
        >>> # Use in multimodal message
        >>> image_url = f"data:image/jpeg;base64,{base64_str}"
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    max_size = max_size or settings.vision_max_image_size

    try:
        with Image.open(image_path) as img:
            original_size = img.size

            # Resize if needed to reduce token usage
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.debug(
                    f"[ImageEncoder] Resized {image_path.name} from "
                    f"{original_size} to {img.size}"
                )

            # Convert RGBA to RGB (Ollama vision models prefer RGB)
            if img.mode == 'RGBA':
                # Create white background
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = rgb_img
                logger.debug(f"[ImageEncoder] Converted {image_path.name} from RGBA to RGB")
            elif img.mode not in ('RGB', 'L'):
                # Convert other modes to RGB
                img = img.convert('RGB')
                logger.debug(f"[ImageEncoder] Converted {image_path.name} to RGB")

            # Encode to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            base64_bytes = base64.b64encode(buffered.getvalue())
            base64_str = base64_bytes.decode('utf-8')

            logger.info(
                f"[ImageEncoder] Encoded {image_path.name} "
                f"(size: {img.size}, bytes: {len(base64_str)})"
            )

            return base64_str

    except Exception as e:
        logger.error(f"[ImageEncoder] Failed to encode {image_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to encode image: {e}") from e


def get_image_info(image_path: str) -> dict:
    """
    Get basic information about an image file.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with image metadata:
        - width: Image width in pixels
        - height: Image height in pixels
        - format: Image format (PNG, JPEG, etc.)
        - mode: Color mode (RGB, RGBA, L, etc.)
        - size_kb: File size in kilobytes

    Example:
        >>> info = get_image_info("photo.jpg")
        >>> print(f"Image is {info['width']}x{info['height']}")
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        with Image.open(image_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
                'size_kb': round(image_path.stat().st_size / 1024, 2)
            }
    except Exception as e:
        logger.error(f"[ImageEncoder] Failed to get info for {image_path}: {e}")
        raise ValueError(f"Failed to read image: {e}") from e


def is_image_file(file_path: str) -> bool:
    """
    Check if a file is a supported image format.

    Args:
        file_path: Path to file

    Returns:
        True if file has image extension

    Example:
        >>> is_image_file("photo.jpg")
        True
        >>> is_image_file("document.pdf")
        False
    """
    image_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp',
        '.tiff', '.tif', '.webp', '.ico', '.svg'
    }

    file_path = Path(file_path)
    return file_path.suffix.lower() in image_extensions
