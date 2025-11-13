"""
Image File Handler
===================
Handler for analyzing image files (PNG, JPG, JPEG, GIF, BMP).

Version: 1.0.0
Created: 2025-01-13
"""

from typing import Dict, Any, List

from backend.utils.logging_utils import get_logger
from ..base_handler import BaseFileHandler

logger = get_logger(__name__)


class ImageHandler(BaseFileHandler):
    """
    Handler for image file analysis.

    Features:
    - Dimension detection
    - Format identification
    - Color mode detection
    - Image metadata extraction
    """

    def __init__(self):
        """Initialize image handler."""
        self.supported_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp']

    def supports(self, file_path: str) -> bool:
        """
        Check if this is an image file.

        Args:
            file_path: Path to the file

        Returns:
            True if file has image extension
        """
        extension = self.get_file_extension(file_path)
        return extension in self.supported_extensions

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported extensions."""
        return self.supported_extensions

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze image file.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with analysis results including:
            - format: 'Image (format_name)'
            - dimensions: 'widthxheight'
            - mode: Color mode (RGB, RGBA, L, etc.)
            - width: Image width in pixels
            - height: Image height in pixels
        """
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                return {
                    "format": f"Image ({img.format})",
                    "dimensions": f"{img.width}x{img.height}",
                    "mode": img.mode,
                    "width": img.width,
                    "height": img.height
                }

        except ImportError:
            return {"format": "Image", "error": "PIL/Pillow not installed"}
        except Exception as e:
            logger.error(f"Image analysis error: {e}", exc_info=True)
            return {"format": "Image", "error": str(e)}
