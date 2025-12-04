"""
Unified Image File Handler
===========================
Handler for image files merging functionality from python_coder and file_analyzer.

Features:
- Dimension detection
- Format identification
- Color mode detection
- Image metadata extraction
- Both extract_metadata() and analyze() support

Version: 2.0.0 (Unified)
Created: 2025-01-20
"""

from typing import Dict, Any

from backend.core.file_handlers.base import FileHandler


class ImageHandler(FileHandler):
    """Unified handler for image files."""

    def __init__(self):
        """Initialize image handler."""
        super().__init__()
        self.supported_extensions = [
            'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'ico', 'svg'
        ]

    def supports(self, file_path: str) -> bool:
        """
        Check if this is an image file.

        Args:
            file_path: Path to the file

        Returns:
            True if file has image extension
        """
        return self.supports_file(file_path)

    def extract_metadata(
        self,
        file_path: str,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Extract metadata from image file for code generation.

        Args:
            file_path: Path to the image file
            quick_mode: If True, extract only essential metadata

        Returns:
            Dictionary containing image metadata
        """
        metadata = {
            'file_type': 'image',
            'file_size': self.get_file_size(file_path),
            'error': None
        }

        try:
            from PIL import Image

            with Image.open(file_path) as img:
                metadata['format'] = img.format
                metadata['width'] = img.width
                metadata['height'] = img.height
                metadata['dimensions'] = f"{img.width}x{img.height}"
                metadata['mode'] = img.mode

        except ImportError:
            metadata['error'] = "PIL/Pillow not installed"
        except Exception as e:
            metadata['error'] = str(e)
            self.logger.error(f"Image metadata extraction error: {e}", exc_info=True)

        return metadata

    def analyze(self, file_path: str, query: str = "") -> Dict[str, Any]:
        """
        Perform comprehensive image analysis.

        Args:
            file_path: Path to the image file
            query: Optional query (not used for image)

        Returns:
            Dictionary with comprehensive analysis results
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
            self.logger.error(f"Image analysis error: {e}", exc_info=True)
            return {"format": "Image", "error": str(e)}

    def build_context_section(
        self,
        filename: str,
        metadata: Dict[str, Any],
        index: int
    ) -> str:
        """
        Build context section for image file.

        Args:
            filename: Name of the image file
            metadata: Metadata from extract_metadata()
            index: File index for numbering

        Returns:
            Formatted context string
        """
        lines = []
        lines.append(f"{index}. {filename} (Image)")

        if metadata.get('error'):
            lines.append(f"   Error: {metadata['error']}")
            return '\n'.join(lines)

        # Basic info
        img_format = metadata.get('format')
        if img_format:
            lines.append(f"   Format: {img_format}")

        dimensions = metadata.get('dimensions')
        if dimensions:
            lines.append(f"   Dimensions: {dimensions}")

        mode = metadata.get('mode')
        if mode:
            lines.append(f"   Mode: {mode}")

        file_size = metadata.get('file_size')
        if file_size:
            lines.append(f"   Size: {self.format_file_size(file_size)}")

        return '\n'.join(lines)
