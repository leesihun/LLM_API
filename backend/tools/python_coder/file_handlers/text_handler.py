"""Text file handler for general text files."""

from pathlib import Path
from typing import Any, Dict

from .base import BaseFileHandler


class TextFileHandler(BaseFileHandler):
    """Handler for general text files."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.txt', '.md', '.py', '.js', '.html', '.css', '.xml', '.log']

    def extract_metadata(
        self,
        file_path: Path,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """Extract metadata from text file."""
        metadata = {
            'file_type': 'text',
            'file_size': self._get_file_size(file_path),
            'extension': file_path.suffix,
            'error': None
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if not quick_mode:
                    content = f.read()
                    lines = content.split('\n')

                    metadata['num_lines'] = len(lines)
                    metadata['num_chars'] = len(content)
                    metadata['preview'] = '\n'.join(lines[:10])  # First 10 lines
                else:
                    # Quick mode: just read first few lines
                    preview_lines = []
                    for i, line in enumerate(f):
                        if i >= 5:
                            break
                        preview_lines.append(line.rstrip())
                    metadata['preview'] = '\n'.join(preview_lines)
                    metadata['num_lines'] = 'unknown'

        except UnicodeDecodeError:
            metadata['error'] = 'Unable to decode as UTF-8 (binary file?)'
        except Exception as e:
            metadata['error'] = str(e)

        return metadata

    def build_context_section(
        self,
        filename: str,
        metadata: Dict[str, Any],
        index: int
    ) -> str:
        """Build context section for text file."""
        lines = []
        ext = metadata.get('extension', 'unknown')
        lines.append(f"{index}. {filename} (Text/{ext})")

        if metadata.get('error'):
            lines.append(f"   Error: {metadata['error']}")
            return '\n'.join(lines)

        # Basic info
        num_lines = metadata.get('num_lines', 'unknown')
        if num_lines != 'unknown':
            lines.append(f"   Lines: {num_lines}")

            num_chars = metadata.get('num_chars', 0)
            lines.append(f"   Characters: {num_chars}")

        # Preview
        preview = metadata.get('preview', '')
        if preview:
            preview_short = preview[:300]
            if len(preview) > 300:
                preview_short += '\n...'
            lines.append(f"   Preview:\n   {preview_short}")

        return '\n'.join(lines)
