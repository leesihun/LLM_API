"""Text File Handler"""
from pathlib import Path
from typing import Any, Dict
from backend.core.file_handlers.base import FileHandler

class TextHandler(FileHandler):
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.txt', '.md', '.log']
        self.file_type = 'text'
    
    def extract_metadata(self, file_path: Path, quick_mode: bool = False) -> Dict[str, Any]:
        metadata = {'file_type': 'text', 'file_size': self._get_file_size(file_path), 'error': None}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            metadata['line_count'] = len(lines)
            metadata['preview'] = ''.join(lines[:5])
        except Exception as e:
            metadata['error'] = str(e)
        return metadata
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        result = {'format': 'Text', 'filename': path.name}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            result['lines'] = len(content.split('
'))
            result['characters'] = len(content)
            result['preview'] = content[:500]
        except Exception as e:
            result['error'] = str(e)
        return result
