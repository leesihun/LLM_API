"""JSON File Handler"""
from pathlib import Path
from typing import Any, Dict
import json
from backend.core.file_handlers.base import FileHandler

class JSONHandler(FileHandler):
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.json']
        self.file_type = 'json'
    
    def extract_metadata(self, file_path: Path, quick_mode: bool = False) -> Dict[str, Any]:
        metadata = {'file_type': 'json', 'file_size': self._get_file_size(file_path), 'error': None}
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            metadata['structure_type'] = type(data).__name__
            if isinstance(data, list):
                metadata['item_count'] = len(data)
                if len(data) > 0 and isinstance(data[0], dict):
                    metadata['keys'] = list(data[0].keys())
            elif isinstance(data, dict):
                metadata['keys'] = list(data.keys())
        except Exception as e:
            metadata['error'] = str(e)
        return metadata
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        result = {'format': 'JSON', 'filename': path.name}
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            result['structure'] = type(data).__name__
            result['preview'] = str(data)[:500]
        except Exception as e:
            result['error'] = str(e)
        return result
