"""Excel File Handler"""
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from backend.core.file_handlers.base import FileHandler

class ExcelHandler(FileHandler):
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.xlsx', '.xls']
        self.file_type = 'excel'
    
    def extract_metadata(self, file_path: Path, quick_mode: bool = False) -> Dict[str, Any]:
        metadata = {'file_type': 'excel', 'file_size': self._get_file_size(file_path), 'error': None}
        try:
            if not quick_mode:
                df = pd.read_excel(file_path)
                metadata.update({
                    'rows': len(df), 'columns': list(df.columns),
                    'dtypes': df.dtypes.astype(str).to_dict(),
                    'preview': df.head(3).to_dict('records')
                })
            else:
                df = pd.read_excel(file_path, nrows=0)
                metadata['columns'] = list(df.columns)
        except Exception as e:
            metadata['error'] = str(e)
        return metadata
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        result = {'format': 'Excel', 'filename': path.name}
        try:
            df = pd.read_excel(file_path)
            result.update({
                'rows': len(df), 'columns': list(df.columns),
                'preview': df.head(5).to_dict('records')
            })
        except Exception as e:
            result['error'] = str(e)
        return result
