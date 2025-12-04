"""
CSV File Handler
================
Unified handler for CSV files supporting both metadata extraction and analysis.

Created: 2025-01-20
Version: 1.0.0
"""

from pathlib import Path
from typing import Any, Dict
import pandas as pd

from backend.core.file_handlers.base import FileHandler


class CSVHandler(FileHandler):
    """Handler for CSV files"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.csv']
        self.file_type = 'csv'
    
    def extract_metadata(
        self,
        file_path: Path,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """Extract CSV metadata for code generation"""
        metadata = {
            'file_type': 'csv',
            'file_size': self._get_file_size(file_path),
            'file_size_human': self._format_file_size(self._get_file_size(file_path)),
            'error': None
        }
        
        try:
            if not quick_mode:
                # Full read with analysis
                df = pd.read_csv(file_path)
                metadata['rows'] = len(df)
                metadata['columns'] = list(df.columns)
                metadata['dtypes'] = df.dtypes.astype(str).to_dict()
                metadata['preview'] = df.head(3).to_dict('records')
                
                # Null analysis
                null_counts = df.isnull().sum()
                metadata['null_analysis'] = {
                    col: int(count) for col, count in null_counts.items() if count > 0
                }
                
                # Basic statistics for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    stats = df[numeric_cols].describe().to_dict()
                    metadata['numeric_stats'] = {
                        col: {k: float(v) for k, v in stats[col].items()}
                        for col in list(numeric_cols)[:5]
                    }
            else:
                # Quick mode: just header
                df = pd.read_csv(file_path, nrows=0)
                metadata['columns'] = list(df.columns)
                metadata['rows'] = 'unknown'
                
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive CSV analysis"""
        path = Path(file_path)
        
        result = {
            'format': 'CSV',
            'file_path': str(path.absolute()),
            'filename': path.name,
            'size_bytes': self._get_file_size(path),
            'size_human': self._format_file_size(self._get_file_size(path))
        }
        
        try:
            df = pd.read_csv(file_path)
            
            result.update({
                'rows': len(df),
                'columns': list(df.columns),
                'column_count': len(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'preview': df.head(5).to_dict('records'),
                'tail_preview': df.tail(2).to_dict('records'),
            })
            
            # Data quality metrics
            result['null_counts'] = df.isnull().sum().to_dict()
            result['duplicate_rows'] = int(df.duplicated().sum())
            
            # Statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                result['numeric_columns'] = numeric_cols
                result['statistics'] = df[numeric_cols].describe().to_dict()
            
            # Categorical columns
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                result['categorical_columns'] = cat_cols
                result['unique_counts'] = {
                    col: int(df[col].nunique()) for col in cat_cols[:10]
                }
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def build_context_section(
        self,
        filename: str,
        metadata: Dict[str, Any],
        index: int
    ) -> str:
        """Build context section for CSV"""
        lines = []
        lines.append(f"{index}. {filename} (CSV)")
        
        if metadata.get('error'):
            lines.append(f"   Error: {metadata['error']}")
            return '\n'.join(lines)
        
        # Basic info
        rows = metadata.get('rows', 'unknown')
        if rows != 'unknown':
            lines.append(f"   Rows: {rows:,}")
        
        columns = metadata.get('columns', [])
        if columns:
            lines.append(f"   Columns ({len(columns)}): {', '.join(str(c) for c in columns[:8])}")
            if len(columns) > 8:
                lines.append(f"   ... and {len(columns) - 8} more columns")
        
        # Data types
        dtypes = metadata.get('dtypes', {})
        if dtypes:
            type_summary = {}
            for dtype in dtypes.values():
                type_summary[dtype] = type_summary.get(dtype, 0) + 1
            lines.append(f"   Types: {', '.join(f'{k}: {v}' for k, v in type_summary.items())}")
        
        return '\n'.join(lines)
