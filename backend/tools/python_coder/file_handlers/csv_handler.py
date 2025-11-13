"""CSV file handler with specialized metadata extraction."""

from pathlib import Path
from typing import Any, Dict

from .base import BaseFileHandler


class CSVFileHandler(BaseFileHandler):
    """Handler for CSV files."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.csv']

    def extract_metadata(
        self,
        file_path: Path,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """Extract metadata from CSV file."""
        metadata = {
            'file_type': 'csv',
            'file_size': self._get_file_size(file_path),
            'error': None
        }

        try:
            import pandas as pd

            if not quick_mode:
                # Full read
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
                        for col in list(numeric_cols)[:5]  # Limit to 5 columns
                    }
            else:
                # Quick mode: just read header
                df = pd.read_csv(file_path, nrows=0)
                metadata['columns'] = list(df.columns)
                metadata['rows'] = 'unknown'

        except Exception as e:
            metadata['error'] = str(e)

        return metadata

    def build_context_section(
        self,
        filename: str,
        metadata: Dict[str, Any],
        index: int
    ) -> str:
        """Build context section for CSV file."""
        lines = []
        lines.append(f"{index}. {filename} (CSV)")

        if metadata.get('error'):
            lines.append(f"   Error: {metadata['error']}")
            return '\n'.join(lines)

        # Basic info
        rows = metadata.get('rows', 'unknown')
        if rows != 'unknown':
            lines.append(f"   Rows: {rows}")

        columns = metadata.get('columns', [])
        if columns:
            cols_str = ', '.join(str(c) for c in columns[:10])
            if len(columns) > 10:
                cols_str += f'... (+{len(columns) - 10} more)'
            lines.append(f"   Columns: {cols_str}")

        # Data types
        dtypes = metadata.get('dtypes', {})
        if dtypes:
            dtype_summary = []
            for col, dtype in list(dtypes.items())[:5]:
                dtype_summary.append(f"{col}: {dtype}")
            lines.append(f"   Types: {', '.join(dtype_summary)}")

        # Null analysis
        null_analysis = metadata.get('null_analysis', {})
        if null_analysis:
            null_summary = []
            for col, count in list(null_analysis.items())[:3]:
                null_summary.append(f"{col}: {count}")
            lines.append(f"   Null values: {', '.join(null_summary)}")

        # Numeric statistics
        numeric_stats = metadata.get('numeric_stats', {})
        if numeric_stats:
            lines.append("   Numeric columns:")
            for col, stats in list(numeric_stats.items())[:2]:
                mean = stats.get('mean', 0)
                std = stats.get('std', 0)
                lines.append(f"     {col}: mean={mean:.2f}, std={std:.2f}")

        return '\n'.join(lines)
