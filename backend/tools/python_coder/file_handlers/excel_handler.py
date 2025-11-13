"""Excel file handler with specialized metadata extraction."""

from pathlib import Path
from typing import Any, Dict, List

from .base import BaseFileHandler


class ExcelFileHandler(BaseFileHandler):
    """Handler for Excel files (.xlsx, .xls)."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.xlsx', '.xls']

    def extract_metadata(
        self,
        file_path: Path,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """Extract metadata from Excel file."""
        metadata = {
            'file_type': 'excel',
            'file_size': self._get_file_size(file_path),
            'error': None
        }

        try:
            import pandas as pd

            # Get sheet names
            excel_file = pd.ExcelFile(file_path)
            metadata['sheet_names'] = excel_file.sheet_names
            metadata['num_sheets'] = len(excel_file.sheet_names)

            if not quick_mode:
                # Read first sheet for analysis
                df = pd.read_excel(file_path, sheet_name=0)
                metadata['first_sheet'] = {
                    'name': excel_file.sheet_names[0],
                    'rows': len(df),
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.astype(str).to_dict(),
                    'preview': df.head(3).to_dict('records')
                }

                # Null analysis
                null_counts = df.isnull().sum()
                metadata['null_analysis'] = {
                    col: int(count) for col, count in null_counts.items() if count > 0
                }
            else:
                # Quick mode: just sheet info
                metadata['first_sheet'] = {
                    'name': excel_file.sheet_names[0],
                    'rows': 'unknown',
                    'columns': []
                }

        except Exception as e:
            metadata['error'] = str(e)

        return metadata

    def build_context_section(
        self,
        filename: str,
        metadata: Dict[str, Any],
        index: int
    ) -> str:
        """Build context section for Excel file."""
        lines = []
        lines.append(f"{index}. {filename} (Excel)")

        if metadata.get('error'):
            lines.append(f"   Error: {metadata['error']}")
            return '\n'.join(lines)

        # Sheet information
        num_sheets = metadata.get('num_sheets', 0)
        sheet_names = metadata.get('sheet_names', [])
        lines.append(f"   Sheets: {num_sheets} ({', '.join(sheet_names)})")

        # First sheet details
        first_sheet = metadata.get('first_sheet', {})
        if first_sheet:
            lines.append(f"   First sheet: {first_sheet.get('name', 'unknown')}")

            if first_sheet.get('rows') != 'unknown':
                lines.append(f"   Rows: {first_sheet.get('rows', 0)}")

                columns = first_sheet.get('columns', [])
                if columns:
                    cols_str = ', '.join(str(c) for c in columns[:10])
                    if len(columns) > 10:
                        cols_str += f'... (+{len(columns) - 10} more)'
                    lines.append(f"   Columns: {cols_str}")

                # Data types
                dtypes = first_sheet.get('dtypes', {})
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

        # Preview
        preview_data = first_sheet.get('preview', [])
        if preview_data:
            lines.append(f"   Preview: {len(preview_data)} sample rows")

        return '\n'.join(lines)
