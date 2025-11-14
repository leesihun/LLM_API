"""
CSV File Handler
=================
Handler for analyzing CSV and TSV files with advanced data quality profiling.

Version: 1.0.0
Created: 2025-01-13
"""

import csv
from typing import Dict, Any, List
from pathlib import Path

from backend.utils.logging_utils import get_logger
from ..base_handler import BaseFileHandler

logger = get_logger(__name__)


class CSVHandler(BaseFileHandler):
    """
    Handler for CSV/TSV file analysis.

    Features:
    - Auto-detect delimiter (comma, tab, semicolon, etc.)
    - Multi-encoding support (UTF-8, CP949, EUC-KR, Latin1, etc.)
    - Data quality profiling (null values, duplicates)
    - Column type detection
    - Statistical summaries for numeric columns
    """

    def __init__(self):
        """Initialize CSV handler."""
        self.supported_extensions = ['csv', 'tsv']

    def supports(self, file_path: str) -> bool:
        """
        Check if this is a CSV/TSV file.

        Args:
            file_path: Path to the file

        Returns:
            True if file has .csv or .tsv extension
        """
        extension = self.get_file_extension(file_path)
        return extension in self.supported_extensions

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported extensions."""
        return self.supported_extensions

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze CSV file with comprehensive metadata extraction.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary with analysis results including:
            - format: 'CSV'
            - encoding: Detected encoding
            - delimiter: Detected delimiter
            - rows: Row count
            - columns: Column count
            - column_names: List of column names
            - column_types: Dict mapping column names to data types
            - preview: Sample rows (first 5)
            - null_counts: Null values per column
            - duplicate_rows: Count of duplicate rows
            - numeric_columns: List of numeric column names
            - memory_usage: Memory footprint
        """
        try:
            import pandas as pd

            # Auto-detect delimiter
            delimiter = self._detect_delimiter(file_path)

            # Detect encoding
            encoding, df_sample = self._read_with_encoding(file_path, delimiter)

            if df_sample is None:
                return {
                    "format": "CSV",
                    "error": "Failed to read CSV with any encoding"
                }

            # Read full dataset
            full_df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)

            # Data quality profiling
            null_counts = full_df.isnull().sum().to_dict()
            total_nulls = sum(null_counts.values())
            duplicate_rows = full_df.duplicated().sum()

            # Detect numeric columns
            numeric_cols = full_df.select_dtypes(include=['number']).columns.tolist()

            return {
                "format": "CSV",
                "encoding": encoding,
                "delimiter": repr(delimiter),  # Show special chars like \t
                "rows": len(full_df),
                "columns": len(full_df.columns),
                "column_names": list(full_df.columns),
                "column_types": {col: str(dtype) for col, dtype in full_df.dtypes.items()},
                "preview": full_df.head(5).to_dict(orient='records'),
                "null_counts": null_counts,
                "total_null_values": int(total_nulls),
                "null_percentage": round(
                    total_nulls / (len(full_df) * len(full_df.columns)) * 100, 2
                ) if len(full_df) > 0 else 0,
                "duplicate_rows": int(duplicate_rows),
                "numeric_columns": numeric_cols,
                "memory_usage": f"{full_df.memory_usage(deep=True).sum() / 1024:.2f} KB"
            }

        except ImportError:
            return {"format": "CSV", "error": "pandas not installed"}
        except Exception as e:
            logger.error(f"CSV analysis error: {e}", exc_info=True)
            return {"format": "CSV", "error": str(e)}

    def _detect_delimiter(self, file_path: str) -> str:
        """
        Auto-detect CSV delimiter.

        Args:
            file_path: Path to CSV file

        Returns:
            Detected delimiter character
        """
        delimiter = ','  # Default

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(10240)  # Read first 10KB
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
        except Exception:
            pass  # Use default comma

        return delimiter

    def _read_with_encoding(self, file_path: str, delimiter: str) -> tuple:
        """
        Try reading CSV with multiple encodings.

        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter

        Returns:
            Tuple of (encoding_used, dataframe) or (None, None) if all fail
        """
        import pandas as pd

        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, nrows=100, delimiter=delimiter)
                return encoding, df
            except Exception:
                continue

        return None, None
