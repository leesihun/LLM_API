"""
Summary Generator
==================
Helper module for generating human-readable analysis summaries.

Version: 1.0.0
Created: 2025-01-13
"""

from typing import List, Dict, Any


class SummaryGenerator:
    """
    Generates human-readable summaries of file analysis results.

    Provides format-specific detail extraction for different file types
    including CSV, Excel, JSON, Word documents, PDF, images, and text files.
    """

    @staticmethod
    def generate_summary(results: List[Dict[str, Any]], user_query: str = "") -> str:
        """
        Generate a human-readable summary of analysis results.

        Args:
            results: List of analysis result dictionaries
            user_query: User's original query

        Returns:
            Human-readable summary string
        """
        summary_lines = []

        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        summary_lines.append(f"[File Analysis Summary] ({len(successful)}/{len(results)} successful)")
        summary_lines.append("")

        for result in successful:
            file_name = result.get("file", "Unknown")
            format_type = result.get("format", "Unknown")
            size = result.get("size_human", "Unknown size")

            summary_lines.append(f"- {file_name} ({format_type}, {size})")

            # Add format-specific details
            SummaryGenerator._add_format_details(summary_lines, result, format_type)
            summary_lines.append("")

        if failed:
            summary_lines.append("[!] Failed files:")
            for result in failed:
                summary_lines.append(
                    f"- {result.get('file', 'Unknown')}: {result.get('error', 'Unknown error')}"
                )

        return "\n".join(summary_lines)

    @staticmethod
    def _add_format_details(
        summary_lines: List[str], result: Dict[str, Any], format_type: str
    ):
        """
        Add format-specific details to summary.

        Args:
            summary_lines: List to append summary lines to
            result: Analysis result dictionary
            format_type: Format type string
        """
        if format_type == "CSV":
            SummaryGenerator._add_csv_details(summary_lines, result)
        elif format_type == "Excel":
            SummaryGenerator._add_excel_details(summary_lines, result)
        elif format_type.startswith("Word Document"):
            SummaryGenerator._add_word_details(summary_lines, result)
        elif format_type == "JSON":
            SummaryGenerator._add_json_details(summary_lines, result)
        elif format_type == "Text":
            SummaryGenerator._add_text_details(summary_lines, result)
        elif "Image" in format_type:
            SummaryGenerator._add_image_details(summary_lines, result)

    @staticmethod
    def _add_csv_details(summary_lines: List[str], result: Dict[str, Any]):
        """Add CSV-specific details."""
        rows = result.get("rows", 0)
        cols = result.get("columns", 0)
        summary_lines.append(f"  - {rows:,} rows × {cols} columns")
        summary_lines.append(f"  - Columns: {', '.join(result.get('column_names', [])[:5])}")

        if result.get("duplicate_rows", 0) > 0:
            summary_lines.append(f"  - Duplicate rows: {result.get('duplicate_rows')}")

        null_pct = result.get("null_percentage", 0)
        if null_pct > 0:
            summary_lines.append(f"  - Missing data: {null_pct}%")

    @staticmethod
    def _add_excel_details(summary_lines: List[str], result: Dict[str, Any]):
        """Add Excel-specific details."""
        sheets = result.get("total_sheets", 0)
        sheet_names = ', '.join(result.get('sheet_names', [])[:3])
        summary_lines.append(f"  - {sheets} sheet(s): {sheet_names}")

        if result.get("has_formulas"):
            summary_lines.append(f"  - Contains formulas")

        if result.get("has_named_ranges"):
            range_count = len(result.get('named_ranges', []))
            summary_lines.append(f"  - Contains named ranges: {range_count}")

        if result.get("has_merged_cells"):
            summary_lines.append(f"  - Contains merged cells")

    @staticmethod
    def _add_word_details(summary_lines: List[str], result: Dict[str, Any]):
        """Add Word document-specific details."""
        words = result.get("total_words", 0)
        tables = result.get("total_tables", 0)
        images = result.get("total_images", 0)
        headings = result.get("total_headings", 0)

        summary_lines.append(f"  - {words:,} words, {tables} table(s), {images} image(s)")

        if headings > 0:
            summary_lines.append(f"  - {headings} heading(s)")

    @staticmethod
    def _add_json_details(summary_lines: List[str], result: Dict[str, Any]):
        """Add JSON-specific details."""
        structure = result.get("structure", "Unknown")
        summary_lines.append(f"  - Structure: {structure}")

        if "items_count" in result:
            summary_lines.append(f"  - Items: {result['items_count']}")

    @staticmethod
    def _add_text_details(summary_lines: List[str], result: Dict[str, Any]):
        """Add text file-specific details."""
        lines = result.get("total_lines", 0)
        words = result.get("total_words", 0)
        summary_lines.append(f"  - {lines:,} lines, {words:,} words")

    @staticmethod
    def _add_image_details(summary_lines: List[str], result: Dict[str, Any]):
        """Add image-specific details."""
        dims = result.get("dimensions", "Unknown")
        summary_lines.append(f"  - Dimensions: {dims}")
