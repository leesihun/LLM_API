"""
Context Builder for Python Coder Tool

Builds rich context strings for LLM prompts with file metadata.
Extracted from python_coder_tool.py for better modularity.
"""

import json
from pathlib import Path
from typing import Any, Dict

from .file_handlers import FileHandlerFactory


class FileContextBuilder:
    """
    Builds context strings for LLM prompts based on file metadata.

    Uses FileHandlerFactory to get metadata and build detailed context sections
    for different file types. Optimized for performance using list + join pattern.
    """

    def __init__(self):
        """Initialize the file context builder."""
        self.file_handler_factory = FileHandlerFactory()

    def build_context(
        self,
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any]
    ) -> str:
        """
        Build file context string for LLM prompts.

        Args:
            validated_files: Dict mapping original file paths to filenames
            file_metadata: Metadata for files

        Returns:
            Formatted file context string
        """
        if not validated_files:
            return ""

        # Use list for efficient string building
        lines = []

        # Critical header warning
        lines.append("")
        lines.append("")
        lines.append("[!!!] CRITICAL - EXACT FILENAMES REQUIRED [!!!]")
        lines.append("ALL files are in the current working directory.")
        lines.append("YOU MUST use the EXACT filenames shown below - NO generic names like 'file.json' or 'data.csv'!")
        lines.append("")
        lines.append("Available files (USE THESE EXACT NAMES):")

        # Build context for each file
        for idx, (original_path, original_filename) in enumerate(validated_files.items(), 1):
            metadata = file_metadata.get(original_path, {})
            # Support both 'type' and 'file_type' field names
            file_type = metadata.get('type') or metadata.get('file_type', 'unknown')

            # File header
            lines.append(f"\n{idx}. \"{original_filename}\" - {file_type.upper()} ({metadata.get('size_mb', 0)}MB)")

            # Add relevant metadata based on file type
            self._add_file_metadata(lines, metadata, file_type)

            # Add file access examples
            self._add_file_access_examples(lines, file_type, original_filename, metadata)

        lines.append("")

        # Join all lines efficiently
        return '\n'.join(lines)

    def _add_file_metadata(
        self,
        lines: list,
        metadata: Dict[str, Any],
        file_type: str
    ) -> None:
        """
        Add file metadata to context lines.

        Args:
            lines: List to append lines to
            metadata: File metadata dictionary
            file_type: Type of file (csv, json, excel, etc.)
        """
        # CSV/Excel columns
        if 'columns' in metadata:
            cols = metadata['columns'][:10]
            line = f"   Columns: {', '.join(cols)}"
            if len(metadata.get('columns', [])) > 10:
                line += f" ... (+{len(metadata['columns']) - 10} more)"
            lines.append(line)

        # JSON structure
        if 'structure' in metadata:
            structure = metadata['structure']
            if isinstance(structure, dict):
                struct_type = structure.get('type', 'unknown')
                if struct_type == 'list':
                    lines.append(f"   Structure: {struct_type} ({structure.get('length', 0)} items)")
                elif struct_type == 'dict':
                    lines.append(f"   Structure: {struct_type} ({structure.get('num_keys', 0)} keys)")
                else:
                    lines.append(f"   Structure: {struct_type}")
            else:
                lines.append(f"   Structure: {structure}")

            # Detailed JSON structure information
            if file_type == 'json':
                self._add_json_metadata(lines, metadata)

        # Text file line count
        if 'line_count' in metadata:
            lines.append(f"   Lines: {metadata['line_count']}")

        # Text preview
        if 'preview' in metadata:
            preview_data = metadata['preview']
            # Handle different preview types (string vs list of dicts)
            if isinstance(preview_data, str):
                preview = preview_data[:100]
                lines.append(f"   Preview: {preview}...")
            elif isinstance(preview_data, list) and len(preview_data) > 0:
                # For CSV/Excel preview (list of dicts), show first row
                lines.append(f"   Preview: {len(preview_data)} sample rows available")
            # Otherwise skip preview display

        # Word document metadata
        if file_type == 'docx':
            self._add_docx_metadata(lines, metadata)

        # Enhanced Excel metadata
        if file_type == 'excel':
            self._add_excel_metadata(lines, metadata)

    def _add_json_metadata(self, lines: list, metadata: Dict[str, Any]) -> None:
        """
        Add JSON-specific metadata to context lines.

        Args:
            lines: List to append lines to
            metadata: File metadata dictionary
        """
        # Show top-level keys for objects
        structure = metadata.get('structure', {})
        if isinstance(structure, dict) and 'keys' in structure and structure['keys']:
            keys_display = structure['keys'][:10]  # Show up to 10 keys
            line = f"   Top-level keys: {', '.join(keys_display)}"
            if len(structure['keys']) > 10:
                line += f" ... (+{len(structure['keys']) - 10} more)"
            lines.append(line)

        # Show depth and item type info
        if 'max_depth' in metadata and metadata['max_depth'] > 1:
            lines.append(f"   Nesting depth: {metadata['max_depth']} levels")

        if 'first_item_type' in metadata and metadata['first_item_type']:
            lines.append(f"   Array items are: {metadata['first_item_type']}")

        # Show smart access patterns (most important!)
        if 'access_patterns' in metadata and metadata['access_patterns']:
            lines.append("   [PATTERNS] Access Patterns (COPY THESE EXACTLY):")
            # Show ALL patterns to reveal complete nested structure
            for pattern in metadata['access_patterns']:
                lines.append(f"      {pattern}")

        # Show safe preview (truncated to avoid context overflow)
        preview_data = metadata.get('preview')
        if preview_data and not isinstance(preview_data, str):
            try:
                preview_str = json.dumps(preview_data, indent=2, ensure_ascii=False)[:500]
                lines.append("   Sample Data (first few items):")
                for line in preview_str.split('\n')[:15]:
                    lines.append(f"      {line}")
            except:
                pass  # Skip if can't serialize

        # Warnings for special cases
        if metadata.get('requires_null_check'):
            lines.append("   [!] IMPORTANT: Contains null values - use .get() method for safe access")
        if metadata.get('max_depth', 0) > 3:
            lines.append("   [!] IMPORTANT: Deep nesting detected - validate each level before accessing")

    def _add_docx_metadata(self, lines: list, metadata: Dict[str, Any]) -> None:
        """
        Add Word document metadata to context lines.

        Args:
            lines: List to append lines to
            metadata: File metadata dictionary
        """
        if 'total_words' in metadata:
            lines.append(f"   Words: {metadata['total_words']}, Paragraphs: {metadata.get('total_paragraphs', 0)}")

        if 'total_tables' in metadata and metadata['total_tables'] > 0:
            lines.append(f"   Tables: {metadata['total_tables']} table(s)")
            if 'table_details' in metadata:
                for table in metadata['table_details'][:2]:  # Show first 2 tables
                    lines.append(f"      Table {table['table_number']}: {table['rows']} rows × {table['columns']} cols")

        if 'headings' in metadata and metadata['headings']:
            lines.append(f"   Headings: {len(metadata['headings'])} found")

        if 'text_preview' in metadata:
            lines.append(f"   Text preview: {metadata['text_preview'][:150]}...")

    def _add_excel_metadata(self, lines: list, metadata: Dict[str, Any]) -> None:
        """
        Add Excel-specific metadata to context lines.

        Args:
            lines: List to append lines to
            metadata: File metadata dictionary
        """
        if 'total_sheets' in metadata:
            sheets = ', '.join(metadata.get('sheet_names', [])[:3])
            lines.append(f"   Sheets ({metadata['total_sheets']} total): {sheets}")

        if metadata.get('has_formulas'):
            lines.append("   [!] Contains formulas - use data_only=True or read_excel()")

        if metadata.get('has_merged_cells'):
            lines.append("   [!] Contains merged cells - may affect data reading")

        if metadata.get('has_named_ranges'):
            lines.append("   Contains named ranges")

        # Show sheet details if available
        if 'sheets_analyzed' in metadata:
            for sheet in metadata['sheets_analyzed'][:2]:  # First 2 sheets
                lines.append(f"      Sheet '{sheet['sheet_name']}': {sheet['rows']} rows × {sheet['columns']} cols")
                if sheet.get('columns'):
                    cols_preview = ', '.join(sheet['columns'][:5])
                    lines.append(f"         Columns: {cols_preview}")

    def _add_file_access_examples(
        self,
        lines: list,
        file_type: str,
        original_filename: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add file access example code to context lines.

        Args:
            lines: List to append lines to
            file_type: Type of file (csv, json, excel, etc.)
            original_filename: Original filename
            metadata: File metadata dictionary
        """
        # CSV access example
        if file_type == 'csv':
            lines.append(f"   Example: df = pd.read_csv('{original_filename}')")

        # Word document access example
        elif file_type == 'docx':
            lines.append("   Example loading code:")
            lines.append("      from docx import Document")
            lines.append(f"      doc = Document('{original_filename}')")
            lines.append("      # Extract text: text = '\\n'.join([p.text for p in doc.paragraphs])")
            if metadata.get('total_tables', 0) > 0:
                lines.append("      # Extract tables: tables = doc.tables")

        # JSON access example
        elif file_type == 'json':
            lines.append("   Example loading code:")
            lines.append("      import json")
            lines.append(f"      with open('{original_filename}', 'r', encoding='utf-8') as f:")
            lines.append("          data = json.load(f)")

            # Add structure-specific access example
            if 'access_patterns' in metadata and metadata['access_patterns']:
                # Use the first access pattern as example
                first_pattern = metadata['access_patterns'][0]
                lines.append("      # Then use the access patterns above, e.g.:")
                lines.append(f"      # {first_pattern}")
            else:
                # Try to get keys from structure dict
                structure = metadata.get('structure', {})
                if isinstance(structure, dict) and 'keys' in structure and structure['keys']:
                    example_key = structure['keys'][0]
                    lines.append(f"      # Access: value = data.get('{example_key}', default)")
                elif 'first_item_type' in metadata and metadata['first_item_type']:
                    lines.append("      # Access: if len(data) > 0: item = data[0]")

            # Add error handling note if JSON had issues
            if 'error' in metadata:
                lines.append("   [!] CRITICAL: Wrap in try/except json.JSONDecodeError (file has parsing issues)")
            elif 'parsing_note' in metadata:
                lines.append(f"   [!] {metadata['parsing_note']}")

        # Excel access example
        elif file_type == 'excel':
            lines.append(f"   Example: df = pd.read_excel('{original_filename}')")
