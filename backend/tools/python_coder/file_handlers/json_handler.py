"""JSON file handler with specialized metadata extraction and context building."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseFileHandler


class JSONFileHandler(BaseFileHandler):
    """Handler for JSON files with specialized analysis."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.json']

    def extract_metadata(
        self,
        file_path: Path,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """Extract metadata from JSON file."""
        metadata = {
            'file_type': 'json',
            'file_size': self._get_file_size(file_path),
            'error': None
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            metadata['structure'] = self._analyze_structure(data)

            if not quick_mode:
                metadata['access_patterns'] = self._generate_access_patterns(data)
                metadata['preview'] = self._create_safe_preview(data)
                metadata['null_check'] = self._check_for_nulls(data)
            else:
                # Quick mode: minimal info
                metadata['preview'] = str(data)[:200] + '...'

        except Exception as e:
            metadata['error'] = str(e)
            metadata['preview'] = f"Error loading JSON: {e}"

        return metadata

    def build_context_section(
        self,
        filename: str,
        metadata: Dict[str, Any],
        index: int
    ) -> str:
        """Build context section for JSON file."""
        lines = []
        lines.append(f"{index}. {filename} (JSON)")

        if metadata.get('error'):
            lines.append(f"   Error: {metadata['error']}")
            return '\n'.join(lines)

        # Structure info
        structure = metadata.get('structure', {})
        lines.append(f"   Type: {structure.get('type', 'unknown')}")

        if structure.get('type') == 'list':
            lines.append(f"   Length: {structure.get('length', 0)} items")
            if structure.get('item_type'):
                lines.append(f"   Item type: {structure['item_type']}")
        elif structure.get('type') == 'dict':
            lines.append(f"   Keys: {structure.get('num_keys', 0)}")
            if structure.get('keys'):
                keys_str = ', '.join(structure['keys'][:10])
                if len(structure['keys']) > 10:
                    keys_str += '...'
                lines.append(f"   Available keys: {keys_str}")

        # Access patterns - SHOW MORE to help LLM understand nested structure
        if metadata.get('access_patterns'):
            lines.append("   Recommended access patterns (deep structure):")
            # Show up to 12 patterns (was 3) to reveal full nesting
            for pattern in metadata['access_patterns'][:12]:
                lines.append(f"     - {pattern}")

        # Null check
        if metadata.get('null_check'):
            lines.append(f"   Null values: {metadata['null_check']}")

        # Preview
        if metadata.get('preview'):
            preview = str(metadata['preview'])[:300]
            lines.append(f"   Preview: {preview}")

        return '\n'.join(lines)

    def _analyze_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze JSON structure."""
        if isinstance(data, dict):
            return {
                'type': 'dict',
                'num_keys': len(data),
                'keys': list(data.keys()),
                'nested': any(isinstance(v, (dict, list)) for v in data.values())
            }
        elif isinstance(data, list):
            item_types = set(type(item).__name__ for item in data[:10])
            return {
                'type': 'list',
                'length': len(data),
                'item_type': ', '.join(item_types) if item_types else 'empty',
                'nested': any(isinstance(item, (dict, list)) for item in data[:10])
            }
        else:
            return {
                'type': type(data).__name__,
                'value': str(data)[:100]
            }

    def _generate_access_patterns(self, data: Any, max_patterns: int = 15) -> List[str]:
        """
        Generate recommended access patterns for the JSON data.

        IMPROVED: Now recursively explores nested structures up to 4-5 levels deep
        to help LLM understand the full data hierarchy and avoid accessing
        fields at wrong nesting levels.

        This is critical for complex JSON with nested arrays/objects (e.g.,
        departments[i].sales[j].revenue) where the LLM needs to see the full path.
        """
        patterns = []

        def explore_structure(obj: Any, path: str, depth: int = 0, max_depth: int = 5):
            """Recursively explore structure and collect access patterns."""
            if depth > max_depth or len(patterns) >= max_patterns:
                return

            # Add current path (skip empty root path)
            if path and path not in patterns:
                patterns.append(path)

            # Explore nested structures
            if isinstance(obj, dict):
                for key, value in list(obj.items())[:5]:  # Explore first 5 keys
                    new_path = f"{path}['{key}']" if path else f"data['{key}']"
                    explore_structure(value, new_path, depth + 1, max_depth)

            elif isinstance(obj, list) and len(obj) > 0:
                # Show array access pattern
                array_path = f"{path}[0]" if path else "data[0]"
                # Continue exploring inside the array item
                explore_structure(obj[0], array_path, depth + 1, max_depth)

        # Start exploration (depth=0 is root, max_depth=5 allows deep nesting)
        explore_structure(data, "", depth=0, max_depth=5)

        # Ensure we have at least top-level patterns
        if not patterns and isinstance(data, dict):
            patterns = [f"data['{key}']" for key in list(data.keys())[:max_patterns]]
        elif not patterns and isinstance(data, list) and data:
            patterns = ["data[0]"]

        return patterns[:max_patterns]

    def _create_safe_preview(
        self,
        data: Any,
        max_depth: int = 4,  # IMPROVED: Increased from 2 to 4 to show deeper nesting
        max_items: int = 3,
        max_size: int = 2000  # IMPROVED: Increased from 1000 to 2000 for more context
    ) -> Any:
        """
        Create a safe preview of JSON data with size limits.

        IMPROVED: Increased max_depth from 2 to 4 and max_size from 1000 to 2000
        to provide LLM with better visibility into nested structures.
        """
        # Check total size
        preview_str = str(data)
        if len(preview_str) > max_size:
            return preview_str[:max_size] + "... (truncated)"

        return self._truncate_data(data, max_depth, max_items, current_depth=0)

    def _truncate_data(
        self,
        data: Any,
        max_depth: int,
        max_items: int,
        current_depth: int = 0
    ) -> Any:
        """Recursively truncate data for preview."""
        if current_depth >= max_depth:
            return "... (max depth reached)"

        if isinstance(data, dict):
            truncated = {}
            for i, (key, value) in enumerate(data.items()):
                if i >= max_items:
                    truncated['...'] = f"({len(data) - max_items} more keys)"
                    break
                truncated[key] = self._truncate_data(
                    value, max_depth, max_items, current_depth + 1
                )
            return truncated

        elif isinstance(data, list):
            truncated = []
            for i, item in enumerate(data):
                if i >= max_items:
                    truncated.append(f"... ({len(data) - max_items} more items)")
                    break
                truncated.append(self._truncate_data(
                    item, max_depth, max_items, current_depth + 1
                ))
            return truncated

        else:
            # Primitive types
            return data

    def _check_for_nulls(self, data: Any, path: str = "root") -> str:
        """Check for null/None values in JSON data."""
        null_count = 0
        null_paths = []

        def check_recursive(obj: Any, current_path: str):
            nonlocal null_count, null_paths

            if obj is None:
                null_count += 1
                if len(null_paths) < 5:  # Limit examples
                    null_paths.append(current_path)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    check_recursive(value, f"{current_path}['{key}']")
            elif isinstance(obj, list):
                for i, item in enumerate(obj[:10]):  # Check first 10 items
                    check_recursive(item, f"{current_path}[{i}]")

        check_recursive(data, path)

        if null_count == 0:
            return "No null values found"
        else:
            examples = ", ".join(null_paths)
            return f"Found {null_count} null values (examples: {examples})"
