"""
JSON File Handler
=================
Unified handler for JSON files supporting both metadata extraction and analysis.

Created: 2025-01-20
Version: 2.0.0
"""

from pathlib import Path
from typing import Any, Dict, List, Set
import json
from backend.core.file_handlers.base import FileHandler

class JSONHandler(FileHandler):
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.json']
        self.file_type = 'json'
    
    def extract_metadata(self, file_path: Path, quick_mode: bool = False) -> Dict[str, Any]:
        """Extract JSON metadata for code generation"""
        metadata = {
            'file_type': 'json',
            'file_size': self._get_file_size(file_path),
            'file_size_human': self._format_file_size(self._get_file_size(file_path)),
            'error': None
        }
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Structure type (list, dict, str, etc.)
            metadata['structure'] = type(data).__name__
            metadata['all_keys'] = self._collect_all_keys(data)

            # For lists of objects (most common case: like CSV rows)
            if isinstance(data, list):
                metadata['item_count'] = len(data)

                # Analyze first item structure if it's a dict
                if len(data) > 0 and isinstance(data[0], dict):
                    metadata['top_level_keys'] = list(data[0].keys())

                    if not quick_mode:
                        # Sample the first few items (for preview)
                        metadata['sample'] = data[:3] if len(data) >= 3 else data

                        # Analyze value types across multiple items for accuracy
                        metadata['value_types'] = self._analyze_value_types_from_list(data[:10])

                        # Get depth of nested structure
                        metadata['max_depth'] = self._get_max_depth(data[0])

                        # Schema inference (like CSV columns and dtypes)
                        metadata['schema'] = self._infer_schema_from_list(data[:100])

                        # Data quality metrics
                        quality_metrics = self._analyze_data_quality(data)
                        metadata.update(quality_metrics)
                else:
                    # List of primitives or mixed types
                    metadata['top_level_keys'] = []
                    if not quick_mode and data:
                        metadata['sample'] = data[:5] if len(data) >= 5 else data
                        metadata['item_types'] = list(set(type(item).__name__ for item in data[:100]))

            # For dictionaries
            elif isinstance(data, dict):
                metadata['top_level_keys'] = list(data.keys())
                metadata['key_count'] = len(data.keys())

                if not quick_mode:
                    # Sample data (first 3 keys or all if less than 3)
                    sample_keys = list(data.keys())[:3]
                    metadata['sample'] = {k: data[k] for k in sample_keys}

                    # Analyze value types
                    metadata['value_types'] = self._analyze_value_types(data)

                    # Get depth of nested structure
                    metadata['max_depth'] = self._get_max_depth(data)

            # For primitives (rare but possible)
            else:
                metadata['value'] = str(data)[:200]

        except json.JSONDecodeError as e:
            metadata['error'] = f"Invalid JSON: {str(e)}"
        except UnicodeDecodeError as e:
            metadata['error'] = f"Encoding error: {str(e)}"
        except Exception as e:
            metadata['error'] = str(e)

        return metadata

    def _analyze_value_types(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Analyze the types of values in a dictionary."""
        value_types = {}
        for key, value in data.items():
            if isinstance(value, dict):
                value_types[key] = f"dict ({len(value)} keys)"
            elif isinstance(value, list):
                value_types[key] = f"list ({len(value)} items)"
            else:
                value_types[key] = type(value).__name__
        return value_types

    def _get_max_depth(self, data: Any, current_depth: int = 0) -> int:
        """Recursively calculate maximum nesting depth."""
        if current_depth > 10:  # Safety limit
            return 10

        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._get_max_depth(v, current_depth + 1) for v in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._get_max_depth(item, current_depth + 1) for item in data[:5])  # Sample first 5
        else:
            return current_depth
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive JSON analysis (matches CSV/Excel level)"""
        path = Path(file_path)

        result = {
            'format': 'JSON',
            'file_path': str(path.absolute()),
            'filename': path.name,
            'size_bytes': self._get_file_size(path),
            'size_human': self._format_file_size(self._get_file_size(path))
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            result['structure'] = type(data).__name__
            result['all_keys'] = self._collect_all_keys(data)

            # Detailed analysis for list of objects
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                result['item_count'] = len(data)
                result['keys'] = list(data[0].keys())
                result['key_count'] = len(data[0].keys())

                # Schema inference
                result['schema'] = self._infer_schema_from_list(data[:100])

                # Value types
                result['value_types'] = self._analyze_value_types_from_list(data[:10])

                # Data quality
                quality = self._analyze_data_quality(data)
                result.update(quality)

                # Preview
                result['preview'] = data[:5] if len(data) >= 5 else data
                result['tail_preview'] = data[-2:] if len(data) >= 2 else []

                # Nesting depth
                result['max_depth'] = self._get_max_depth(data[0])

            # Detailed analysis for dictionaries
            elif isinstance(data, dict):
                result['top_level_keys'] = list(data.keys())
                result['key_count'] = len(data)
                result['value_types'] = self._analyze_value_types(data)
                result['max_depth'] = self._get_max_depth(data)

                # Sample (first 5 keys)
                sample_keys = list(data.keys())[:5]
                result['sample'] = {k: data[k] for k in sample_keys}

            # For other types
            else:
                result['preview'] = str(data)[:500]

        except json.JSONDecodeError as e:
            result['error'] = f"Invalid JSON: {str(e)}"
        except Exception as e:
            result['error'] = str(e)

        return result

    def _analyze_value_types_from_list(self, items: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Analyze value types across multiple items in a list.
        More accurate than analyzing just the first item.
        """
        if not items:
            return {}

        # Collect all keys from all items
        all_keys: Set[str] = set()
        for item in items:
            if isinstance(item, dict):
                all_keys.update(item.keys())

        value_types = {}
        for key in all_keys:
            types_seen = set()
            for item in items:
                if isinstance(item, dict) and key in item:
                    value = item[key]
                    if isinstance(value, dict):
                        types_seen.add(f"dict")
                    elif isinstance(value, list):
                        types_seen.add(f"list")
                    elif value is None:
                        types_seen.add("null")
                    else:
                        types_seen.add(type(value).__name__)

            # Format type info
            if len(types_seen) == 1:
                value_types[key] = list(types_seen)[0]
            else:
                value_types[key] = " | ".join(sorted(types_seen))

        return value_types

    def _infer_schema_from_list(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Infer schema from list of objects (similar to CSV dtypes).
        Returns a dictionary mapping keys to inferred types.
        """
        if not items:
            return {}

        # Collect all keys
        all_keys: Set[str] = set()
        for item in items:
            if isinstance(item, dict):
                all_keys.update(item.keys())

        schema = {}
        for key in all_keys:
            # Collect non-null values for this key
            values = []
            null_count = 0
            for item in items:
                if isinstance(item, dict):
                    if key in item:
                        if item[key] is None:
                            null_count += 1
                        else:
                            values.append(item[key])
                    else:
                        null_count += 1

            # Infer type
            if not values:
                schema[key] = "null"
            else:
                types_seen = set(type(v).__name__ for v in values)
                if len(types_seen) == 1:
                    type_name = list(types_seen)[0]
                    # Add nullable annotation if nulls exist
                    if null_count > 0:
                        schema[key] = f"{type_name} (nullable)"
                    else:
                        schema[key] = type_name
                else:
                    schema[key] = " | ".join(sorted(types_seen))

        return schema

    def _analyze_data_quality(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze data quality for list of objects.
        Returns metrics similar to CSV analysis (null counts, missing keys, etc.)
        """
        if not items or not isinstance(items[0], dict):
            return {}

        # Collect all possible keys
        all_keys: Set[str] = set()
        for item in items:
            if isinstance(item, dict):
                all_keys.update(item.keys())

        # Analyze each key
        missing_key_counts = {}
        null_value_counts = {}
        type_inconsistencies = {}

        for key in all_keys:
            missing_count = 0
            null_count = 0
            types_seen = set()

            for item in items:
                if isinstance(item, dict):
                    if key not in item:
                        missing_count += 1
                    elif item[key] is None:
                        null_count += 1
                    else:
                        types_seen.add(type(item[key]).__name__)

            # Record metrics
            if missing_count > 0:
                missing_key_counts[key] = missing_count

            if null_count > 0:
                null_value_counts[key] = null_count

            if len(types_seen) > 1:
                type_inconsistencies[key] = list(types_seen)

        result = {}

        if missing_key_counts:
            result['missing_keys'] = missing_key_counts

        if null_value_counts:
            result['null_values'] = null_value_counts

        if type_inconsistencies:
            result['type_inconsistencies'] = type_inconsistencies

        # Overall stats
        result['total_items'] = len(items)
        result['unique_keys'] = len(all_keys)

        return result

    def _collect_all_keys(self, data: Any, max_items: int = 50000, depth_limit: int = 20) -> List[str]:
        """
        Traverse JSON data and collect all dictionary keys without truncation,
        with safety caps to avoid runaway traversal.
        """
        collected = []
        seen = set()

        def visit(node: Any, depth: int, remaining: List[int]) -> None:
            if depth > depth_limit or remaining[0] <= 0:
                return
            if isinstance(node, dict):
                for k, v in node.items():
                    if k not in seen:
                        seen.add(k)
                        collected.append(k)
                    remaining[0] -= 1
                    if remaining[0] <= 0:
                        return
                    visit(v, depth + 1, remaining)
            elif isinstance(node, list):
                for item in node:
                    if remaining[0] <= 0:
                        return
                    remaining[0] -= 1
                    visit(item, depth + 1, remaining)

        visit(data, 0, [max_items])
        return collected

    def build_context_section(
        self,
        filename: str,
        metadata: Dict[str, Any],
        index: int
    ) -> str:
        """Build formatted context section for JSON files"""
        lines = []
        lines.append(f"{index}. {filename} (JSON)")

        if metadata.get('error'):
            lines.append(f"   Error: {metadata['error']}")
            return '\n'.join(lines)

        # Structure info
        structure = metadata.get('structure', 'unknown')
        lines.append(f"   Structure: {structure}")

        # For list of objects
        if structure == 'list':
            item_count = metadata.get('item_count', 0)
            if item_count:
                lines.append(f"   Items: {item_count:,}")

            keys = metadata.get('top_level_keys', [])
            if keys:
                lines.append(f"   Keys ({len(keys)}): {', '.join(str(k) for k in keys[:8])}")
                if len(keys) > 8:
                    lines.append(f"   ... and {len(keys) - 8} more keys")

            # Schema info
            schema = metadata.get('schema', {})
            if schema:
                lines.append("   Schema:")
                for key, dtype in list(schema.items())[:10]:
                    lines.append(f"     - {key}: {dtype}")
                if len(schema) > 10:
                    lines.append(f"     ... and {len(schema) - 10} more")

        # For dictionaries
        elif structure == 'dict':
            key_count = metadata.get('key_count', 0)
            if key_count:
                lines.append(f"   Keys: {key_count}")

            keys = metadata.get('top_level_keys', [])
            if keys:
                lines.append(f"   Top-level keys: {', '.join(str(k) for k in keys[:8])}")

        # File size
        if 'file_size_human' in metadata:
            lines.append(f"   Size: {metadata['file_size_human']}")

        return '\n'.join(lines)
