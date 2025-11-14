"""
JSON File Handler
==================
Handler for analyzing JSON files with deep structure exploration.

Version: 1.0.0
Created: 2025-01-13
"""

import json
from typing import Dict, Any, List, Optional

from backend.utils.logging_utils import get_logger
from ..base_handler import BaseFileHandler

logger = get_logger(__name__)


class JSONHandler(BaseFileHandler):
    """
    Handler for JSON file analysis.

    Features:
    - Deep recursive structure exploration
    - Maximum depth detection
    - Complete path enumeration
    - Schema inference
    - Human-readable structure summary
    - Support for nested dictionaries and arrays
    """

    def __init__(self):
        """Initialize JSON handler."""
        self.supported_extensions = ['json']

    def supports(self, file_path: str) -> bool:
        """
        Check if this is a JSON file.

        Args:
            file_path: Path to the file

        Returns:
            True if file has .json extension
        """
        extension = self.get_file_extension(file_path)
        return extension in self.supported_extensions

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported extensions."""
        return self.supported_extensions

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze JSON file with deep structure exploration.

        Args:
            file_path: Path to the JSON file

        Returns:
            Dictionary with analysis results including:
            - format: 'JSON'
            - structure: Type of root element (dict/list)
            - items_count: Number of items (for lists)
            - keys: Keys in root object (for dicts)
            - preview: Sample data
            - depth_analysis: Recursive structure information
            - structure_summary: Human-readable summary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            result = {
                "format": "JSON",
                "structure": type(data).__name__
            }

            # Basic structure info
            if isinstance(data, list):
                result["items_count"] = len(data)
                if len(data) > 0:
                    result["first_item_type"] = type(data[0]).__name__
                    result["preview"] = data[:]
                    if isinstance(data[0], dict):
                        result["keys"] = list(data[0].keys())
            elif isinstance(data, dict):
                result["keys"] = list(data.keys())
                result["preview"] = {k: data[k] for k in list(data.keys())[:]}
            else:
                result["value"] = str(data)[:]

            # Deep structure analysis (recursive exploration)
            structure_info = self._explore_json_structure(data)
            result["depth_analysis"] = structure_info

            # Add human-readable structure summary
            result["structure_summary"] = self._generate_structure_summary(structure_info)

            return result

        except json.JSONDecodeError as e:
            return {"format": "JSON", "error": f"Invalid JSON: {str(e)}"}
        except Exception as e:
            logger.error(f"JSON analysis error: {e}", exc_info=True)
            return {"format": "JSON", "error": str(e)}

    def _explore_json_structure(
        self, obj: Any, current_depth: int = 0, max_depth: int = 10, path: str = "root"
    ) -> Dict[str, Any]:
        """
        Recursively explore JSON structure to find all nested levels.

        Args:
            obj: Current object to analyze
            current_depth: Current nesting depth
            max_depth: Maximum depth to explore
            path: Current path in the structure

        Returns:
            Dictionary with structure information
        """
        if current_depth > max_depth:
            return {
                "type": "max_depth_reached",
                "depth": current_depth,
                "path": path
            }

        if isinstance(obj, dict):
            keys = list(obj.keys())
            children = {}

            # Analyze ALL keys (complete structure exploration)
            for key in keys:
                child_path = f"{path}.{key}"
                children[key] = self._explore_json_structure(
                    obj[key], current_depth + 1, max_depth, child_path
                )

            return {
                "type": "dict",
                "depth": current_depth,
                "path": path,
                "keys": keys,
                "key_count": len(keys),
                "children": children
            }

        elif isinstance(obj, list):
            length = len(obj)
            sample_item = None

            # Analyze first item as representative
            if length > 0:
                sample_path = f"{path}[0]"
                sample_item = self._explore_json_structure(
                    obj[0], current_depth + 1, max_depth, sample_path
                )

            return {
                "type": "list",
                "depth": current_depth,
                "path": path,
                "length": length,
                "sample_item": sample_item
            }

        else:
            # Leaf node (primitive type)
            return {
                "type": type(obj).__name__,
                "depth": current_depth,
                "path": path,
                "value_sample": str(obj)[:50] if obj is not None else None
            }

    def _generate_structure_summary(self, structure_info: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of JSON structure.

        Args:
            structure_info: Structure information from _explore_json_structure

        Returns:
            Human-readable summary string
        """
        summary_lines = []
        max_depth = self._find_max_depth(structure_info)
        all_paths = self._collect_all_paths(structure_info)

        summary_lines.append(f"Maximum nesting depth: {max_depth}")
        summary_lines.append(f"Total unique paths: {len(all_paths)}")
        summary_lines.append("")
        summary_lines.append("Structure hierarchy:")

        # Show example paths at different depths
        paths_by_depth = {}
        for path_info in all_paths:
            depth = path_info["depth"]
            if depth not in paths_by_depth:
                paths_by_depth[depth] = []
            paths_by_depth[depth].append(path_info)

        for depth in sorted(paths_by_depth.keys()):
            paths = paths_by_depth[depth]
            summary_lines.append(f"  Depth {depth}: {len(paths)} node(s)")
            # Show ALL paths at each depth
            for path_info in paths:
                type_str = path_info["type"]
                path_str = path_info["path"]
                if type_str == "list":
                    summary_lines.append(f"    - {path_str} (array, {path_info.get('length', 0)} items)")
                elif type_str == "dict":
                    summary_lines.append(f"    - {path_str} (object, {path_info.get('key_count', 0)} keys)")
                else:
                    summary_lines.append(f"    - {path_str} ({type_str})")

        return "\n".join(summary_lines)

    def _find_max_depth(self, structure_info: Dict[str, Any]) -> int:
        """
        Find maximum depth in structure.

        Args:
            structure_info: Structure information dictionary

        Returns:
            Maximum depth as integer
        """
        max_depth = structure_info.get("depth", 0)

        if structure_info.get("type") == "dict" and "children" in structure_info:
            for child in structure_info["children"].values():
                child_max = self._find_max_depth(child)
                max_depth = max(max_depth, child_max)
        elif structure_info.get("type") == "list" and "sample_item" in structure_info:
            if structure_info["sample_item"]:
                child_max = self._find_max_depth(structure_info["sample_item"])
                max_depth = max(max_depth, child_max)

        return max_depth

    def _collect_all_paths(
        self, structure_info: Dict[str, Any], paths: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect all paths in the structure.

        Args:
            structure_info: Structure information dictionary
            paths: Accumulated paths list

        Returns:
            List of all path dictionaries
        """
        if paths is None:
            paths = []

        path_entry = {
            "path": structure_info.get("path", "root"),
            "type": structure_info.get("type", "unknown"),
            "depth": structure_info.get("depth", 0)
        }

        if structure_info.get("type") == "list":
            path_entry["length"] = structure_info.get("length", 0)
        elif structure_info.get("type") == "dict":
            path_entry["key_count"] = structure_info.get("key_count", 0)

        paths.append(path_entry)

        # Recurse into children
        if structure_info.get("type") == "dict" and "children" in structure_info:
            for child in structure_info["children"].values():
                self._collect_all_paths(child, paths)
        elif structure_info.get("type") == "list" and "sample_item" in structure_info:
            if structure_info["sample_item"]:
                self._collect_all_paths(structure_info["sample_item"], paths)

        return paths
