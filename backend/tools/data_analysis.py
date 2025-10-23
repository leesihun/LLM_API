"""
Data Analysis Tool
Provides statistical analysis functions for JSON data
"""

import json
import statistics
from typing import Any, Dict, List, Union, Optional
from pathlib import Path

from backend.config.settings import settings


class DataAnalysisTool:
    """Tool for analyzing JSON data with statistical functions"""

    def __init__(self):
        self.uploads_path = Path(settings.uploads_path)

    async def analyze_json(self, query: str) -> str:
        """
        Analyze JSON data based on user query

        Supports operations:
        - min: Find minimum value
        - max: Find maximum value
        - mean/average: Calculate average
        - sum: Calculate sum
        - count: Count elements
        """
        try:
            # Parse query to extract operation and field
            operation, field_name = self._parse_query(query)

            # Load all JSON files
            data_values = await self._extract_values_from_uploads(field_name)

            if not data_values:
                return f"No numeric data found for field '{field_name}'"

            # Perform requested operation
            result = self._perform_operation(operation, data_values, field_name)

            return result

        except Exception as e:
            return f"Error analyzing data: {str(e)}"

    def _parse_query(self, query: str) -> tuple[str, str]:
        """
        Parse query to extract operation and field name

        Examples:
        - "find max price" -> ("max", "price")
        - "calculate mean age" -> ("mean", "age")
        - "get minimum value" -> ("min", "value")
        """
        query_lower = query.lower()

        # Determine operation
        operation = "unknown"
        if any(word in query_lower for word in ["max", "maximum", "highest", "largest"]):
            operation = "max"
        elif any(word in query_lower for word in ["min", "minimum", "lowest", "smallest"]):
            operation = "min"
        elif any(word in query_lower for word in ["mean", "average", "avg"]):
            operation = "mean"
        elif any(word in query_lower for word in ["sum", "total"]):
            operation = "sum"
        elif any(word in query_lower for word in ["count", "number of", "how many"]):
            operation = "count"

        # Extract field name (simple heuristic: last word or specific keywords)
        words = query_lower.split()
        field_name = None

        # Look for common field indicators
        for i, word in enumerate(words):
            if word in ["of", "for", "in"] and i + 1 < len(words):
                field_name = words[i + 1]
                break

        # If not found, use last word
        if not field_name and words:
            # Skip common operation words
            skip_words = {"find", "get", "calculate", "show", "what", "is", "the",
                         "max", "min", "mean", "sum", "count", "average", "value"}
            for word in reversed(words):
                if word not in skip_words:
                    field_name = word
                    break

        if not field_name:
            field_name = "value"

        return operation, field_name

    async def _extract_values_from_uploads(self, field_name: str) -> List[Union[int, float]]:
        """Extract numeric values for a specific field from all uploaded JSON files"""
        values = []

        if not self.uploads_path.exists():
            return values

        for json_file in self.uploads_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Extract values based on field name
                    extracted = self._extract_field_values(data, field_name)
                    values.extend(extracted)

            except Exception as e:
                continue

        return values

    def _extract_field_values(self, data: Any, field_name: str) -> List[Union[int, float]]:
        """
        Recursively extract numeric values for a field from JSON data
        """
        values = []

        if isinstance(data, dict):
            # Check if field exists at this level
            if field_name in data:
                value = data[field_name]
                if isinstance(value, (int, float)):
                    values.append(value)
                elif isinstance(value, list):
                    # If it's a list of numbers
                    values.extend([v for v in value if isinstance(v, (int, float))])

            # Recursively search in nested dictionaries
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    values.extend(self._extract_field_values(value, field_name))

        elif isinstance(data, list):
            # Recursively search in list items
            for item in data:
                if isinstance(item, (dict, list)):
                    values.extend(self._extract_field_values(item, field_name))

        return values

    def _perform_operation(self, operation: str, values: List[Union[int, float]], field_name: str) -> str:
        """Perform statistical operation on values"""

        if not values:
            return f"No numeric values found for '{field_name}'"

        try:
            if operation == "max":
                result = max(values)
                return f"Maximum {field_name}: {result} (found {len(values)} values)"

            elif operation == "min":
                result = min(values)
                return f"Minimum {field_name}: {result} (found {len(values)} values)"

            elif operation == "mean":
                result = statistics.mean(values)
                return f"Mean {field_name}: {result:.2f} (from {len(values)} values)"

            elif operation == "sum":
                result = sum(values)
                return f"Sum of {field_name}: {result} (from {len(values)} values)"

            elif operation == "count":
                return f"Count of {field_name} values: {len(values)}"

            else:
                # Default: return basic statistics
                return f"""Statistics for '{field_name}':
- Count: {len(values)}
- Min: {min(values)}
- Max: {max(values)}
- Mean: {statistics.mean(values):.2f}
- Sum: {sum(values)}"""

        except Exception as e:
            return f"Error calculating {operation}: {str(e)}"

    async def get_all_statistics(self, field_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a field"""
        values = await self._extract_values_from_uploads(field_name)

        if not values:
            return {"error": f"No numeric data found for field '{field_name}'"}

        try:
            stats = {
                "field": field_name,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values) if len(values) > 0 else None,
                "sum": sum(values),
            }

            # Add standard deviation if we have enough values
            if len(values) > 1:
                stats["stdev"] = statistics.stdev(values)

            return stats

        except Exception as e:
            return {"error": f"Error calculating statistics: {str(e)}"}


# Global instance
data_analysis_tool = DataAnalysisTool()
