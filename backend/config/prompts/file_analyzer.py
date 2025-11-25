"""
Prompts for file analyzer LLM-powered analysis.
Provides specialized prompts for different file types.
"""

import os
from typing import Optional
from .base import section_border, MARKER_OK, MARKER_ERROR, get_current_time_context


def get_deep_analysis_prompt(file_path: str, user_query: Optional[str] = None) -> str:
    """
    Generate prompt for deep file analysis using LLM.

    Args:
        file_path: Path to the file being analyzed
        user_query: Optional user question about the file

    Returns:
        Deep analysis prompt string
    """
    filename = os.path.basename(file_path)
    query_section = f"\nUser question: {user_query}" if user_query else ""
    
    return f"""Analyze the structure of this file in detail:

File: {filename}
{query_section}

{section_border("ANALYSIS TASKS")}

1. Find maximum nesting depth
2. Map ALL key paths (e.g., data[0].user.profile.name)
3. Count items at each level
4. Show example values at leaf nodes
5. Identify nested dictionaries and lists
6. Show complete hierarchy

Output a comprehensive JSON structure report."""


def get_json_analysis_prompt(file_path: str, user_query: Optional[str] = None) -> str:
    """
    Specialized prompt for JSON file analysis.

    Args:
        file_path: Path to JSON file
        user_query: Optional user question
    """
    filename = os.path.basename(file_path)
    query_section = f"\nUser question: {user_query}" if user_query else ""
    
    return f"""Analyze this JSON file structure:

File: {filename}
{query_section}

{section_border("JSON ANALYSIS")}

1. Root type: dict or list?
2. If dict: List all top-level keys and their types
3. If list: Count items, show structure of first item
4. Nesting depth: How deep does it go?
5. Key paths: List all unique paths (e.g., data.users[].name)
6. Data types: What types appear? (string, number, bool, null, nested)
7. Sample values: Show 2-3 example values per path

{section_border("OUTPUT FORMAT")}

{{
  "root_type": "dict|list",
  "structure": {{
    "key_paths": ["path1", "path2"],
    "max_depth": 3,
    "item_count": 100
  }},
  "sample_data": {{
    "path1": "example_value"
  }},
  "access_patterns": [
    "data.get('key', default)",
    "data[0]['field']"
  ]
}}"""


def get_csv_analysis_prompt(file_path: str, user_query: Optional[str] = None) -> str:
    """
    Specialized prompt for CSV file analysis.

    Args:
        file_path: Path to CSV file
        user_query: Optional user question
    """
    filename = os.path.basename(file_path)
    query_section = f"\nUser question: {user_query}" if user_query else ""
    
    return f"""Analyze this CSV file structure:

File: {filename}
{query_section}

{section_border("CSV ANALYSIS")}

1. Column names: List all columns
2. Row count: Total number of rows
3. Data types: Infer type for each column (numeric, text, date, categorical)
4. Missing values: Count per column
5. Sample data: First 3-5 rows
6. Numeric columns: Min, max, mean if applicable
7. Categorical columns: Unique value counts

{section_border("OUTPUT FORMAT")}

{{
  "columns": ["col1", "col2"],
  "row_count": 1000,
  "column_types": {{"col1": "numeric", "col2": "text"}},
  "missing_values": {{"col1": 0, "col2": 5}},
  "numeric_stats": {{"col1": {{"min": 0, "max": 100, "mean": 50}}}},
  "categorical_summary": {{"col2": {{"unique": 10, "top_values": ["a", "b"]}}}}
}}"""


def get_excel_analysis_prompt(file_path: str, user_query: Optional[str] = None) -> str:
    """
    Specialized prompt for Excel file analysis.

    Args:
        file_path: Path to Excel file
        user_query: Optional user question
    """
    filename = os.path.basename(file_path)
    query_section = f"\nUser question: {user_query}" if user_query else ""
    
    return f"""Analyze this Excel file structure:

File: {filename}
{query_section}

{section_border("EXCEL ANALYSIS")}

1. Sheet names: List all sheets
2. Per sheet:
   - Column names
   - Row count
   - Data types per column
   - Missing values
3. Cross-sheet relationships: Any common columns?
4. Sample data: First 3 rows per sheet

{section_border("OUTPUT FORMAT")}

{{
  "sheets": ["Sheet1", "Sheet2"],
  "sheet_details": {{
    "Sheet1": {{
      "columns": ["A", "B"],
      "row_count": 500,
      "column_types": {{"A": "numeric"}},
      "sample_rows": [...]
    }}
  }},
  "relationships": ["Both sheets have 'ID' column"]
}}"""


def get_structure_comparison_prompt(
    file1_path: str,
    file2_path: str,
    user_query: Optional[str] = None
) -> str:
    """
    Prompt for comparing structure of two files.

    Args:
        file1_path: Path to first file
        file2_path: Path to second file
        user_query: Optional comparison question
    """
    file1 = os.path.basename(file1_path)
    file2 = os.path.basename(file2_path)
    query_section = f"\nComparison focus: {user_query}" if user_query else ""
    
    return f"""Compare the structure of these two files:

File 1: {file1}
File 2: {file2}
{query_section}

{section_border("COMPARISON TASKS")}

1. Structure similarity: Same schema/columns?
2. Common fields: What fields/columns exist in both?
3. Unique fields: What's unique to each file?
4. Data type differences: Same field, different types?
5. Size comparison: Row/item counts
6. Compatibility: Can they be merged/joined?

{section_border("OUTPUT FORMAT")}

{{
  "similarity_score": 0.85,
  "common_fields": ["id", "name"],
  "file1_only": ["extra_col"],
  "file2_only": ["other_col"],
  "type_differences": {{"field": {{"file1": "string", "file2": "number"}}}},
  "merge_compatible": true,
  "merge_key": "id"
}}"""


def get_anomaly_detection_prompt(file_path: str, user_query: Optional[str] = None) -> str:
    """
    Prompt for detecting anomalies in file data.

    Args:
        file_path: Path to file
        user_query: Optional focus area
    """
    filename = os.path.basename(file_path)
    query_section = f"\nFocus area: {user_query}" if user_query else ""
    
    return f"""Detect anomalies and data quality issues in this file:

File: {filename}
{query_section}

{section_border("ANOMALY DETECTION")}

1. Missing data patterns: Systematic gaps?
2. Outliers: Values far from normal range
3. Inconsistencies: Format variations (dates, numbers)
4. Duplicates: Repeated records
5. Invalid values: Out of expected range/type
6. Encoding issues: Character problems

{section_border("OUTPUT FORMAT")}

{{
  "data_quality_score": 0.92,
  "issues": [
    {{
      "type": "outlier",
      "field": "price",
      "description": "Value 999999 is 10x above mean",
      "severity": "high",
      "affected_rows": [45, 67]
    }}
  ],
  "recommendations": [
    "Review outliers in 'price' column",
    "Fill missing values in 'date' column"
  ]
}}"""


__all__ = [
    'get_deep_analysis_prompt',
    'get_json_analysis_prompt',
    'get_csv_analysis_prompt',
    'get_excel_analysis_prompt',
    'get_structure_comparison_prompt',
    'get_anomaly_detection_prompt',
]
