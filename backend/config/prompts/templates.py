"""
Code Templates for Python Coder Prompts

Provides concrete, copy-paste ready code templates that LLM can use as starting points.
These templates dramatically reduce errors by showing exact patterns to follow.

Created: 2025-11-20
"""

from typing import List, Dict, Any


def get_json_loading_template(filename: str) -> str:
    """Get template for loading JSON file safely."""
    return f"""import json

# Load JSON file
filename = "{filename}"
try:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {{type(data).__name__}} with {{len(data) if isinstance(data, (list, dict)) else 0}} items")
except json.JSONDecodeError as e:
    print(f"JSON Error: {{e}}")
    exit(1)
except FileNotFoundError:
    print(f"File not found: {{filename}}")
    exit(1)"""


def get_csv_loading_template(filename: str) -> str:
    """Get template for loading CSV file."""
    return f"""import pandas as pd

# Load CSV file
filename = "{filename}"
try:
    df = pd.read_csv(filename)
    print(f"Loaded {{len(df)}} rows, {{len(df.columns)}} columns")
    print(f"Columns: {{list(df.columns)}}")
except Exception as e:
    print(f"Error loading CSV: {{e}}")
    exit(1)"""


def get_excel_loading_template(filename: str, sheet_name: str = None) -> str:
    """Get template for loading Excel file."""
    sheet_part = f", sheet_name='{sheet_name}'" if sheet_name else ""
    return f"""import pandas as pd

# Load Excel file
filename = "{filename}"
try:
    df = pd.read_excel(filename{sheet_part})
    print(f"Loaded {{len(df)}} rows, {{len(df.columns)}} columns")
    print(f"Columns: {{list(df.columns)}}")
except Exception as e:
    print(f"Error loading Excel: {{e}}")
    exit(1)"""


def get_json_access_code(patterns: List[str]) -> str:
    """
    Convert access patterns into copy-paste ready code snippets.

    Args:
        patterns: List of access patterns like ["data['key1']", "data['key2']['nested']"]

    Returns:
        Formatted code showing how to safely access each pattern
    """
    if not patterns:
        return ""

    lines = []
    lines.append("# === SAFE ACCESS PATTERNS (Copy these) ===")
    lines.append("# All patterns below are pre-validated from the JSON structure")
    lines.append("")

    for i, pattern in enumerate(patterns, 1):
        # Convert pattern to safe access code
        safe_pattern = pattern.replace("data", "data")

        # Build safe access with .get() for dict keys
        parts = pattern.split("']")
        if len(parts) > 1:
            # Has dict access - show safe version
            variable_name = f"value_{i}"
            lines.append(f"# Pattern {i}: {pattern}")
            lines.append(f"{variable_name} = {safe_pattern}")
            lines.append(f"print(f'Pattern {i}: {{{variable_name}}}')")
            lines.append("")

    return "\n".join(lines)


def get_complete_json_template(filename: str, patterns: List[str], query: str) -> str:
    """
    Get complete JSON processing template with loading + access patterns.

    Args:
        filename: Exact filename
        patterns: List of access patterns
        query: User's query for context

    Returns:
        Complete ready-to-run template
    """
    lines = []
    lines.append("#!/usr/bin/env python3")
    lines.append('"""')
    lines.append(f"Task: {query[:100]}")
    lines.append('"""')
    lines.append("")
    lines.append("import json")
    lines.append("")
    lines.append(f"# Step 1: Load JSON file (EXACT filename from list)")
    lines.append(f'filename = "{filename}"')
    lines.append("")
    lines.append("try:")
    lines.append("    with open(filename, 'r', encoding='utf-8') as f:")
    lines.append("        data = json.load(f)")
    lines.append("    ")
    lines.append("    # Verify data type")
    lines.append("    print(f'Loaded: {type(data).__name__}')")
    lines.append("    if isinstance(data, dict):")
    lines.append("        print(f'Keys: {list(data.keys())}')")
    lines.append("    elif isinstance(data, list):")
    lines.append("        print(f'Items: {len(data)}')")
    lines.append("    print()")
    lines.append("    ")
    lines.append("    # Step 2: Access data using pre-validated patterns")

    if patterns:
        lines.append("    # COPY these access patterns (they match your JSON structure):")
        lines.append("    ")
        for i, pattern in enumerate(patterns[:5], 1):  # Show first 5 as examples
            var_name = f"value{i}"
            lines.append(f"    # Access pattern {i}:")
            lines.append(f"    {var_name} = {pattern}")
            lines.append(f"    print(f'Pattern {i}: {{{var_name}}}')")
            lines.append("    ")

    lines.append("    # Step 3: Process data to answer the query")
    lines.append("    # TODO: Add your specific logic here")
    lines.append("    ")
    lines.append("except json.JSONDecodeError as e:")
    lines.append("    print(f'JSON parsing error: {e}')")
    lines.append("    exit(1)")
    lines.append("except FileNotFoundError:")
    lines.append(f"    print(f'File not found: {{filename}}')")
    lines.append("    exit(1)")
    lines.append("except KeyError as e:")
    lines.append("    print(f'Key not found in JSON: {e}')")
    lines.append("    exit(1)")

    return "\n".join(lines)


def get_multi_file_template(files: Dict[str, str]) -> str:
    """
    Get template for handling multiple files.

    Args:
        files: Dict mapping filenames to file types (json, csv, excel, etc.)

    Returns:
        Template showing how to load all files
    """
    lines = []
    lines.append("# === LOAD ALL FILES ===")
    lines.append("# All filenames are from the file list - do NOT change them")
    lines.append("")

    for filename, file_type in files.items():
        var_name = filename.replace('.', '_').replace('-', '_').replace(' ', '_')

        if file_type == 'json':
            lines.append(f"# Load {filename} (JSON)")
            lines.append(f"with open('{filename}', 'r', encoding='utf-8') as f:")
            lines.append(f"    {var_name} = json.load(f)")
            lines.append("")
        elif file_type == 'csv':
            lines.append(f"# Load {filename} (CSV)")
            lines.append(f"{var_name} = pd.read_csv('{filename}')")
            lines.append("")
        elif file_type in ['xlsx', 'xls', 'excel']:
            lines.append(f"# Load {filename} (Excel)")
            lines.append(f"{var_name} = pd.read_excel('{filename}')")
            lines.append("")

    return "\n".join(lines)


def format_critical_rules() -> str:
    """Get the top 3 critical rules in prominent format."""
    return """
╔════════════════════════════════════════════════════════════════════════════╗
║                     🚨 TOP 3 CRITICAL RULES 🚨                            ║
║                        (READ BEFORE CODING)                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

1️⃣  EXACT FILENAMES
    ✓ Copy the EXACT filename from the "AVAILABLE FILES" section above
    ✗ DON'T use generic names like 'file.json', 'data.csv', 'input.xlsx'
    📝 Example: Use "sales_2024_Q4.json" not "data.json"

2️⃣  NO INTERACTIVE INPUT
    ✓ Hardcode all filenames directly in your code
    ✗ DON'T use sys.argv[1], input(), or argparse
    📝 Example: filename = "sales_2024_Q4.json"  # Hardcoded

3️⃣  COPY ACCESS PATTERNS
    ✓ Copy the access patterns shown in the file section
    ✗ DON'T guess keys or make up field names
    📝 Example: revenue = data['company']['sales'][0]['revenue']  # From patterns
"""


def format_section_header(title: str, char: str = "=") -> str:
    """Create a prominent section header."""
    line = char * 80
    return f"\n{line}\n{title.center(80)}\n{line}\n"


__all__ = [
    'get_json_loading_template',
    'get_csv_loading_template',
    'get_excel_loading_template',
    'get_json_access_code',
    'get_complete_json_template',
    'get_multi_file_template',
    'format_critical_rules',
    'format_section_header',
]
