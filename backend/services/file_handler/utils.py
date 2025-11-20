"""
File Handler Utilities
=======================
Common utilities for file handlers.

Version: 2.0.0 (Unified)
Created: 2025-01-20
"""

import csv
from typing import List, Optional, Tuple


def detect_csv_delimiter(file_path: str, sample_size: int = 10240) -> str:
    """
    Auto-detect CSV delimiter.

    Args:
        file_path: Path to CSV file
        sample_size: Number of bytes to sample (default 10KB)

    Returns:
        Detected delimiter character
    """
    delimiter = ','  # Default

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = f.read(sample_size)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
    except Exception:
        pass  # Use default comma

    return delimiter


def try_multiple_encodings(
    file_path: str,
    encodings: Optional[List[str]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Try reading file with multiple encodings.

    Args:
        file_path: Path to file
        encodings: List of encodings to try

    Returns:
        Tuple of (encoding_used, content) or (None, None) if all fail
    """
    if encodings is None:
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return encoding, content
        except Exception:
            continue

    return None, None


def truncate_dict_for_preview(
    data: dict,
    max_items: int = 3,
    max_depth: int = 4,
    current_depth: int = 0
) -> dict:
    """
    Truncate dictionary for preview display.

    Args:
        data: Dictionary to truncate
        max_items: Maximum items per level
        max_depth: Maximum depth to traverse
        current_depth: Current depth level

    Returns:
        Truncated dictionary
    """
    if current_depth >= max_depth:
        return {"...": "max depth reached"}

    if not isinstance(data, dict):
        return data

    truncated = {}
    for i, (key, value) in enumerate(data.items()):
        if i >= max_items:
            truncated['...'] = f"({len(data) - max_items} more keys)"
            break

        if isinstance(value, dict):
            truncated[key] = truncate_dict_for_preview(
                value, max_items, max_depth, current_depth + 1
            )
        elif isinstance(value, list):
            truncated[key] = truncate_list_for_preview(
                value, max_items, max_depth, current_depth + 1
            )
        else:
            truncated[key] = value

    return truncated


def truncate_list_for_preview(
    data: list,
    max_items: int = 3,
    max_depth: int = 4,
    current_depth: int = 0
) -> list:
    """
    Truncate list for preview display.

    Args:
        data: List to truncate
        max_items: Maximum items to show
        max_depth: Maximum depth to traverse
        current_depth: Current depth level

    Returns:
        Truncated list
    """
    if current_depth >= max_depth:
        return ["... (max depth reached)"]

    if not isinstance(data, list):
        return data

    truncated = []
    for i, item in enumerate(data):
        if i >= max_items:
            truncated.append(f"... ({len(data) - max_items} more items)")
            break

        if isinstance(item, dict):
            truncated.append(truncate_dict_for_preview(
                item, max_items, max_depth, current_depth + 1
            ))
        elif isinstance(item, list):
            truncated.append(truncate_list_for_preview(
                item, max_items, max_depth, current_depth + 1
            ))
        else:
            truncated.append(item)

    return truncated


def format_null_counts(null_counts: dict, max_items: int = 5) -> str:
    """
    Format null counts for display.

    Args:
        null_counts: Dictionary of column->null_count
        max_items: Maximum items to show

    Returns:
        Formatted string
    """
    if not null_counts:
        return "No null values"

    # Filter out zero counts
    non_zero = {k: v for k, v in null_counts.items() if v > 0}

    if not non_zero:
        return "No null values"

    items = []
    for i, (col, count) in enumerate(non_zero.items()):
        if i >= max_items:
            items.append(f"... ({len(non_zero) - max_items} more)")
            break
        items.append(f"{col}: {count}")

    return ', '.join(items)
