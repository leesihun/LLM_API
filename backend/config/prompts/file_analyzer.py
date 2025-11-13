"""
Prompts for file analyzer LLM-powered deep analysis.
"""

import os
from typing import Optional


def get_deep_analysis_prompt(file_path: str, user_query: Optional[str] = None) -> str:
    """
    Generate prompt for deep file analysis using LLM.

    Args:
        file_path: Path to the file being analyzed
        user_query: Optional user question about the file

    Returns:
        Deep analysis prompt string
    """
    prompt = f"""
Analyze the structure of this file in extreme detail:

1. Find the maximum nesting depth
2. Map out ALL key paths (e.g., data[0].user.profile.name)
3. Count items at each level
4. Show example values at leaf nodes
5. Identify all nested dictionaries and lists
6. Show the complete hierarchy

File: {os.path.basename(file_path)}

{f'User question: {user_query}' if user_query else ''}

Output a comprehensive JSON structure report.
"""
    return prompt.strip()
