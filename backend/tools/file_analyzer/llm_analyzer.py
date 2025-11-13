"""
LLM-Powered Deep Analyzer
===========================
Optional LLM-powered deep analysis using python_coder_tool.

Version: 1.0.0
Created: 2025-01-13
"""

import os
import uuid
from typing import Dict, Any

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMAnalyzer:
    """
    LLM-powered deep file analyzer.

    Uses python_coder_tool to generate and execute custom analysis code
    for complex file structures. More flexible but slower than direct analysis.
    """

    @staticmethod
    def deep_analyze(file_path: str, user_query: str = "") -> Dict[str, Any]:
        """
        Use LLM to generate and execute custom analysis code.

        Args:
            file_path: Path to file to analyze
            user_query: Specific user question to guide analysis

        Returns:
            Dict with LLM analysis results including:
            - format: 'LLM Deep Analysis'
            - success: Boolean indicating success
            - analysis_result: Analysis output
            - code_generated: Generated Python code
            - execution_time: Time taken for execution
            - session_id: Session identifier
        """
        try:
            from backend.tools.python_coder import python_coder_tool

            # Build analysis query for LLM
            analysis_prompt = f"""
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

            # Execute analysis via python_coder_tool
            session_id = f"deep_analysis_{uuid.uuid4().hex[:8]}"

            result = python_coder_tool.execute_code_task(
                query=analysis_prompt,
                file_paths=[file_path],
                session_id=session_id
            )

            return {
                "format": "LLM Deep Analysis",
                "success": result.get("success", False),
                "analysis_result": result.get("result", ""),
                "code_generated": result.get("code", ""),
                "execution_time": result.get("execution_time", 0),
                "session_id": session_id
            }

        except ImportError:
            return {
                "format": "LLM Deep Analysis",
                "error": "python_coder_tool not available"
            }
        except Exception as e:
            logger.error(f"LLM deep analysis failed: {e}", exc_info=True)
            return {
                "format": "LLM Deep Analysis",
                "error": str(e)
            }
