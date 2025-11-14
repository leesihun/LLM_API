"""
Code Fixer Module - LLM-based Code Fixing

This module provides LLM-based code fixing for execution errors.
Automatic fixes have been moved to auto_fixer.py (separate pre-processor).

Extracted from python_coder_tool.py for better modularity.
"""

from typing import Optional, Tuple, List

from langchain_core.messages import HumanMessage

from backend.config import prompts
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CodeFixer:
    """
    Handles LLM-based code fixing for Python code generation.

    Features:
    - LLM-based execution error fixing

    Note: Automatic fixes (filenames, imports, etc.) are handled by AutoFixer
    """

    def __init__(self, llm):
        """
        Initialize the code fixer.

        Args:
            llm: LangChain LLM instance for LLM-based fixing
        """
        self.llm = llm
        logger.info("[CodeFixer] Initialized")

    async def fix_execution_error(
        self,
        code: str,
        query: str,
        error_message: str,
        context: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Fix code based on execution error.

        Args:
            code: Current Python code
            query: Original user query
            error_message: Error from execution
            context: Optional additional context from agent execution history

        Returns:
            Tuple of (fixed_code, list of changes made)
        """
        # Use centralized prompt from prompts module
        prompt = prompts.get_python_code_execution_fix_prompt(
            query=query,
            context=context,
            code=code,
            error_message=error_message
        )

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            # Extract code
            fixed_code = response.content.strip()
            if fixed_code.startswith("```python"):
                fixed_code = fixed_code.split("```python")[1].split("```")[0]
            elif fixed_code.startswith("```"):
                fixed_code = fixed_code.split("```")[1].split("```")[0]

            fixed_code = fixed_code.strip()

            changes = [f"Fixed execution error: {error_message[:100]}"]
            logger.info(f"[CodeFixer] Fixed code after execution error")

            return fixed_code, changes

        except Exception as e:
            logger.error(f"[CodeFixer] Failed to fix execution error: {e}")
            return code, []  # Return original code if fix fails
