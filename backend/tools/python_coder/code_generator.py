"""Code generation module for generating Python code using LLM."""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage

from backend.config import prompts
from backend.utils.logging_utils import get_logger
from .utils import extract_code_from_markdown

logger = get_logger(__name__)


class CodeGenerator:
    """Generates Python code using LLM."""

    def __init__(self, llm):
        """
        Initialize code generator.

        Args:
            llm: LangChain LLM instance
        """
        self.llm = llm

    async def generate_code(
        self,
        query: str,
        context: Optional[str],
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any],
        file_context: str,
        is_prestep: bool = False
    ) -> str:
        """
        Generate Python code using LLM.

        Args:
            query: User's question
            context: Optional additional context
            validated_files: Dict of validated files
            file_metadata: Metadata for files
            file_context: Pre-built file context string
            is_prestep: Whether this is pre-step execution (uses specialized prompt)

        Returns:
            Generated code
        """
        # Check if any JSON files are present
        has_json_files = any(
            metadata.get('type') == 'json'
            for metadata in file_metadata.values()
        )

        # Use centralized prompt from prompts module
        prompt = prompts.get_python_code_generation_prompt(
            query=query,
            context=context,
            file_context=file_context,
            is_prestep=is_prestep,
            has_json_files=has_json_files
        )

        try:
            logger.info("[CodeGenerator] Generating code...")
            logger.info("=" * 80)
            if file_context:
                logger.info("[CodeGenerator] File Context:")
                for line in file_context.strip().split('\n'):
                    logger.info(f"  {line}")
            if context:
                logger.info("[CodeGenerator] Agent Context:")
                for line in context.strip().split('\n')[:20]:  # First 20 lines
                    logger.info(f"  {line}")
            logger.info("=" * 80)

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            # Extract code from response (remove markdown if present)
            code = extract_code_from_markdown(response.content)

            logger.info("[CodeGenerator] Generated code:")
            logger.info("=" * 80)
            for line in code.split('\n'):
                logger.info(f"  {line}")
            logger.info("=" * 80)

            return code

        except Exception as e:
            logger.error(f"[CodeGenerator] Failed to generate code: {e}")
            return ""

    async def modify_code(
        self,
        code: str,
        issues: List[str],
        query: str,
        context: Optional[str]
    ) -> Tuple[str, List[str]]:
        """
        Modify code to fix issues.

        Args:
            code: Current Python code
            issues: List of issues to fix
            query: Original user query
            context: Optional additional context

        Returns:
            Tuple of (modified_code, list of changes made)
        """
        # Use centralized prompt from prompts module
        prompt = prompts.get_python_code_modification_prompt(
            query=query,
            context=context,
            code=code,
            issues=issues
        )

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            # Extract code
            modified_code = extract_code_from_markdown(response.content)

            changes = [f"Fixed: {issue}" for issue in issues]
            logger.info(f"[CodeGenerator] Modified code ({len(changes)} changes)")

            return modified_code, changes

        except Exception as e:
            logger.error(f"[CodeGenerator] Failed to modify code: {e}")
            return code, []  # Return original code if modification fails
