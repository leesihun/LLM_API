"""Code verification module for checking if generated code answers the user's question."""

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage

from backend.config import prompts
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CodeVerifier:
    """Verifies that generated code answers the user's question."""

    def __init__(self, llm, executor):
        """
        Initialize code verifier.

        Args:
            llm: LangChain LLM instance
            executor: CodeExecutor instance for static validation
        """
        self.llm = llm
        self.executor = executor

    async def verify_code_answers_question(
        self,
        code: str,
        query: str,
        context: Optional[str] = None,
        file_context: str = "",
        file_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Verify code focuses on answering user's question.
        Simplified verification focused on the core goal.

        Args:
            code: Python code to verify
            query: Original user query
            context: Optional additional context
            file_context: Information about available files
            file_metadata: Optional file metadata to check for JSON files

        Returns:
            Tuple of (is_verified, list of issues)
        """
        issues = []

        # Static analysis checks (safety)
        is_valid, static_issues = self.executor.validate_imports(code)
        if not is_valid:
            issues.extend(static_issues)

        # LLM-based semantic check: Does it answer the question?
        semantic_issues = await self._llm_verify_answers_question(
            code, query, context, file_context, file_metadata
        )
        issues.extend(semantic_issues)

        is_verified = len(issues) == 0
        return is_verified, issues

    async def _llm_verify_answers_question(
        self,
        code: str,
        query: str,
        context: Optional[str] = None,
        file_context: str = "",
        file_metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Use LLM to verify if code answers the user's question.
        Simplified verification focused on core requirements.

        Args:
            code: Python code to verify
            query: Original user query
            context: Optional additional context
            file_context: Information about available files
            file_metadata: Optional file metadata to check for JSON files

        Returns:
            List of issues found
        """
        # Check if any JSON files are present
        has_json_files = False
        if file_metadata:
            has_json_files = any(
                metadata.get('type') == 'json'
                for metadata in file_metadata.values()
            )

        # Use centralized prompt from prompts module
        prompt = prompts.get_python_code_verification_prompt(
            query=query,
            context=context,
            file_context=file_context,
            code=code,
            has_json_files=has_json_files
        )

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            # Try to parse JSON response
            response_text = response.content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            result = json.loads(response_text.strip())
            return result.get("issues", [])

        except Exception as e:
            logger.warning(f"[CodeVerifier] LLM verification failed: {e}")
            return []  # Don't block on LLM verification failure
