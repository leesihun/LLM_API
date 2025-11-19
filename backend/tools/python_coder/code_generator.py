"""Code generation module for generating Python code using LLM."""

import re
from typing import Any, Dict, List, Optional, Tuple, Set

from langchain_core.messages import HumanMessage

from backend.config import prompts
from backend.utils.logging_utils import get_logger
from .utils import extract_code_from_markdown
from .variable_storage import VariableStorage

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

    def inject_variable_loading(
        self,
        code: str,
        session_id: Optional[str],
        query: str
    ) -> Tuple[str, List[str]]:
        """
        Automatically inject variable loading code for referenced saved variables.

        This method:
        1. Detects which saved variables are referenced in the code or query
        2. Prepends loading code for those variables
        3. Returns modified code and list of loaded variables

        Args:
            code: Generated Python code
            session_id: Session ID (to find saved variables)
            query: User's query

        Returns:
            Tuple of (modified_code, list of loaded variable names)
        """
        if not session_id:
            return code, []

        try:
            # Get available saved variables
            variable_metadata = VariableStorage.get_metadata(session_id)

            if not variable_metadata:
                return code, []

            # Detect which variables are referenced
            referenced_vars = self._detect_referenced_variables(
                code, query, variable_metadata
            )

            if not referenced_vars:
                return code, []

            # Build loading code
            loading_code_lines = []
            loading_code_lines.append("# Auto-loading saved variables from previous execution")
            loading_code_lines.append("import json")

            needs_pandas = False
            needs_numpy = False
            loaded_vars = []

            for var_name in referenced_vars:
                metadata = variable_metadata[var_name]
                var_type = metadata.get("type", "")
                load_code = metadata.get("load_code", "")

                if var_type == "pandas.DataFrame" and not needs_pandas:
                    loading_code_lines.append("import pandas as pd")
                    needs_pandas = True
                elif var_type == "numpy.ndarray" and not needs_numpy:
                    loading_code_lines.append("import numpy as np")
                    needs_numpy = True

                if load_code:
                    loading_code_lines.append(load_code)
                    loaded_vars.append(var_name)
                    logger.info(f"[CodeGenerator] Auto-injecting load code for variable: {var_name}")

            loading_code_lines.append("")  # Blank line

            # Prepend to code
            modified_code = "\n".join(loading_code_lines) + "\n" + code

            return modified_code, loaded_vars

        except Exception as e:
            logger.error(f"[CodeGenerator] Failed to inject variable loading: {e}")
            return code, []

    def _detect_referenced_variables(
        self,
        code: str,
        query: str,
        variable_metadata: Dict[str, Any]
    ) -> Set[str]:
        """
        Detect which saved variables are referenced in code or query.

        Args:
            code: Generated code
            query: User query
            variable_metadata: Metadata of saved variables

        Returns:
            Set of referenced variable names
        """
        referenced = set()

        # Combine code and query for analysis
        text_to_analyze = f"{code}\n{query}".lower()

        for var_name in variable_metadata.keys():
            # Check if variable name appears as a word boundary
            # This prevents matching 'df' in 'pdf' or 'data' in 'metadata'
            pattern = r'\b' + re.escape(var_name.lower()) + r'\b'

            if re.search(pattern, text_to_analyze):
                # Additional check: make sure it's not being assigned in the code
                # (we want to load it, not overwrite it)
                assignment_pattern = r'\b' + re.escape(var_name) + r'\s*='
                if not re.search(assignment_pattern, code, re.IGNORECASE):
                    referenced.add(var_name)
                    logger.info(f"[CodeGenerator] Detected reference to saved variable: {var_name}")

        return referenced

    def build_variable_context(
        self,
        session_id: Optional[str]
    ) -> str:
        """
        Build context string describing available saved variables.

        This is injected into the LLM prompt so it knows what variables
        are available to use.

        Args:
            session_id: Session ID

        Returns:
            Formatted context string
        """
        if not session_id:
            return ""

        try:
            variable_metadata = VariableStorage.get_metadata(session_id)

            if not variable_metadata:
                return ""

            lines = []
            lines.append("\n=== Available Saved Variables ===")
            lines.append("These variables were saved from previous code executions.")
            lines.append("You can use them directly in your code - they will be auto-loaded.\n")

            for var_name, metadata in variable_metadata.items():
                var_type = metadata.get("type", "unknown")
                lines.append(f"- {var_name} ({var_type})")

                # Add type-specific details
                if var_type == "pandas.DataFrame":
                    shape = metadata.get("shape", [])
                    columns = metadata.get("columns", [])
                    lines.append(f"  Shape: {shape}")
                    if columns:
                        col_preview = ", ".join(columns[:5])
                        more = f" ... (+{len(columns)-5} more)" if len(columns) > 5 else ""
                        lines.append(f"  Columns: {col_preview}{more}")

                elif var_type == "numpy.ndarray":
                    shape = metadata.get("shape", [])
                    dtype = metadata.get("dtype", "")
                    lines.append(f"  Shape: {shape}, dtype: {dtype}")

                elif var_type == "dict":
                    keys = metadata.get("keys", [])
                    if keys:
                        key_preview = ", ".join(str(k) for k in keys[:5])
                        more = f" ... (+{len(keys)-5} more)" if len(keys) > 5 else ""
                        lines.append(f"  Keys: {key_preview}{more}")

                elif var_type == "list":
                    length = metadata.get("length", 0)
                    lines.append(f"  Length: {length}")

                lines.append("")

            lines.append("Note: Just use the variable names directly - loading code will be added automatically.\n")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"[CodeGenerator] Failed to build variable context: {e}")
            return ""
