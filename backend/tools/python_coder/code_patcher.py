"""
Code Patcher Module for Incremental Code Building

This module provides intelligent code patching capabilities that preserve working
sections and only regenerate failed parts, leading to faster convergence and
better code quality.

Key Features:
- Error location analysis (identify which code section failed)
- Section-based code structure parsing
- Targeted regeneration (fix only broken parts)
- Working section preservation
- Traceback parsing for precise error location

Architecture:
- CodePatcher: Main class that orchestrates patching workflow
- Section types: imports, data_loading, processing, output
- LLM-based targeted regeneration with context preservation
"""

from typing import Dict, List, Optional, Any, Tuple
from langchain_core.messages import HumanMessage
import re

from backend.utils.logging_utils import get_logger
from .utils import extract_code_from_markdown

logger = get_logger(__name__)


class CodePatcher:
    """
    Incrementally builds code by preserving working sections and only
    regenerating failed parts.

    This class analyzes execution errors, identifies which code section failed,
    and generates targeted patches while preserving sections that executed successfully.
    """

    def __init__(self, llm, code_generator=None, code_verifier=None, max_verification_iterations: int = 3):
        """
        Initialize CodePatcher.

        Args:
            llm: LangChain LLM instance for code generation
            code_generator: Optional CodeGenerator instance (for code generation)
            code_verifier: Optional CodeVerifier instance (for verification)
            max_verification_iterations: Maximum verification attempts
        """
        self.llm = llm
        self.code_generator = code_generator
        self.code_verifier = code_verifier
        self.max_verification_iterations = max_verification_iterations

    def analyze_execution_error(
        self,
        code: str,
        error_message: str,
        execution_output: str
    ) -> Dict[str, Any]:
        """
        Identify which code section caused the error.

        This method parses the traceback to find the error line, analyzes
        the code structure, and determines which logical section failed.

        Args:
            code: Original Python code that was executed
            error_message: Error message from execution (including traceback)
            execution_output: Full execution output

        Returns:
            Dict with keys:
                - error_location: Section type that failed (str)
                - line_range: Tuple of (start_line, end_line) for failed section
                - error_type: Classified error type (str)
                - working_sections: List of sections that executed successfully
                - failed_section: The section dict that failed
        """
        # Parse traceback to find error line
        error_line = self._extract_error_line(error_message)

        # Analyze code structure
        sections = self._parse_code_sections(code)

        # Determine which section failed
        failed_section = None
        working_sections = []

        for section in sections:
            if section["start"] <= error_line <= section["end"]:
                failed_section = section
            elif section["end"] < error_line:
                # Sections that completed before error occurred
                working_sections.append(section)

        if not failed_section:
            # Error occurred outside identified sections - treat as processing
            failed_section = {
                "type": "processing",
                "start": error_line,
                "end": error_line + 10
            }

        logger.info(
            f"[CodePatcher] Error analysis: {failed_section['type']} section "
            f"(lines {failed_section['start']}-{failed_section['end']})"
        )
        logger.info(f"[CodePatcher] {len(working_sections)} working sections preserved")

        return {
            "error_location": failed_section["type"],
            "line_range": (failed_section["start"], failed_section["end"]),
            "error_type": self._classify_error(error_message),
            "working_sections": working_sections,
            "failed_section": failed_section
        }

    async def patch_code(
        self,
        original_code: str,
        error_analysis: Dict[str, Any],
        query: str,
        file_context: str,
        attempt_num: int
    ) -> str:
        """
        Generate patched code preserving working sections.

        This method uses the same generate-verify workflow as CodeGenerator:
        1. Generate fixed section code
        2. Verify it with CodeVerifier (if available)
        3. Iterate if needed (up to max_verification_iterations)

        Args:
            original_code: Original code that failed
            error_analysis: Error analysis from analyze_execution_error()
            query: User's original query/task
            file_context: File metadata and context
            attempt_num: Current attempt number

        Returns:
            Complete patched code with fixed section
        """
        failed_section_code = self._extract_section(
            original_code,
            error_analysis["line_range"]
        )

        working_sections = error_analysis["working_sections"]
        working_sections_code = [
            {
                "type": s["type"],
                "code": self._extract_section(original_code, (s["start"], s["end"]))
            }
            for s in working_sections
        ]

        # Build working sections summary
        working_summary = "\n".join([
            f"# {s['type'].upper()} section (PRESERVED):\n{s['code']}\n"
            for s in working_sections_code
        ])

        # Generate-verify loop
        fixed_section = None
        verification_iteration = 0

        while verification_iteration < self.max_verification_iterations:
            verification_iteration += 1

            # Generate fixed section
            fixed_section = await self._generate_fixed_section(
                query=query,
                file_context=file_context,
                working_summary=working_summary,
                failed_section_code=failed_section_code,
                error_analysis=error_analysis,
                attempt_num=attempt_num,
                verification_iteration=verification_iteration,
                previous_issues=[] if verification_iteration == 1 else issues
            )

            if not fixed_section or len(fixed_section) < 10:
                logger.warning("[CodePatcher] Generated code too short, falling back to original")
                return original_code

            # If no verifier available, use the generated code
            if not self.code_verifier:
                logger.info("[CodePatcher] No verifier available, skipping verification")
                break

            # Verify the fixed section
            is_verified, issues = await self.code_verifier.verify_code_answers_question(
                code=fixed_section,
                query=query,
                context=f"This is a patch for: {error_analysis['error_location']} section",
                file_context=file_context
            )

            if is_verified:
                logger.info(f"[CodePatcher] Patch verified successfully (iteration {verification_iteration})")
                break
            else:
                logger.warning(f"[CodePatcher] Verification failed (iteration {verification_iteration})")
                for issue in issues:
                    logger.warning(f"[CodePatcher]   - {issue}")

                if verification_iteration >= self.max_verification_iterations:
                    logger.warning("[CodePatcher] Max verification iterations reached, using last generated code")

        # Rebuild code with fixed section
        try:
            patched_code = self._rebuild_code(
                original_code,
                error_analysis["line_range"],
                fixed_section
            )
            return patched_code

        except Exception as e:
            logger.error(f"[CodePatcher] Failed to rebuild code: {e}")
            logger.warning("[CodePatcher] Returning original code as fallback")
            return original_code

    async def _generate_fixed_section(
        self,
        query: str,
        file_context: str,
        working_summary: str,
        failed_section_code: str,
        error_analysis: Dict[str, Any],
        attempt_num: int,
        verification_iteration: int,
        previous_issues: List[str]
    ) -> str:
        """
        Generate fixed code section using LLM.

        Args:
            query: User's original query
            file_context: File metadata and context
            working_summary: Summary of working code sections
            failed_section_code: The section that failed
            error_analysis: Error analysis results
            attempt_num: Current execution attempt number
            verification_iteration: Current verification iteration
            previous_issues: Issues from previous verification attempt

        Returns:
            Fixed code section
        """
        # Build issues context if this is a re-generation
        issues_context = ""
        if previous_issues:
            issues_context = "\n\nPREVIOUS VERIFICATION ISSUES:\n" + "\n".join([
                f"- {issue}" for issue in previous_issues
            ])

        prompt = f"""Fix the failed section of Python code while preserving working sections.

Task: {query}

File Context:
{file_context}

WORKING CODE SECTIONS (DO NOT MODIFY - these executed successfully):
{working_summary if working_summary else "No working sections identified"}

FAILED CODE SECTION (FIX THIS ONLY):
```python
{failed_section_code}
```

Error Type: {error_analysis["error_type"]}
Error Location: {error_analysis["error_location"]} section
Execution Attempt: {attempt_num}
Verification Iteration: {verification_iteration}{issues_context}

INSTRUCTIONS:
1. Fix ONLY the failed section above
2. The working sections will be automatically preserved
3. Maintain compatibility with working sections (variable names, data structures)
4. Focus on the specific error type: {error_analysis["error_type"]}
5. Keep the same structure/purpose as the original failed section

Respond ONLY with the fixed Python code (no explanations, no markdown):
"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            # Extract code from markdown blocks if present
            fixed_section = extract_code_from_markdown(response_text)

            logger.info(f"[CodePatcher] Generated fix ({len(fixed_section)} chars, iteration {verification_iteration})")

            return fixed_section

        except Exception as e:
            logger.error(f"[CodePatcher] Failed to generate fix: {e}")
            return ""

    def _extract_error_line(self, error_message: str) -> int:
        """
        Extract the line number where the error occurred from traceback.

        Args:
            error_message: Error message with traceback

        Returns:
            Line number where error occurred (1-indexed)
        """
        # Look for patterns like "line 42" in traceback
        match = re.search(r'line (\d+)', error_message, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Fallback: assume error is in the middle of the code
        return 50

    def _classify_error(self, error_message: str) -> str:
        """
        Classify the error type based on error message.

        Args:
            error_message: Error message from execution

        Returns:
            Error type classification (str)
        """
        error_lower = error_message.lower()

        if "importerror" in error_lower or "modulenotfounderror" in error_lower:
            return "import_error"
        elif "filenotfounderror" in error_lower or "no such file" in error_lower:
            return "file_not_found"
        elif "keyerror" in error_lower:
            return "key_error"
        elif "attributeerror" in error_lower:
            return "attribute_error"
        elif "typeerror" in error_lower:
            return "type_error"
        elif "valueerror" in error_lower:
            return "value_error"
        elif "indexerror" in error_lower:
            return "index_error"
        elif "zerodivisionerror" in error_lower:
            return "zero_division_error"
        elif "syntaxerror" in error_lower:
            return "syntax_error"
        elif "nameerror" in error_lower:
            return "name_error"
        else:
            return "runtime_error"

    def _parse_code_sections(self, code: str) -> List[Dict[str, Any]]:
        """
        Parse code into logical sections (imports, data_loading, processing, output).

        This method analyzes code structure and identifies logical boundaries
        between different types of operations.

        Args:
            code: Python code to parse

        Returns:
            List of section dicts with keys: type, start, end
        """
        lines = code.split('\n')
        sections = []

        current_section = None
        start_line = 1

        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()

            # Skip empty lines and comments
            if not line_stripped or line_stripped.startswith('#'):
                continue

            # Detect section changes based on code patterns
            if line_stripped.startswith('import ') or line_stripped.startswith('from '):
                if current_section != "imports":
                    if current_section:
                        sections.append({"type": current_section, "start": start_line, "end": i-1})
                    current_section = "imports"
                    start_line = i

            elif any(keyword in line_stripped for keyword in ['open(', 'read_csv', 'read_excel', 'read_json', 'load(']):
                if current_section != "data_loading":
                    if current_section:
                        sections.append({"type": current_section, "start": start_line, "end": i-1})
                    current_section = "data_loading"
                    start_line = i

            elif any(keyword in line_stripped for keyword in ['print(', 'savefig', 'to_csv', 'to_excel', 'save(']):
                if current_section != "output":
                    if current_section:
                        sections.append({"type": current_section, "start": start_line, "end": i-1})
                    current_section = "output"
                    start_line = i

            elif current_section is None:
                # Start processing section if no other section identified
                current_section = "processing"
                start_line = i

        # Add final section
        if current_section:
            sections.append({"type": current_section, "start": start_line, "end": len(lines)})

        logger.info(f"[CodePatcher] Parsed {len(sections)} code sections")
        for section in sections:
            logger.info(f"[CodePatcher]   - {section['type']}: lines {section['start']}-{section['end']}")

        return sections

    def _extract_section(self, code: str, line_range: Tuple[int, int]) -> str:
        """
        Extract a section of code by line range.

        Args:
            code: Full code
            line_range: Tuple of (start_line, end_line) - 1-indexed

        Returns:
            Extracted section code
        """
        lines = code.split('\n')
        start, end = line_range
        # Convert to 0-indexed and handle bounds
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)

        return '\n'.join(lines[start_idx:end_idx])

    def _rebuild_code(
        self,
        original_code: str,
        failed_line_range: Tuple[int, int],
        fixed_section: str
    ) -> str:
        """
        Rebuild code by replacing failed section with fixed version.

        Args:
            original_code: Original complete code
            failed_line_range: Line range of failed section
            fixed_section: Fixed code for that section

        Returns:
            Complete rebuilt code
        """
        lines = original_code.split('\n')
        start, end = failed_line_range

        # Convert to 0-indexed
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)

        # Rebuild: before + fixed + after
        before = lines[:start_idx]
        after = lines[end_idx:]

        rebuilt_lines = before + [fixed_section] + after

        rebuilt_code = '\n'.join(rebuilt_lines)

        logger.info(f"[CodePatcher] Rebuilt code: {len(before)} lines before, {fixed_section.count(chr(10))+1} fixed lines, {len(after)} lines after")

        return rebuilt_code
