"""
LLM Response Parser Utility
===========================
Unified parsing utilities for LLM responses.

Extracts JSON, code blocks, and structured data from LLM responses
with markdown fences.

Extracted from duplicate implementations in:
- backend/agents/react/planning.py
- backend/agents/react/execution.py
- backend/agents/plan_execute.py
- backend/tools/python_coder/orchestrator.py
- backend/tools/python_coder/code_verifier.py
- backend/tools/python_coder/code_fixer.py

Version: 1.0.0
Created: 2025-01-20
"""

import json
import re
from typing import Optional, Any, Dict, List


class LLMResponseParser:
    """
    Unified LLM response parsing utilities.
    
    Handles extraction of JSON, code blocks, and structured data
    from LLM responses that may include markdown fences or extra text.
    """

    @staticmethod
    def extract_json(response_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from LLM response, handling markdown fences.

        Args:
            response_text: Raw LLM response text

        Returns:
            Parsed JSON dict, or None if parsing fails
        """
        try:
            cleaned = LLMResponseParser._extract_code_fence(response_text, "json")
            if cleaned:
                return json.loads(cleaned.strip())
            
            # Try direct JSON parsing
            return json.loads(response_text.strip())
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def extract_json_array(response_text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract JSON array from LLM response.

        Args:
            response_text: Raw LLM response text

        Returns:
            Parsed JSON array, or None if parsing fails
        """
        try:
            cleaned = LLMResponseParser._extract_code_fence(response_text, "json")
            if cleaned:
                parsed = json.loads(cleaned.strip())
                if isinstance(parsed, list):
                    return parsed
            
            # Try direct parsing
            parsed = json.loads(response_text.strip())
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None

    @staticmethod
    def extract_code(response_text: str, language: str = "python") -> str:
        """
        Extract code block from LLM response.

        Args:
            response_text: Raw LLM response text
            language: Expected language (default: "python")

        Returns:
            Extracted code string, or original text if no code block found
        """
        cleaned = LLMResponseParser._extract_code_fence(response_text, language)
        if cleaned:
            return cleaned.strip()
        
        # Fallback: try generic code fence
        cleaned = LLMResponseParser._extract_code_fence(response_text, None)
        if cleaned:
            return cleaned.strip()
        
        return response_text.strip()

    @staticmethod
    def _extract_code_fence(text: str, language: Optional[str] = None) -> Optional[str]:
        """
        Extract content from markdown code fence.

        Args:
            text: Text containing code fence
            language: Optional language hint (e.g., "json", "python")

        Returns:
            Extracted content, or None if not found
        """
        if language:
            # Try language-specific fence first
            pattern = f"```{language}"
            if pattern in text:
                after = text.split(pattern, 1)[1]
                return after.split("```", 1)[0].strip()
        
        # Try generic code fence
        if "```" in text:
            after = text.split("```", 1)[1]
            return after.split("```", 1)[0].strip()
        
        return None

    @staticmethod
    def extract_json_block(text: str) -> Optional[str]:
        """
        Extract JSON-looking array block using regex.

        Args:
            text: Text containing JSON

        Returns:
            Extracted JSON block, or None if not found
        """
        match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if match:
            return match.group(0)
        return None

    @staticmethod
    def parse_with_fallbacks(
        response_text: str,
        expected_type: type = dict
    ) -> Optional[Any]:
        """
        Parse response with multiple fallback strategies.

        Args:
            response_text: Raw LLM response
            expected_type: Expected type (dict or list)

        Returns:
            Parsed object, or None if all strategies fail
        """
        import ast
        
        candidates: List[str] = []
        stripped = response_text.strip()

        # Strategy 1: Extract code fence
        fenced = LLMResponseParser._extract_code_fence(stripped, "json")
        if fenced:
            candidates.append(fenced)

        # Strategy 2: Extract JSON block
        json_block = LLMResponseParser.extract_json_block(stripped)
        if json_block:
            candidates.append(json_block)

        # Strategy 3: Use original text
        candidates.append(stripped)

        # Try parsing each candidate
        for candidate in candidates:
            cleaned = candidate.strip()
            if not cleaned:
                continue

            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, expected_type):
                    return parsed
            except json.JSONDecodeError:
                try:
                    parsed_literal = ast.literal_eval(cleaned)
                    if isinstance(parsed_literal, expected_type):
                        return parsed_literal
                except (ValueError, SyntaxError):
                    continue

        return None

