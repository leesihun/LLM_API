"""
Answer Generator Module
=======================
Generates natural language answers from search results using LLM.
Synthesizes information from multiple sources with source attribution.

Version: 1.0.0
Created: 2025-01-13
"""

from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging

from backend.utils.llm_factory import LLMFactory
from backend.config.prompts import PromptRegistry
from backend.models.tool_metadata import SearchResult
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AnswerGenerator:
    """
    Generates natural language answers from search results.

    Uses LLM to synthesize information from multiple search results
    into comprehensive, well-cited answers with temporal and location awareness.

    Features:
    - Multi-source synthesis
    - Source attribution (e.g., "According to Source 1...")
    - Temporal context awareness (current date/time)
    - Location context awareness (user location)
    - Fallback to formatted results on LLM failure
    """

    def __init__(
        self,
        temperature: float = 0.5,
        timeout: Optional[int] = None
    ):
        """
        Initialize the AnswerGenerator.

        Args:
            temperature: LLM temperature for answer generation (default: 0.5)
            timeout: LLM call timeout in milliseconds (optional)
        """
        self.temperature = temperature
        self.timeout = timeout
        self.llm = None  # Lazy-loaded

        logger.debug(
            f"[AnswerGenerator] Initialized with temperature={temperature}, "
            f"timeout={timeout}ms"
        )

    def _get_llm(self):
        """Lazy-load LLM instance for answer generation."""
        if self.llm is None:
            self.llm = LLMFactory.create_llm(
                temperature=self.temperature,
                timeout=self.timeout
            )
        return self.llm

    def generate(
        self,
        query: str,
        results: List[SearchResult],
        user_location: Optional[str] = None,
        temporal_context: Optional[Dict[str, str]] = None
    ) -> Tuple[str, List[str]]:
        """
        Generate LLM-based answer from search results.

        Returns: Tuple of (answer, sources_used)
        """
        if not results:
            logger.warning("[AnswerGenerator] No results provided for answer generation")
            return "No search results found to answer your question.", []

        logger.info(f"[AnswerGenerator] Generating answer for query: {query[:100]}")

        # Get temporal context if not provided
        if temporal_context is None:
            temporal_context = self._get_temporal_context()

        try:
            # Build context from search results
            search_context, sources = self._build_search_context(results)

            # Get prompts from PromptRegistry
            system_prompt = PromptRegistry.get(
                'search_answer_generation_system',
                current_date=temporal_context['current_date'],
                day_of_week=temporal_context['day_of_week'],
                current_time=temporal_context['current_time'],
                month=temporal_context['month'],
                year=temporal_context['year'],
                user_location=user_location
            )

            user_prompt = PromptRegistry.get(
                'search_answer_generation_user',
                query=query,
                search_context=search_context,
                user_location=user_location
            )

            # Generate answer using LLM
            answer = self._generate_with_llm(system_prompt, user_prompt)

            logger.info(f"[AnswerGenerator] Answer generated successfully ({len(answer)} chars)")

            return answer, sources

        except Exception as e:
            logger.error(f"[AnswerGenerator] Error generating answer: {e}")
            # Fallback to formatted results if LLM fails
            return self._generate_fallback_answer(results), self._extract_sources(results)

    def _get_temporal_context(self) -> Dict[str, str]:
        """Get current temporal context for answer generation."""
        now = datetime.now()
        return {"current_date": now.strftime("%Y-%m-%d"), "current_time": now.strftime("%H:%M:%S"),
                "current_datetime": now.strftime("%Y-%m-%d %H:%M:%S"), "day_of_week": now.strftime("%A"),
                "month": now.strftime("%B"), "year": str(now.year)}

    def _build_search_context(self, results: List[SearchResult]) -> Tuple[str, List[str]]:
        """Build formatted search context from results."""
        context_parts = []
        sources = []

        for i, result in enumerate(results, 1):
            context_parts.append(f"Source {i}: {result.title}")
            context_parts.append(f"URL: {result.url}")
            context_parts.append(f"Content: {result.content}")
            context_parts.append("")  # Empty line for readability
            sources.append(result.url)

        search_context = "\n".join(context_parts)

        logger.debug(
            f"[AnswerGenerator] Built search context from {len(results)} results "
            f"({len(search_context)} chars)"
        )

        return search_context, sources

    def _generate_with_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Generate answer using LLM."""
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        llm = self._get_llm()

        # Call LLM (with timeout if configured)
        config = {}
        if self.timeout:
            config["timeout"] = self.timeout / 1000  # Convert to seconds

        response = llm.invoke(messages, config=config) if config else llm.invoke(messages)

        answer = response.content.strip()

        return answer

    def _extract_sources(self, results: List[SearchResult]) -> List[str]:
        """Extract source URLs from results."""
        return [result.url for result in results]

    def _generate_fallback_answer(self, results: List[SearchResult]) -> str:
        """Generate fallback answer when LLM fails."""
        logger.info("[AnswerGenerator] Generating fallback answer")

        fallback = f"I found {len(results)} search results for your query:\n\n"

        for i, result in enumerate(results, 1):
            fallback += f"{i}. {result.title}\n"
            fallback += f"   URL: {result.url}\n"
            fallback += f"   {result.content}\n\n"

        return fallback

    def generate_batch(self, queries_and_results: List[Tuple[str, List[SearchResult]]], user_location: Optional[str] = None) -> List[Tuple[str, List[str]]]:
        """Generate answers for multiple query-result pairs in batch."""
        logger.info(f"[AnswerGenerator] Generating {len(queries_and_results)} answers in batch")

        answers = []
        for query, results in queries_and_results:
            answer, sources = self.generate(query, results, user_location)
            answers.append((answer, sources))

        return answers

    def validate_answer(self, answer: str, min_length: int = 50) -> bool:
        """Validate that generated answer meets quality criteria."""
        if not answer or not answer.strip():
            logger.warning("[AnswerGenerator] Answer is empty")
            return False

        if len(answer) < min_length:
            logger.warning(f"[AnswerGenerator] Answer too short ({len(answer)} < {min_length})")
            return False

        return True
