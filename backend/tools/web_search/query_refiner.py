"""
Query Refiner Module
====================
Optimizes search queries using LLM for better search results.
Converts natural language questions into optimal search keywords.

Version: 1.0.0
Created: 2025-01-13
"""

from typing import Optional, Dict
import logging

from backend.utils.llm_factory import LLMFactory
from backend.config.prompts import PromptRegistry
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class QueryRefiner:
    """
    Optimizes search queries for better results using LLM.

    Converts natural language questions into optimal search keywords by:
    - Removing question words and filler words
    - Extracting key entities and concepts
    - Adding temporal context when relevant
    - Incorporating location context when applicable

    Features:
    - Optional query refinement (can be disabled)
    - Configurable timeout for LLM calls
    - Automatic fallback to original query on failure
    - Query length validation and truncation
    """

    def __init__(
        self,
        enable_refinement: bool = True,
        timeout: float = 10.0,
        max_query_length: int = 120
    ):
        """
        Initialize the QueryRefiner.

        Args:
            enable_refinement: Whether to enable query refinement (default: True)
            timeout: LLM call timeout in seconds (default: 10.0)
            max_query_length: Maximum length for refined queries (default: 120)
        """
        self.enable_refinement = enable_refinement
        self.timeout = timeout
        self.max_query_length = max_query_length
        self.llm = None  # Lazy-loaded

        logger.debug(
            f"[QueryRefiner] Initialized with refinement={'enabled' if enable_refinement else 'disabled'}, "
            f"timeout={timeout}s"
        )

    def _get_llm(self):
        """Lazy-load LLM instance for query refinement."""
        if self.llm is None:
            self.llm = LLMFactory.create_llm(
                temperature=0.3,  # Low temperature for consistent refinement
                timeout=int(self.timeout * 1000)  # Convert to milliseconds
            )
        return self.llm

    def refine(
        self,
        query: str,
        temporal_context: Optional[Dict[str, str]] = None,
        user_location: Optional[str] = None
    ) -> str:
        """
        Refine search query for better results.

        Converts natural language questions into optimal search keywords
        using LLM-based optimization with contextual awareness.

        Args:
            query: Original search query (natural language)
            temporal_context: Current date/time information (optional)
            user_location: User location for context (optional)

        Returns:
            Refined search keywords optimized for web search engines.
            Returns original query if refinement is disabled or fails.

        Examples:
            >>> refiner = QueryRefiner()
            >>> refiner.refine("what is the latest news about AI")
            "AI artificial intelligence latest news November 2025"

            >>> refiner.refine("best restaurants near me", user_location="Seoul")
            "best restaurants in Seoul 2025"
        """
        # Skip refinement if disabled
        if not self.enable_refinement:
            logger.debug("[QueryRefiner] Refinement disabled, using original query")
            return query

        logger.info(f"[QueryRefiner] Original query: {query}")

        # Get temporal context if not provided
        if temporal_context is None:
            from datetime import datetime
            now = datetime.now()
            temporal_context = {
                "current_date": now.strftime("%Y-%m-%d"),
                "day_of_week": now.strftime("%A"),
                "month": now.strftime("%B"),
                "year": str(now.year)
            }

        try:
            # Get prompt from PromptRegistry
            refinement_prompt = PromptRegistry.get(
                'search_query_refinement',
                query=query,
                current_date=temporal_context.get('current_date', ''),
                day_of_week=temporal_context.get('day_of_week', ''),
                month=temporal_context.get('month', ''),
                year=temporal_context.get('year', ''),
                user_location=user_location
            )

            # Get LLM
            llm = self._get_llm()

            # Call LLM with timeout
            from langchain_core.messages import HumanMessage
            response = llm.invoke(
                [HumanMessage(content=refinement_prompt)],
                config={"timeout": self.timeout}
            )

            refined_query = response.content.strip()

            # Validate response
            if not refined_query or len(refined_query) < 3:
                logger.warning("[QueryRefiner] Invalid refinement result, using original")
                return query

            # Truncate if too long
            if len(refined_query) > self.max_query_length:
                refined_query = refined_query[:self.max_query_length].rsplit(' ', 1)[0]
                logger.info(f"[QueryRefiner] Truncated query to {self.max_query_length} chars")

            logger.info(f"[QueryRefiner] Refined query: {refined_query}")

            return refined_query

        except Exception as e:
            logger.error(f"[QueryRefiner] Error during refinement: {e}")
            logger.info("[QueryRefiner] Falling back to original query")
            return query

    def refine_batch(
        self,
        queries: list[str],
        temporal_context: Optional[Dict[str, str]] = None,
        user_location: Optional[str] = None
    ) -> list[str]:
        """
        Refine multiple queries in batch.

        Args:
            queries: List of queries to refine
            temporal_context: Current date/time information (optional)
            user_location: User location for context (optional)

        Returns:
            List of refined queries (same length as input)
        """
        if not self.enable_refinement:
            return queries

        logger.info(f"[QueryRefiner] Refining {len(queries)} queries in batch")

        refined_queries = []
        for query in queries:
            refined = self.refine(query, temporal_context, user_location)
            refined_queries.append(refined)

        return refined_queries

    def set_enable_refinement(self, enabled: bool) -> None:
        """
        Enable or disable query refinement.

        Args:
            enabled: Whether to enable refinement
        """
        self.enable_refinement = enabled
        logger.info(f"[QueryRefiner] Refinement {'enabled' if enabled else 'disabled'}")
