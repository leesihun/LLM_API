"""
Web Search Tool - Main Orchestrator
====================================
Main WebSearchTool class with LLM-powered query refinement and answer generation.
Integrates Tavily search with modular components for optimal results.

Version: 1.0.0
Created: 2025-01-13
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import httpx
import logging

from backend.config.settings import settings
from backend.models.tool_metadata import WebSearchMetadata, SearchResult
from backend.utils.logging_utils import get_logger

from .query_refiner import QueryRefiner
from .result_processor import ResultProcessor
from .answer_generator import AnswerGenerator

logger = get_logger(__name__)


class WebSearchTool:
    """
    Web search using Tavily API with LLM-enhanced query and answer processing.

    Provides a complete web search pipeline:
    1. Optional query refinement (LLM-based)
    2. Tavily API search (with fallback)
    3. Result processing and ranking
    4. Optional answer generation (LLM-based)

    Features:
    - LLM-powered query optimization
    - Temporal and location context awareness
    - Multi-source answer synthesis
    - Fallback search engine support
    - Comprehensive result metadata
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_query_refinement: bool = True,
        max_results: int = 5,
        search_depth: str = "basic",
        refinement_timeout: float = 3600.0,  # 1 hour timeout for query refinement
        answer_temperature: float = 0.5
    ):
        """Initialize the WebSearchTool with modular components."""
        self.tavily_api_key = api_key or settings.tavily_api_key
        self.tavily_url = "https://api.tavily.com/search"
        self.max_results = max_results
        self.search_depth = search_depth

        # Initialize modular components
        self.query_refiner = QueryRefiner(enable_refinement=enable_query_refinement, timeout=refinement_timeout)
        self.result_processor = ResultProcessor(max_results=max_results)
        # Set explicit timeout for answer generation (1 hour = 3600000 ms)
        self.answer_generator = AnswerGenerator(temperature=answer_temperature, timeout=3600000)

        logger.info(f"[WebSearchTool] Initialized with refinement={'enabled' if enable_query_refinement else 'disabled'}, max_results={max_results}, depth={search_depth}")

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_context: bool = True,
        user_location: Optional[str] = None,
        generate_answer: bool = True
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Perform web search with optional refinement and answer generation.

        Args:
            query: Search query
            max_results: Maximum results to return (override default)
            include_context: Whether to enhance query with temporal/location context
            user_location: Optional user location for location-aware searches
            generate_answer: Whether to generate LLM answer from results

        Returns:
            Tuple of (List of SearchResult objects, context dictionary)
        """
        # Get temporal context
        temporal_context = self._get_temporal_context()

        # Step 1: Query refinement (optional)
        original_query = query
        query_refined = False

        refined_query = self.query_refiner.refine(
            query,
            temporal_context=temporal_context,
            user_location=user_location
        )

        if refined_query != query:
            query = refined_query
            query_refined = True
            logger.info(f"[WebSearchTool] Query refined: {original_query} → {query}")

        # Step 2: Enhance query with context (optional)
        search_query = query
        if include_context:
            search_query = self._enhance_query_with_context(
                query,
                temporal_context,
                user_location
            )

        # Log search execution
        logger.subsection("Web Search Execution")
        search_info = {"Timestamp": temporal_context['current_datetime'], "Original Query": original_query}
        if query_refined:
            search_info["Refined Query"] = query
        if user_location:
            search_info["User Location"] = user_location
        if search_query != query:
            search_info["Enhanced Query"] = search_query
        logger.key_values(search_info)

        # Step 3: Execute search (Tavily with fallback)
        raw_results = await self._execute_search(search_query, max_results or self.max_results)

        # Step 4: Process and rank results
        processed_results = self.result_processor.process(
            raw_results,
            max_results=max_results or self.max_results
        )

        # Step 5: Generate answer (optional)
        answer = None
        sources_used = []

        if generate_answer and processed_results:
            answer, sources_used = await self.answer_generator.generate(
                query=original_query,
                results=processed_results,
                user_location=user_location,
                temporal_context=temporal_context
            )

        # Build context metadata
        return processed_results, {
            **temporal_context, "user_location": user_location, "query_enhanced": search_query != query,
            "original_query": original_query, "refined_query": query if query_refined else None,
            "query_refinement_applied": query_refined, "enhanced_query": search_query if search_query != query else None,
            "answer": answer, "sources_used": sources_used, "num_results": len(processed_results)
        }

    async def _execute_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Execute search with Tavily (and fallback if needed)."""
        # Try Tavily first
        try:
            results = await self._tavily_search(query, max_results)
            if results:
                return results
        except Exception as e:
            logger.warning(f"[WebSearchTool] Tavily search failed: {e}, trying fallback...")

        # Fallback to websearch_ts
        try:
            results = await self._websearch_ts_fallback(query, max_results)
            return results
        except Exception as e:
            logger.error(f"[WebSearchTool] Fallback search failed: {e}")
            return []

    async def _tavily_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Tavily API."""
        # Use corporate CA certificate if available (for corporate proxy SSL inspection)
        ssl_verify = "C:/DigitalCity.crt" if Path("C:/DigitalCity.crt").exists() else True

        # Increased timeout to 1 hour (3600s) to handle slow network or deep searches
        async with httpx.AsyncClient(timeout=3600.0, verify=ssl_verify) as client:
            response = await client.post(
                self.tavily_url,
                json={
                    "api_key": self.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": self.search_depth,
                    "include_answer": True,
                    "include_raw_content": False
                }
            )

            if response.status_code != 200:
                raise Exception(f"Tavily API error: {response.status_code}")

            data = response.json()
            results = []

            for item in data.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score")
                ))

            logger.info(f"[WebSearchTool] Tavily returned {len(results)} results")

            return results

    async def _websearch_ts_fallback(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback to websearch_ts module."""
        logger.info("[WebSearchTool] Using websearch_ts fallback")

        # Add websearch_ts to path
        websearch_path = Path(__file__).parent.parent.parent.parent / "websearch_ts"
        if str(websearch_path) not in sys.path:
            sys.path.insert(0, str(websearch_path))

        try:
            # Import websearch_ts dynamically
            import websearch_ts

            # Use websearch_ts search function
            search_results = websearch_ts.search(query, num_results=max_results)

            results = []
            for item in search_results:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("snippet", ""),
                    score=None
                ))

            logger.info(f"[WebSearchTool] Fallback returned {len(results)} results")

            return results

        except ImportError as e:
            logger.error(f"[WebSearchTool] Could not import websearch_ts: {e}")
            return []

    def _get_temporal_context(self) -> Dict[str, str]:
        """Get current temporal context for search enhancement."""
        now = datetime.now()
        return {"current_date": now.strftime("%Y-%m-%d"), "current_time": now.strftime("%H:%M:%S"),
                "current_datetime": now.strftime("%Y-%m-%d %H:%M:%S"), "day_of_week": now.strftime("%A"),
                "month": now.strftime("%B"), "year": str(now.year), "iso_datetime": now.isoformat()}

    def _enhance_query_with_context(
        self,
        query: str,
        context: Dict[str, str],
        user_location: Optional[str] = None
    ) -> str:
        """Enhance search query with temporal and location context when relevant."""
        query_lower = query.lower()
        enhanced_parts = [query]

        # Temporal and location keywords
        temporal_keywords = ["today", "now", "current", "latest", "recent", "this week", "this month", "this year", "updated", "new", "breaking"]
        location_keywords = ["near me", "nearby", "local", "in my area", "around here", "weather", "restaurants", "stores", "events"]

        has_temporal_intent = any(keyword in query_lower for keyword in temporal_keywords)
        has_location_intent = any(keyword in query_lower for keyword in location_keywords)

        # Enhance with temporal context if relevant
        if has_temporal_intent:
            # Avoid adding date if already present
            if not any(context["year"] in query for context in [context]):
                enhanced_parts.append(context['current_date'])
                logger.info(f"[WebSearchTool] Added temporal context: {context['current_date']}")

        # Enhance with location context if relevant and provided
        if has_location_intent and user_location:
            # Replace "near me" with actual location
            if "near me" in query_lower or "nearby" in query_lower:
                enhanced_query = query.replace("near me", f"in {user_location}")
                enhanced_query = enhanced_query.replace("nearby", f"in {user_location}")
                logger.info(f"[WebSearchTool] Replaced location placeholder with: {user_location}")
                return enhanced_query
            else:
                enhanced_parts.append(f"in {user_location}")
                logger.info(f"[WebSearchTool] Added location context: {user_location}")

        enhanced_query = " ".join(enhanced_parts)

        if enhanced_query != query:
            logger.info(f"[WebSearchTool] Enhanced query: {query} -> {enhanced_query}")

        return enhanced_query

    def format_results(self, results: List[SearchResult]) -> str:
        """Format search results as text (convenience method)."""
        return self.result_processor.format_as_list(results)


# Global search tool instance (for backward compatibility)
web_search_tool = WebSearchTool()
