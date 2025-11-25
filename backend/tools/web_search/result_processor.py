"""
Result Processor Module
=======================
Processes and ranks web search results for optimal consumption.
Filters, formats, and prepares search results for LLM processing.

Version: 1.0.0
Created: 2025-01-13
"""

from typing import List, Dict, Any, Optional
import logging

from backend.models.tool_metadata import SearchResult
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResultProcessor:
    """
    Processes and ranks web search results.

    Handles:
    - Relevance filtering by score
    - Duplicate URL detection and removal
    - Result ranking and sorting
    - Formatting for LLM consumption
    - Key information extraction

    Features:
    - Configurable relevance threshold
    - Configurable maximum results
    - Clean formatting for LLM input
    - Preserves source attribution
    """

    def __init__(
        self,
        min_score: float = 0.0,
        max_results: int = 5,
        remove_duplicates: bool = True
    ):
        """
        Initialize the ResultProcessor.

        Args:
            min_score: Minimum relevance score to include (default: 0.0)
            max_results: Maximum number of results to return (default: 5)
            remove_duplicates: Whether to remove duplicate URLs (default: True)
        """
        self.min_score = min_score
        self.max_results = max_results
        self.remove_duplicates = remove_duplicates

        logger.debug(
            f"[ResultProcessor] Initialized with min_score={min_score}, "
            f"max_results={max_results}, remove_duplicates={remove_duplicates}"
        )

    def process(
        self,
        results: List[SearchResult],
        max_results: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Filter and rank search results.

        Applies relevance filtering, duplicate removal, and result limiting.

        Args:
            results: Raw search results from search engine
            max_results: Override default max_results (optional)

        Returns:
            Processed and ranked list of SearchResult objects

        Examples:
            >>> processor = ResultProcessor(min_score=0.5, max_results=3)
            >>> results = processor.process(raw_results)
            >>> len(results) <= 3
            True
        """
        if not results:
            logger.info("[ResultProcessor] No results to process")
            return []

        max_results = max_results or self.max_results

        logger.info(f"[ResultProcessor] Processing {len(results)} raw results")

        # Step 1: Filter by relevance score
        filtered = self._filter_by_score(results)

        # Step 2: Remove duplicates if enabled
        if self.remove_duplicates:
            filtered = self._remove_duplicates(filtered)

        # Step 3: Sort by relevance score (highest first)
        sorted_results = self._sort_by_score(filtered)

        # Step 4: Limit to max_results
        limited = sorted_results[:max_results]

        logger.info(
            f"[ResultProcessor] Processed {len(results)} → {len(limited)} results "
            f"(filtered: {len(filtered)}, deduplicated: {len(sorted_results)})"
        )

        return limited

    def _filter_by_score(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Filter results by minimum relevance score.

        Args:
            results: Search results to filter

        Returns:
            Filtered results meeting score threshold
        """
        if self.min_score <= 0.0:
            return results

        filtered = [
            result for result in results
            if result.score is not None and result.score >= self.min_score
        ]

        logger.debug(
            f"[ResultProcessor] Score filter: {len(results)} → {len(filtered)} "
            f"(min_score={self.min_score})"
        )

        return filtered

    def _remove_duplicates(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate URLs from results.

        Keeps the first occurrence (highest ranked) of each URL.

        Args:
            results: Search results to deduplicate

        Returns:
            Deduplicated results
        """
        seen_urls = set()
        unique_results = []

        for result in results:
            url_normalized = result.url.lower().strip()

            if url_normalized not in seen_urls:
                seen_urls.add(url_normalized)
                unique_results.append(result)

        if len(unique_results) < len(results):
            logger.debug(
                f"[ResultProcessor] Removed {len(results) - len(unique_results)} duplicates"
            )

        return unique_results

    def _sort_by_score(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Sort results by relevance score (highest first).

        Results without scores are placed at the end.

        Args:
            results: Search results to sort

        Returns:
            Sorted results
        """
        # Separate results with and without scores
        with_scores = [r for r in results if r.score is not None]
        without_scores = [r for r in results if r.score is None]

        # Sort results with scores
        sorted_with_scores = sorted(
            with_scores,
            key=lambda r: r.score,
            reverse=True
        )

        # Combine: scored results first, then unscored
        return sorted_with_scores + without_scores

    def format_for_llm(self, results: List[SearchResult]) -> str:
        """Format search results into readable text for LLM consumption."""
        if not results:
            return "No search results available."

        formatted_parts = []

        for i, result in enumerate(results, 1):
            formatted_parts.append(f"Source {i}: {result.title}")
            formatted_parts.append(f"URL: {result.url}")
            formatted_parts.append(f"Content: {result.content}")
            formatted_parts.append("")  # Empty line for readability

        formatted_text = "\n".join(formatted_parts).strip()

        logger.debug(
            f"[ResultProcessor] Formatted {len(results)} results "
            f"({len(formatted_text)} chars)"
        )

        return formatted_text

    def format_as_list(self, results: List[SearchResult]) -> str:
        """Format search results as a simple numbered list for users."""
        if not results:
            return "No search results found."

        formatted = "Search Results:\n\n"

        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result.title}\n"
            formatted += f"   URL: {result.url}\n"
            formatted += f"   {result.content}\n\n"

        return formatted

    def extract_sources(self, results: List[SearchResult]) -> List[str]:
        """Extract source URLs from results."""
        return [result.url for result in results]

    def extract_key_info(self, result: SearchResult) -> Dict[str, Any]:
        """Extract important fields from a single search result."""
        return {"title": result.title, "url": result.url, "content_preview": result.content if result.content else "",
                "score": result.score, "has_content": bool(result.content)}

    def get_summary(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Get statistical summary of results."""
        if not results:
            return {"total_results": 0, "with_scores": 0, "without_scores": 0, "avg_score": 0.0, "min_score": 0.0, "max_score": 0.0}
        with_scores = [r for r in results if r.score is not None]
        scores = [r.score for r in with_scores]
        return {"total_results": len(results), "with_scores": len(with_scores), "without_scores": len(results) - len(with_scores),
                "avg_score": sum(scores) / len(scores) if scores else 0.0, "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0}
