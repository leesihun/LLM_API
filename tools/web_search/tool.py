"""
Web Search Tool Implementation
Uses Tavily API for web search with LLM-enhanced query and summarization
"""
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

import config


class WebSearchTool:
    """
    Web search using Tavily with LLM query optimization and result summarization
    """

    def __init__(self):
        """Initialize web search tool"""
        self.api_key = config.TAVILY_API_KEY
        self.max_results = config.TAVILY_MAX_RESULTS
        self.search_depth = config.TAVILY_SEARCH_DEPTH
        self.include_domains = config.TAVILY_INCLUDE_DOMAINS
        self.exclude_domains = config.TAVILY_EXCLUDE_DOMAINS

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform web search using Tavily

        Args:
            query: Search query
            max_results: Override default max results
            search_depth: "basic" or "advanced"
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude

        Returns:
            Search results dictionary
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError(
                "Tavily client not installed. Install with: pip install tavily-python"
            )

        # Use provided values or defaults
        max_res = max_results or self.max_results
        depth = search_depth or self.search_depth
        inc_domains = include_domains or self.include_domains
        exc_domains = exclude_domains or self.exclude_domains

        # Initialize client
        client = TavilyClient(api_key=self.api_key)

        # Perform search
        start_time = time.time()

        search_params = {
            "query": query,
            "max_results": max_res,
            "search_depth": depth
        }

        if inc_domains:
            search_params["include_domains"] = inc_domains
        if exc_domains:
            search_params["exclude_domains"] = exc_domains

        # Log API call
        print(f"\n[TAVILY API] Searching with parameters:")
        print(f"  Query: {query}")
        print(f"  Max results: {max_res}")
        print(f"  Search depth: {depth}")

        results = client.search(**search_params)

        execution_time = time.time() - start_time

        # Log API response
        num_results = len(results.get("results", []))
        print(f"[TAVILY API] Search completed in {execution_time:.2f}s")
        print(f"[TAVILY API] Found {num_results} results")

        return {
            "success": True,
            "results": results.get("results", []),
            "query": query,
            "execution_time": execution_time,
            "num_results": len(results.get("results", []))
        }

    def format_results_for_llm(self, results: List[Dict]) -> str:
        """
        Format search results for LLM consumption

        Args:
            results: List of search result dictionaries

        Returns:
            Formatted string
        """
        formatted = []

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content")
            score = result.get("score", 0.0)

            formatted.append(f"""
Result {i} (relevance: {score:.2f}):
Title: {title}
URL: {url}
Content: {content}
""".strip())

        return "\n\n".join(formatted)
