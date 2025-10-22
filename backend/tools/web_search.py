"""
Web Search Tool
Uses Tavily API with websearch_ts fallback
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx

from backend.config.settings import settings
from backend.models.schemas import SearchResult


class WebSearchTool:
    """Web search using Tavily API with fallback"""

    def __init__(self):
        self.tavily_api_key = settings.tavily_api_key
        self.tavily_url = "https://api.tavily.com/search"

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Perform web search using Tavily or fallback
        """
        # Try Tavily first
        try:
            results = await self._tavily_search(query, max_results)
            if results:
                return results
        except Exception as e:
            print(f"Tavily search failed: {e}, trying fallback...")

        # Fallback to websearch_ts
        try:
            results = await self._websearch_ts_fallback(query, max_results)
            return results
        except Exception as e:
            print(f"Fallback search failed: {e}")
            return []

    async def _tavily_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Tavily API"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.tavily_url,
                json={
                    "api_key": self.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
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

            return results

    async def _websearch_ts_fallback(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback to websearch_ts module"""
        # Add websearch_ts to path
        websearch_path = Path(__file__).parent.parent.parent / "websearch_ts"
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

            return results

        except ImportError as e:
            print(f"Could not import websearch_ts: {e}")
            return []

    def format_results(self, results: List[SearchResult]) -> str:
        """Format search results as text"""
        if not results:
            return "No search results found."

        formatted = "Search Results:\n\n"

        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result.title}\n"
            formatted += f"   URL: {result.url}\n"
            formatted += f"   {result.content}\n\n"

        return formatted


# Global search tool instance
web_search_tool = WebSearchTool()
