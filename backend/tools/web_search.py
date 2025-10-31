"""
Web Search Tool
Uses Tavily API with websearch_ts fallback
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import logging

from backend.config.settings import settings
from backend.models.schemas import SearchResult

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Web search using Tavily API with fallback"""

    def __init__(self):
        self.tavily_api_key = settings.tavily_api_key
        self.tavily_url = "https://api.tavily.com/search"
        self.llm = None  # Lazy-loaded LLM for answer generation

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
        # Use corporate CA certificate if available (for corporate proxy SSL inspection)
        ssl_verify = "C:/DigitalCity.crt" if Path("C:/DigitalCity.crt").exists() else True
        async with httpx.AsyncClient(timeout=30.0, verify=ssl_verify) as client:
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

    def _get_llm(self) -> ChatOllama:
        """Lazy-load LLM instance for answer generation"""
        if self.llm is None:
            async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.ollama_timeout / 1000, connect=60.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            self.llm = ChatOllama(
                base_url=settings.ollama_host,
                model=settings.ollama_model,
                temperature=settings.ollama_temperature,
                num_ctx=settings.ollama_num_ctx,
                timeout=settings.ollama_timeout / 1000,
                async_client=async_client
            )
        return self.llm

    async def generate_answer(
        self,
        query: str,
        results: List[SearchResult]
    ) -> Tuple[str, List[str]]:
        """
        Generate LLM-based answer from search results

        Args:
            query: Original user query
            results: List of search results

        Returns:
            Tuple of (answer, sources_used)
            - answer: LLM-generated answer synthesizing the search results
            - sources_used: List of URLs used as sources
        """
        if not results:
            return "No search results found to answer your question.", []

        logger.info(f"[WebSearch] Generating answer for query: {query[:]}")

        # Get LLM
        llm = self._get_llm()

        # Build context from search results
        context_parts = []
        sources = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"Source {i}: {result.title}")
            context_parts.append(f"URL: {result.url}")
            context_parts.append(f"Content: {result.content}")
            context_parts.append("")  # Empty line for readability
            sources.append(result.url)

        search_context = "\n".join(context_parts)

        # Create prompt for answer generation
        system_prompt = """You are a helpful AI assistant that answers questions based on web search results.

Your task is to:
1. Read the provided search results carefully
2. Synthesize information from multiple sources
3. Generate a clear, accurate, and comprehensive answer
4. Cite sources by mentioning "Source 1", "Source 2", etc. when referencing specific information
5. If the search results don't contain enough information, say so clearly

Guidelines:
- Be concise but thorough
- Use natural language
- Prioritize accuracy over creativity
- Include source numbers in your answer (e.g., "According to Source 1...")
- If results are conflicting, mention both perspectives"""

        user_prompt = f"""Question: {query}

Search Results:
{search_context}

Based on these search results, please provide a comprehensive answer to the question. Remember to cite sources using "Source 1", "Source 2", etc."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await llm.ainvoke(messages)
            answer = response.content.strip()

            logger.info(f"[WebSearch] Answer generated successfully ({len(answer)} chars)")

            return answer, sources

        except Exception as e:
            logger.error(f"[WebSearch] Error generating answer: {e}")
            # Fallback to formatted results if LLM fails
            fallback_answer = f"I found {len(results)} search results for your query:\n\n"
            fallback_answer += self.format_results(results)
            return fallback_answer, sources


# Global search tool instance
web_search_tool = WebSearchTool()
