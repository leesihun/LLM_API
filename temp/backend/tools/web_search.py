"""
Web Search Tool
===============
Consolidated web search tool with Tavily API integration.
Combines searching, refinement, result processing, and answer generation.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import httpx
import logging

from backend.config.settings import settings
from backend.core import BaseTool, ToolResult
from backend.models.tool_metadata import SearchResult
from backend.utils.logging_utils import get_logger
from backend.config.prompts import WEB_SEARCH_ANSWER_PROMPT
from backend.utils.llm_manager import LLMManager
from langchain_core.messages import HumanMessage

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """
    Web search using Tavily API with LLM-enhanced query and answer processing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = "basic"
    ):
        """Initialize the WebSearchTool."""
        super().__init__()
        self.tavily_api_key = api_key or settings.tavily_api_key
        self.tavily_url = "https://api.tavily.com/search"
        self.max_results = max_results
        self.search_depth = search_depth
        self.llm_manager = LLMManager()

    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute web search tool.

        Args:
            query: Search query
            context: Optional context (unused)
            **kwargs: Additional parameters (max_results, include_answer, etc.)

        Returns:
            ToolResult with search results and optional answer
        """
        self._log_execution_start(query=query)

        if not self.validate_inputs(query=query):
            return self._handle_validation_error("Query cannot be empty", parameter="query")

        try:
            max_results = kwargs.get('max_results', self.max_results)
            include_answer = kwargs.get('include_answer', True)
            user_location = kwargs.get('user_location')

            results, metadata = await self.search(
                query=query,
                max_results=max_results,
                user_location=user_location
            )

            if not results:
                return ToolResult.failure_result(
                    error="No search results found",
                    error_type="NoResultsError",
                    execution_time=self._elapsed_time()
                )

            output = {
                "results": [r.model_dump() if hasattr(r, 'model_dump') else r.__dict__ for r in results],
                "result_count": len(results),
                "formatted_results": self.format_results(results),
                "metadata": metadata
            }

            if include_answer:
                answer = await self.generate_answer(query, results)
                output["answer"] = answer

            result = ToolResult.success_result(
                output=output,
                execution_time=self._elapsed_time()
            )
            self._log_execution_end(result)
            return result

        except Exception as e:
            return self._handle_error(e, "execute")

    def validate_inputs(self, **kwargs) -> bool:
        """Validate search inputs."""
        query = kwargs.get("query", "")
        return bool(query and query.strip())

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_context: bool = True,
        user_location: Optional[str] = None
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Perform web search.
        """
        temporal_context = self._get_temporal_context()
        
        # Simple context enhancement
        search_query = query
        if include_context:
            search_query = f"{query} {temporal_context['month']} {temporal_context['year']}"
            if user_location:
                search_query += f" in {user_location}"

        logger.info(f"[WebSearchTool] Searching: {search_query}")

        try:
            results = await self._tavily_search(search_query, max_results or self.max_results)
        except Exception as e:
            logger.warning(f"[WebSearchTool] Tavily failed: {e}. Trying fallback.")
            results = [] 

        return results, {"temporal_context": temporal_context, "enhanced_query": search_query}

    async def generate_answer(self, query: str, results: List[SearchResult]) -> str:
        """Generate a summary answer from search results."""
        if not results:
            return "No results found to answer the query."
        
        context = self.format_results(results)
        prompt = WEB_SEARCH_ANSWER_PROMPT.format(query=query, context=context)
        
        try:
            response = await self.llm_manager.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "Failed to generate answer from results."

    async def _tavily_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Execute search with Tavily with SSL fallback."""
        # SSL verification options in order of preference:
        # 1. Corporate certificate (if exists)
        # 2. Default SSL verification
        # 3. Disabled SSL verification (fallback for problematic certs)
        ssl_options = []
        if Path("C:/DigitalCity.crt").exists():
            ssl_options.append("C:/DigitalCity.crt")
        ssl_options.append(True)   # Default SSL verification
        ssl_options.append(False)  # Fallback: disable SSL verification
        
        last_error = None
        for ssl_verify in ssl_options:
            try:
                async with httpx.AsyncClient(timeout=60.0, verify=ssl_verify) as client:
                    response = await client.post(
                        self.tavily_url,
                        json={
                            "api_key": self.tavily_api_key,
                            "query": query,
                            "max_results": max_results,
                            "search_depth": self.search_depth,
                            "include_answer": True
                        }
                    )
                    
                    if response.status_code != 200:
                        raise Exception(f"Tavily API error: {response.status_code}")

                    data = response.json()
                    if ssl_verify is False:
                        logger.warning("[WebSearchTool] SSL verification disabled for this request")
                    return [
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            content=item.get("content", ""),
                            score=item.get("score")
                        )
                        for item in data.get("results", [])
                    ]
            except Exception as e:
                error_msg = str(e)
                # Only retry with different SSL option if it's an SSL-related error
                if "SSL" in error_msg or "CERTIFICATE" in error_msg or "certificate" in error_msg.lower():
                    logger.warning(f"[WebSearchTool] SSL error with verify={ssl_verify}: {e}")
                    last_error = e
                    continue
                else:
                    # Non-SSL error, raise immediately
                    raise
        
        # All SSL options failed
        raise last_error if last_error else Exception("All SSL verification options failed")

    def _get_temporal_context(self) -> Dict[str, str]:
        now = datetime.now()
        return {
            "current_date": now.strftime("%Y-%m-%d"),
            "month": now.strftime("%B"),
            "year": str(now.year)
        }

    def format_results(self, results: List[SearchResult]) -> str:
        """Format search results as text."""
        if not results:
            return "No results found."
        
        parts = []
        for i, res in enumerate(results, 1):
            parts.append(f"[{i}] {res.title}\nURL: {res.url}\nSnippet: {res.content[:500]}...\n")
        return "\n".join(parts)

web_search_tool = WebSearchTool()
