"""
Web Search Tool
Uses Tavily API with websearch_ts fallback
Enhanced with temporal and contextual data
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import logging

from backend.config.settings import settings
from backend.models.schemas import SearchResult
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class WebSearchTool:
    """Web search using Tavily API with fallback, enhanced with contextual data"""

    def __init__(self):
        self.tavily_api_key = settings.tavily_api_key
        self.tavily_url = "https://api.tavily.com/search"
        self.llm = None  # Lazy-loaded LLM for answer generation and query refinement

        # Query refinement settings (hardcoded)
        self.enable_query_refinement = True  # Set to False to disable
        self.refinement_timeout = 10.0  # LLM call timeout in seconds

    def _get_temporal_context(self) -> Dict[str, str]:
        """
        Get current temporal context for search enhancement

        Returns:
            Dictionary with current datetime information
        """
        now = datetime.now()
        return {
            "current_date": now.strftime("%Y-%m-%d"),
            "current_time": now.strftime("%H:%M:%S"),
            "current_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "month": now.strftime("%B"),
            "year": str(now.year),
            "iso_datetime": now.isoformat()
        }

    def _enhance_query_with_context(
        self,
        query: str,
        context: Dict[str, str],
        user_location: Optional[str] = None
    ) -> str:
        """
        Enhance search query with temporal and location context when relevant

        Args:
            query: Original search query
            context: Temporal context dictionary
            user_location: Optional user location (e.g., "Seoul, Korea", "New York, USA")

        Returns:
            Enhanced query string
        """
        query_lower = query.lower()
        enhanced_parts = [query]

        # Keywords that suggest temporal relevance
        temporal_keywords = [
            "today", "now", "current", "latest", "recent", "this week",
            "this month", "this year", "updated", "new", "breaking"
        ]

        # Keywords that suggest location relevance
        location_keywords = [
            "near me", "nearby", "local", "in my area", "around here",
            "weather", "restaurants", "stores", "events"
        ]

        # Check if query has temporal intent
        has_temporal_intent = any(keyword in query_lower for keyword in temporal_keywords)

        # Check if query has location intent
        has_location_intent = any(keyword in query_lower for keyword in location_keywords)

        # Enhance with temporal context if relevant
        if has_temporal_intent:
            # Avoid adding date if already present
            if not any(context["year"] in query for context in [context]):
                enhanced_parts.append(context['current_date'])
                logger.info(f"[WebSearch] Added temporal context: {context['current_date']}")

        # Enhance with location context if relevant and provided
        if has_location_intent and user_location:
            # Replace "near me" with actual location
            if "near me" in query_lower or "nearby" in query_lower:
                enhanced_query = query.replace("near me", f"in {user_location}")
                enhanced_query = enhanced_query.replace("nearby", f"in {user_location}")
                logger.info(f"[WebSearch] Replaced location placeholder with: {user_location}")
                return enhanced_query
            else:
                enhanced_parts.append(f"in {user_location}")
                logger.info(f"[WebSearch] Added location context: {user_location}")

        enhanced_query = " ".join(enhanced_parts)

        if enhanced_query != query:
            logger.info(f"[WebSearch] Enhanced query: {query} -> {enhanced_query}")

        return enhanced_query

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_context: bool = True,
        user_location: Optional[str] = None
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Perform web search using Tavily or fallback with contextual enhancement

        NEW: Includes LLM-based query refinement for better search results

        Args:
            query: Search query
            max_results: Maximum number of results to return
            include_context: Whether to enhance query with temporal/location context
            user_location: Optional user location for location-aware searches

        Returns:
            Tuple of (List of SearchResult objects, context dictionary)
        """
        # Get temporal context
        temporal_context = self._get_temporal_context()

        # === NEW: Query Refinement Step ===
        original_query = query
        query_refined = False

        if self.enable_query_refinement:
            try:
                refined_query = await self.refine_search_query(
                    query,
                    temporal_context,
                    user_location
                )

                if refined_query != query:
                    query = refined_query
                    query_refined = True
                    logger.info(f"[WebSearch] Query refined: {original_query} → {query}")

            except Exception as e:
                logger.warning(f"[WebSearch] Query refinement failed: {e}, using original")
        # ===================================

        # Enhance query if requested (existing logic)
        search_query = query
        if include_context:
            search_query = self._enhance_query_with_context(query, temporal_context, user_location)

        # Log search attempt with context
        logger.subsection("Web Search Execution")

        search_info = {
            "Timestamp": temporal_context['current_datetime'],
            "Original Query": original_query
        }

        if query_refined:
            search_info["Refined Query"] = query
        if user_location:
            search_info["User Location"] = user_location
        if search_query != query:
            search_info["Enhanced Query"] = search_query

        logger.key_values(search_info)

        # Build context metadata
        context_metadata = {
            **temporal_context,
            "user_location": user_location,
            "query_enhanced": search_query != query,
            "original_query": original_query,
            "refined_query": query if query_refined else None,  # NEW
            "query_refinement_applied": query_refined,  # NEW
            "enhanced_query": search_query if search_query != query else None
        }

        # Try Tavily first
        try:
            results = await self._tavily_search(search_query, max_results)
            if results:
                return results, context_metadata
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}, trying fallback...")

        # Fallback to websearch_ts
        try:
            results = await self._websearch_ts_fallback(search_query, max_results)
            return results, context_metadata
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return [], context_metadata

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

    async def refine_search_query(
        self,
        query: str,
        temporal_context: Dict[str, str],
        user_location: Optional[str] = None
    ) -> str:
        """
        Use LLM to refine and optimize search query for better results

        Args:
            query: Original query (from ReAct agent)
            temporal_context: Current date/time information
            user_location: Optional user location

        Returns:
            Refined search keywords optimized for web search
        """
        logger.info(f"[QueryRefinement] Original query: {query}")

        # Get LLM instance
        llm = self._get_llm()

        # Build context info
        context_info = f"""Current Context:
- Date: {temporal_context['current_date']} ({temporal_context['day_of_week']})
- Month/Year: {temporal_context['month']} {temporal_context['year']}"""

        if user_location:
            context_info += f"\n- Location: {user_location}"

        # Query refinement prompt with few-shot examples
        refinement_prompt = f"""You are a search query optimization expert. Convert natural language questions into optimal search keywords for web search engines.

{context_info}

RULES:
1. Remove question words (what, where, when, why, how, who)
2. Remove unnecessary words (the, a, an, is, are, about, for)
3. Use 3-10 specific, concrete keywords
4. Include important entities (names, places, products, dates)
5. Add current date/year if query asks for "latest", "recent", "current", "new"
6. Keep proper nouns and technical terms
7. Use keywords that a search engine would match against

EXAMPLES:

Input: "what is the latest news about artificial intelligence"
Output: AI artificial intelligence latest news November 2025

Input: "how does machine learning work"
Output: machine learning explanation tutorial how it works

Input: "where can I find information about Python programming"
Output: Python programming tutorial documentation guide

Input: "tell me about OpenAI GPT-4"
Output: OpenAI GPT-4 overview features capabilities

Input: "what's the weather like tomorrow"
Output: weather forecast tomorrow {temporal_context['current_date']}

Input: "best restaurants near me"
Output: best restaurants {"in " + user_location if user_location else "local area"} 2025

Input: "Python vs JavaScript which is better"
Output: Python vs JavaScript comparison pros cons 2025

Now optimize this query:

Input: {query}
Output:"""

        try:
            # Call LLM
            response = await llm.ainvoke(
                [HumanMessage(content=refinement_prompt)],
                config={"timeout": self.refinement_timeout}
            )

            refined_query = response.content.strip()

            # Validate response (too short or empty → use original)
            if not refined_query or len(refined_query) < 3:
                logger.warning(f"[QueryRefinement] Invalid refinement result, using original")
                return query

            # Truncate if too long (120 char limit)
            if len(refined_query) > 120:
                refined_query = refined_query[:120].rsplit(' ', 1)[0]

            logger.info(f"[QueryRefinement] Refined query: {refined_query}")

            return refined_query

        except Exception as e:
            logger.error(f"[QueryRefinement] Error during refinement: {e}")
            logger.info(f"[QueryRefinement] Falling back to original query")
            return query

    def _get_llm(self) -> ChatOllama:
        """Lazy-load LLM instance for answer generation and query refinement"""
        if self.llm is None:
            async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.ollama_timeout / 1000, connect=60.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            self.llm = ChatOllama(
                base_url=settings.ollama_host,
                model=settings.ollama_model,
                temperature=0.3,  # Lower temperature for more consistent query refinement
                num_ctx=settings.ollama_num_ctx,
                timeout=settings.ollama_timeout / 1000,
                async_client=async_client
            )
        return self.llm

    async def generate_answer(
        self,
        query: str,
        results: List[SearchResult],
        user_location: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Generate LLM-based answer from search results with temporal/location context

        Args:
            query: Original user query
            results: List of search results
            user_location: Optional user location context

        Returns:
            Tuple of (answer, sources_used)
            - answer: LLM-generated answer synthesizing the search results
            - sources_used: List of URLs used as sources
        """
        if not results:
            return "No search results found to answer your question.", []

        logger.info(f"[WebSearch] Generating answer for query: {query[:]}")

        # Get temporal context for answer generation
        temporal_context = self._get_temporal_context()

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

        # Build context section
        context_lines = [
            f"- Current Date: {temporal_context['current_date']} ({temporal_context['day_of_week']})",
            f"- Current Time: {temporal_context['current_time']}",
            f"- Month/Year: {temporal_context['month']} {temporal_context['year']}"
        ]
        if user_location:
            context_lines.append(f"- User Location: {user_location}")

        context_section = "\n".join(context_lines)

        # Create prompt for answer generation with temporal/location awareness
        system_prompt = f"""You are a helpful AI assistant that answers questions based on web search results.

CURRENT CONTEXT:
{context_section}

Your task is to:
1. Read the provided search results carefully
2. Synthesize information from multiple sources
3. Generate a clear, accurate, and comprehensive answer
4. Cite sources by mentioning "Source 1", "Source 2", etc. when referencing specific information
5. If the search results don't contain enough information, say so clearly
6. Be aware of temporal context - if the user asks about "today", "now", "current", etc., use the current date/time provided above
{"7. Be aware of location context - consider the user's location when providing location-specific information" if user_location else ""}

Guidelines:
- Be concise but thorough
- Use natural language
- Prioritize accuracy over creativity
- Include source numbers in your answer (e.g., "According to Source 1...")
- If results are conflicting, mention both perspectives
- When discussing time-sensitive information, acknowledge the current date/time context
{"- When discussing location-specific information, consider the user's location" if user_location else ""}"""

        user_prompt = f"""Question: {query}

Search Results:
{search_context}

Based on these search results and the context provided above (current date/time{", user location" if user_location else ""}), please provide a comprehensive answer to the question. Remember to cite sources using "Source 1", "Source 2", etc."""

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
