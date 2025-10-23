"""
Wikipedia Tool
Search and retrieve information from Wikipedia
"""

import logging
import httpx
from typing import List, Dict, Any, Optional


logger = logging.getLogger(__name__)


class WikipediaTool:
    """
    Wikipedia search and content retrieval

    Features:
    - Search Wikipedia articles
    - Get article summaries
    - Get full article content
    - Multi-language support
    """

    def __init__(self, language: str = "en", timeout: int = 10):
        self.language = language
        self.timeout = timeout
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"

    async def search(self, query: str, max_results: int = 5) -> List[str]:
        """
        Search Wikipedia for articles

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of article titles
        """
        logger.info(f"[Wikipedia] Searching for: {query}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "action": "opensearch",
                    "search": query,
                    "limit": max_results,
                    "namespace": 0,
                    "format": "json"
                }

                response = await client.get(self.base_url, params=params)
                response.raise_for_status()

                data = response.json()

                # OpenSearch returns [query, [titles], [descriptions], [urls]]
                if len(data) >= 2:
                    titles = data[1]
                    logger.info(f"[Wikipedia] Found {len(titles)} results")
                    return titles

                return []

        except Exception as e:
            logger.error(f"[Wikipedia] Search error: {e}")
            return []

    async def get_summary(self, title: str, sentences: int = 3) -> Optional[str]:
        """
        Get a summary of a Wikipedia article

        Args:
            title: Article title
            sentences: Number of sentences in summary

        Returns:
            Article summary or None
        """
        logger.info(f"[Wikipedia] Getting summary for: {title}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "extracts",
                    "exsentences": sentences,
                    "explaintext": True,
                    "exsectionformat": "plain"
                }

                response = await client.get(self.base_url, params=params)
                response.raise_for_status()

                data = response.json()

                # Extract the page content
                pages = data.get("query", {}).get("pages", {})

                for page_id, page_data in pages.items():
                    if page_id != "-1":  # -1 means page not found
                        extract = page_data.get("extract", "")
                        if extract:
                            logger.info(f"[Wikipedia] Retrieved summary for: {title}")
                            return extract

                logger.warning(f"[Wikipedia] No summary found for: {title}")
                return None

        except Exception as e:
            logger.error(f"[Wikipedia] Error getting summary: {e}")
            return None

    async def get_full_content(self, title: str) -> Optional[str]:
        """
        Get full content of a Wikipedia article

        Args:
            title: Article title

        Returns:
            Full article content or None
        """
        logger.info(f"[Wikipedia] Getting full content for: {title}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "extracts",
                    "explaintext": True,
                    "exsectionformat": "plain"
                }

                response = await client.get(self.base_url, params=params)
                response.raise_for_status()

                data = response.json()

                pages = data.get("query", {}).get("pages", {})

                for page_id, page_data in pages.items():
                    if page_id != "-1":
                        extract = page_data.get("extract", "")
                        if extract:
                            logger.info(f"[Wikipedia] Retrieved full content for: {title}")
                            return extract

                logger.warning(f"[Wikipedia] No content found for: {title}")
                return None

        except Exception as e:
            logger.error(f"[Wikipedia] Error getting content: {e}")
            return None

    async def search_and_summarize(self, query: str, sentences: int = 3) -> str:
        """
        Search for a topic and return the summary of the first result

        Args:
            query: Search query
            sentences: Number of sentences in summary

        Returns:
            Formatted summary
        """
        logger.info(f"[Wikipedia] Search and summarize: {query}")

        # Search for articles
        results = await self.search(query, max_results=1)

        if not results:
            return f"No Wikipedia articles found for: {query}"

        # Get summary of first result
        title = results[0]
        summary = await self.get_summary(title, sentences=sentences)

        if summary:
            return f"**{title}** (Wikipedia)\n\n{summary}\n\nSource: https://{self.language}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        else:
            return f"Found article '{title}' but couldn't retrieve summary"

    def format_search_results(self, titles: List[str]) -> str:
        """
        Format search results for display

        Args:
            titles: List of article titles

        Returns:
            Formatted string
        """
        if not titles:
            return "No results found"

        result = "Wikipedia Search Results:\n"
        for i, title in enumerate(titles, 1):
            url = f"https://{self.language}.wikipedia.org/wiki/{title.replace(' ', '_')}"
            result += f"{i}. [{title}]({url})\n"

        return result


# Global instance
wikipedia_tool = WikipediaTool(language="en", timeout=10)
