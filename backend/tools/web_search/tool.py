"""
Web Search Tool - BaseTool Implementation
==========================================
Wraps WebSearchTool with standardized BaseTool interface.

This provides consistent interface for ReAct agent while maintaining
backward compatibility.

Created: 2025-01-20
Version: 2.0.0 (with BaseTool)
"""

from typing import List, Dict, Any, Optional

from backend.core import BaseTool, ToolResult
from backend.utils.logging_utils import get_logger
from backend.tools.web_search.searcher import WebSearchTool as WebSearchOrchestrator
from backend.models.tool_metadata import SearchResult

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """
    Web search tool with BaseTool interface.
    
    Features:
    - LLM-powered query refinement
    - Tavily API integration with fallback
    - Multi-source answer synthesis
    - Returns standardized ToolResult
    
    Usage:
        >>> tool = WebSearchTool()
        >>> result = await tool.execute(
        ...     query="latest AI developments in 2025",
        ...     max_results=5
        ... )
        >>> print(result.output)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_query_refinement: bool = True,
        max_results: int = 5,
        search_depth: str = "basic"
    ):
        """Initialize Web Search Tool"""
        super().__init__()
        
        # Initialize orchestrator
        self.orchestrator = WebSearchOrchestrator(
            api_key=api_key,
            enable_query_refinement=enable_query_refinement,
            max_results=max_results,
            search_depth=search_depth
        )
        
        self.default_max_results = max_results
        
        logger.info("[WebSearchTool] Initialized with BaseTool interface")
    
    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        max_results: Optional[int] = None,
        include_context: bool = True,
        user_location: Optional[str] = None,
        generate_answer: bool = True,
        **kwargs
    ) -> ToolResult:
        """
        Execute web search.
        
        Args:
            query: Search query
            context: Optional additional context (unused)
            max_results: Maximum results to return
            include_context: Whether to enhance query with temporal context
            user_location: Optional user location for location-aware searches
            generate_answer: Whether to generate LLM answer from results
            **kwargs: Additional parameters
            
        Returns:
            ToolResult with search results and answer
        """
        self._log_execution_start(
            query=query[:100],
            max_results=max_results or self.default_max_results
        )
        
        try:
            # Validate inputs
            if not self.validate_inputs(query=query):
                return self._handle_validation_error(
                    "Query cannot be empty",
                    parameter="query"
                )
            
            # Execute search via orchestrator
            results, metadata = await self.orchestrator.search(
                query=query,
                max_results=max_results or self.default_max_results,
                include_context=include_context,
                user_location=user_location,
                generate_answer=generate_answer
            )
            
            # Convert to ToolResult
            tool_result = self._convert_to_tool_result(results, metadata)
            
            self._log_execution_end(tool_result)
            return tool_result
            
        except Exception as e:
            return self._handle_error(e, "execute")
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate tool inputs.
        
        Args:
            **kwargs: Must contain 'query' key
            
        Returns:
            True if inputs are valid
        """
        query = kwargs.get("query", "")
        
        # Query must be non-empty string
        if not isinstance(query, str) or len(query.strip()) == 0:
            return False
        
        # Max results must be positive integer if provided
        max_results = kwargs.get("max_results")
        if max_results is not None and (not isinstance(max_results, int) or max_results <= 0):
            return False
        
        return True
    
    def _convert_to_tool_result(
        self,
        results: List[SearchResult],
        metadata: Dict[str, Any]
    ) -> ToolResult:
        """
        Convert search results to ToolResult.
        
        Args:
            results: List of SearchResult objects
            metadata: Search metadata dictionary
            
        Returns:
            Standardized ToolResult
        """
        # Build output string
        answer = metadata.get("answer")
        
        if answer:
            # If LLM generated answer, use that as main output
            output = f"{answer}\n\n"
        else:
            output = ""
        
        # Add source information
        if results:
            output += f"Found {len(results)} relevant results:\n\n"
            for i, result in enumerate(results[:5], 1):  # Top 5 results
                output += f"{i}. {result.title}\n"
                output += f"   {result.url}\n"
                if result.content:
                    # Truncate content for readability
                    content = result.content[:200] + "..." if len(result.content) > 200 else result.content
                    output += f"   {content}\n"
                output += "\n"
        else:
            output = "No results found for the query."
        
        # Build detailed metadata
        result_metadata = {
            "num_results": len(results),
            "original_query": metadata.get("original_query"),
            "refined_query": metadata.get("refined_query"),
            "query_refinement_applied": metadata.get("query_refinement_applied", False),
            "enhanced_query": metadata.get("enhanced_query"),
            "answer_generated": bool(answer),
            "sources_used": metadata.get("sources_used", []),
            "search_timestamp": metadata.get("current_datetime"),
            "user_location": metadata.get("user_location"),
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "content": r.content[:300] if r.content else "",
                    "score": r.score
                }
                for r in results
            ]
        }
        
        return ToolResult(
            success=True,
            output=output,
            metadata=result_metadata
        )


# Global singleton instance for backward compatibility
web_search_tool = WebSearchTool()


# Export for backward compatibility
__all__ = [
    'WebSearchTool',
    'web_search_tool',
]
