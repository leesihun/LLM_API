"""
Tools Routes
Handles tool-specific endpoints for testing and direct tool access
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any

from backend.models.schemas import (
    ToolListResponse,
    ToolInfo,
    WebSearchRequest,
    WebSearchResponse,
    RAGSearchResponse
)
from backend.utils.auth import get_current_user
from backend.tools.rag_retriever import rag_retriever
from backend.tools.web_search import web_search_tool
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ============================================================================
# Router Setup
# ============================================================================

tools_router = APIRouter(prefix="/api/tools", tags=["Tools"])


# ============================================================================
# Tool Endpoints
# ============================================================================

@tools_router.get("/list", response_model=ToolListResponse)
async def list_tools(_: Dict[str, Any] = Depends(get_current_user)):
    """List all available tools and their descriptions"""
    tools = [
        ToolInfo(name="web_search", description="Search the web for current information"),
        ToolInfo(name="rag_retrieval", description="Retrieve from your uploaded documents"),
        ToolInfo(name="python_code", description="Run safe Python code snippets"),
        ToolInfo(name="python_coder", description="Generate and execute complex Python code"),
        ToolInfo(name="wikipedia", description="Search summaries from Wikipedia"),
        ToolInfo(name="weather", description="Get weather for a location"),
        ToolInfo(name="sql_query", description="Run parameterized SQL queries (configured)"),
    ]
    return ToolListResponse(tools=tools)


@tools_router.post("/websearch", response_model=WebSearchResponse)
async def tool_websearch(request: WebSearchRequest, _: Dict[str, Any] = Depends(get_current_user)):
    """
    Perform web search and generate LLM-based answer from results

    Returns:
        - results: Raw search results with title, URL, content
        - answer: LLM-generated answer synthesizing the search results
        - sources_used: List of URLs used as sources
    """
    logger.info(f"[Websearch Endpoint] Query: {request.query}")

    # Get search results with contextual enhancement
    results, context_metadata = await web_search_tool.search(
        request.query,
        max_results=request.max_results,
        include_context=request.include_context,
        user_location=request.user_location
    )

    # Generate LLM answer from results with context
    answer, sources_used = await web_search_tool.generate_answer(
        request.query,
        results,
        user_location=request.user_location
    )

    logger.info(f"[Websearch Endpoint] Found {len(results)} results, generated answer with {len(sources_used)} sources")
    if context_metadata.get('query_enhanced'):
        logger.info(f"[Websearch Endpoint] Query enhanced: {context_metadata.get('enhanced_query')}")

    return WebSearchResponse(
        results=results,
        answer=answer,
        sources_used=sources_used,
        context_used=context_metadata
    )


@tools_router.get("/rag/search", response_model=RAGSearchResponse)
async def tool_rag_search(query: str, top_k: int = 5, _: Dict[str, Any] = Depends(get_current_user)):
    """Search uploaded documents using RAG retrieval"""
    results = await rag_retriever.retrieve(query=query, top_k=top_k)
    return RAGSearchResponse(results=results)
