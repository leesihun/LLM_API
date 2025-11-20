"""
RAG Retriever Module
====================
Modular RAG (Retrieval-Augmented Generation) system.

Components:
- RAGRetrieverTool: Main tool with BaseTool interface
- RAGRetrieverCore: Core retrieval logic
- Data models: RAGDocument, RAGRetrievalRequest, RAGRetrievalResponse

Public API:
    >>> from backend.tools.rag_retriever import rag_retriever_tool
    >>> result = await rag_retriever_tool.execute(
    ...     query="What is the document about?",
    ...     top_k=5
    ... )

Created: 2025-01-20
Version: 1.0.0
"""

from backend.tools.rag_retriever.tool import RAGRetrieverTool, rag_retriever_tool
from backend.tools.rag_retriever.retriever import RAGRetrieverCore
from backend.tools.rag_retriever.models import (
    RAGDocument,
    RAGRetrievalRequest,
    RAGRetrievalResponse
)

__all__ = [
    'RAGRetrieverTool',
    'rag_retriever_tool',
    'RAGRetrieverCore',
    'RAGDocument',
    'RAGRetrievalRequest',
    'RAGRetrievalResponse',
]

__version__ = '1.0.0'
