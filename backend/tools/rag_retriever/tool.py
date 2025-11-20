"""
RAG Retriever Tool - BaseTool Implementation
=============================================
Wraps RAG retriever with standardized BaseTool interface.

This provides consistent interface for ReAct agent while maintaining
backward compatibility.

Created: 2025-01-20
Version: 2.0.0 (with BaseTool)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.core import BaseTool, ToolResult
from backend.utils.logging_utils import get_logger
from backend.tools.rag_retriever.retriever import RAGRetrieverCore
from backend.tools.rag_retriever.models import RAGDocument

logger = get_logger(__name__)


class RAGRetrieverTool(BaseTool):
    """
    RAG (Retrieval-Augmented Generation) tool with BaseTool interface.
    
    Features:
    - Document indexing and chunking
    - Semantic search using embeddings
    - FAISS/Chroma vector store support
    - Returns standardized ToolResult
    
    Supported formats: PDF, DOCX, TXT, XLSX, JSON
    
    Usage:
        >>> tool = RAGRetrieverTool()
        >>> # Index documents
        >>> doc_id = await tool.index_document(Path("document.pdf"))
        >>> # Retrieve relevant chunks
        >>> result = await tool.execute(
        ...     query="What is the main topic?",
        ...     document_ids=[doc_id]
        ... )
        >>> print(result.output)
    """
    
    def __init__(self):
        """Initialize RAG Retriever Tool"""
        super().__init__()
        
        # Initialize core retriever
        self.retriever = RAGRetrieverCore()
        
        logger.info("[RAGRetrieverTool] Initialized with BaseTool interface")
    
    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        file_paths: Optional[List[str]] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute RAG retrieval.
        
        Args:
            query: Search query
            context: Optional additional context (unused)
            document_ids: Optional list of document IDs to search
            top_k: Number of results to return
            file_paths: Optional list of files to index before retrieval
            **kwargs: Additional parameters
            
        Returns:
            ToolResult with retrieval results
        """
        self._log_execution_start(
            query=query[:100],
            top_k=top_k,
            doc_count=len(document_ids) if document_ids else "all"
        )
        
        try:
            # Validate inputs
            if not self.validate_inputs(query=query):
                return self._handle_validation_error(
                    "Query cannot be empty",
                    parameter="query"
                )
            
            # Index files if provided
            if file_paths:
                logger.info(f"[RAGRetrieverTool] Indexing {len(file_paths)} file(s) before retrieval")
                indexed_ids = []
                for file_path in file_paths:
                    try:
                        doc_id = await self.retriever.index_document(Path(file_path))
                        indexed_ids.append(doc_id)
                    except Exception as e:
                        logger.warning(f"Failed to index {file_path}: {e}")
                
                # Use indexed documents if no document_ids provided
                if not document_ids and indexed_ids:
                    document_ids = indexed_ids
            
            # Execute retrieval via core retriever
            documents = await self.retriever.retrieve(
                query=query,
                document_ids=document_ids,
                top_k=top_k
            )
            
            # Convert to ToolResult
            tool_result = self._convert_to_tool_result(query, documents)
            
            self._log_execution_end(tool_result)
            return tool_result
            
        except Exception as e:
            return self._handle_error(e, "execute")
    
    async def index_document(self, file_path: Path) -> str:
        """
        Index a document for later retrieval.
        
        Args:
            file_path: Path to document
            
        Returns:
            Document ID
        """
        return await self.retriever.index_document(file_path)
    
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
        
        # top_k must be positive integer if provided
        top_k = kwargs.get("top_k")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            return False
        
        return True
    
    def _convert_to_tool_result(
        self,
        query: str,
        documents: List[RAGDocument]
    ) -> ToolResult:
        """
        Convert retrieval documents to ToolResult.
        
        Args:
            query: The search query
            documents: List of retrieved documents
            
        Returns:
            Standardized ToolResult
        """
        if not documents:
            output = f"No relevant documents found for query: '{query}'"
        else:
            output = f"Found {len(documents)} relevant document(s) for query: '{query}'\n\n"
            
            for i, doc in enumerate(documents, 1):
                output += f"{i}. (Score: {doc.score:.3f})\n"
                output += f"   Source: {doc.source or 'Unknown'}\n"
                
                # Truncate content for readability
                content = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
                output += f"   {content}\n\n"
        
        # Build detailed metadata
        metadata = {
            "query": query,
            "num_results": len(documents),
            "documents": [
                {
                    "content": doc.content,
                    "score": doc.score,
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
        }
        
        return ToolResult(
            success=True,
            output=output,
            metadata=metadata
        )


# Global singleton instance for backward compatibility
rag_retriever_tool = RAGRetrieverTool()


# Export for backward compatibility
__all__ = [
    'RAGRetrieverTool',
    'rag_retriever_tool',
]
