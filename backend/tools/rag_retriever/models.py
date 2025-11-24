"""
RAG Retriever Models
====================
Data models for RAG retrieval system.

Created: 2025-01-20
Version: 1.0.0
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class RAGDocument(BaseModel):
    """Represents a retrieved document chunk"""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float
    doc_id: Optional[str] = None
    source: Optional[str] = None


class RAGRetrievalRequest(BaseModel):
    """Request for RAG retrieval"""
    query: str
    document_ids: Optional[List[str]] = None
    top_k: int = 5


class RAGRetrievalResponse(BaseModel):
    """Response from RAG retrieval"""
    query: str
    documents: List[RAGDocument]
    total_results: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
