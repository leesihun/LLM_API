"""
RAG Retriever Core
==================
Core retrieval logic for document indexing and semantic search.

Created: 2025-01-20
Version: 1.0.0
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

from backend.config.settings import settings
from backend.utils.logging_utils import get_logger
from backend.tools.rag_retriever.models import RAGDocument

logger = get_logger(__name__)


class RAGRetrieverCore:
    """Core document processing and retrieval system"""
    
    def __init__(self):
        """Initialize RAG retriever with embeddings and vector store"""
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_host
        )
        self.vector_db_path = Path(settings.vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.vector_stores: Dict[str, Any] = {}
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        logger.info("[RAGRetrieverCore] Initialized")
    
    def _load_json_as_document(self, file_path: Path) -> List[Document]:
        """Load JSON file and convert to Document objects"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert JSON to formatted text
        text_content = json.dumps(data, indent=2)
        
        return [Document(
            page_content=text_content,
            metadata={"source": str(file_path), "type": "json"}
        )]
    
    def _get_document_loader(self, file_path: Path):
        """Get appropriate document loader based on file extension"""
        extension = file_path.suffix.lower()
        
        if extension == ".json":
            return lambda: self._load_json_as_document(file_path)
        
        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls": UnstructuredExcelLoader,
        }
        
        loader_class = loaders.get(extension)
        if loader_class is None:
            raise ValueError(f"Unsupported file type: {extension}")
        
        return loader_class(str(file_path))
    
    def _get_doc_id(self, file_path: Path) -> str:
        """Generate unique document ID from file path"""
        return hashlib.md5(str(file_path).encode()).hexdigest()
    
    async def index_document(self, file_path: Path) -> str:
        """
        Process and index a document.
        
        Args:
            file_path: Path to document
            
        Returns:
            Document ID
        """
        # Load document
        loader = self._get_document_loader(file_path)
        
        if file_path.suffix.lower() == ".json":
            documents = loader()
        else:
            documents = loader.load()
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add metadata
        doc_id = self._get_doc_id(file_path)
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["source"] = str(file_path)
        
        # Create or update vector store
        if doc_id in self.vector_stores:
            self.vector_stores[doc_id].add_documents(chunks)
        else:
            if settings.vector_db_type == "faiss":
                vector_store = FAISS.from_documents(chunks, self.embeddings)
            elif settings.vector_db_type == "chroma":
                vector_store = Chroma.from_documents(
                    chunks,
                    self.embeddings,
                    persist_directory=str(self.vector_db_path / doc_id)
                )
            else:
                raise ValueError(f"Unsupported vector DB: {settings.vector_db_type}")
            
            self.vector_stores[doc_id] = vector_store
            
            if settings.vector_db_type == "faiss":
                vector_store.save_local(str(self.vector_db_path / doc_id))
        
        logger.info(f"[RAGRetrieverCore] Indexed document: {file_path.name} (id: {doc_id[:8]})")
        return doc_id
    
    async def retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[RAGDocument]:
        """
        Retrieve relevant chunks from indexed documents.
        
        Args:
            query: Search query
            document_ids: Optional list of document IDs to search
            top_k: Number of results to return
            
        Returns:
            List of RAGDocument objects
        """
        results = []
        
        # Determine which stores to search
        stores_to_search = {}
        
        if document_ids:
            for doc_id in document_ids:
                if doc_id in self.vector_stores:
                    stores_to_search[doc_id] = self.vector_stores[doc_id]
                else:
                    store = self._load_vector_store(doc_id)
                    if store:
                        self.vector_stores[doc_id] = store
                        stores_to_search[doc_id] = store
        else:
            # Search all available stores
            stores_to_search = self.vector_stores.copy()
            
            for doc_path in self.vector_db_path.iterdir():
                if doc_path.is_dir():
                    doc_id = doc_path.name
                    if doc_id not in stores_to_search:
                        store = self._load_vector_store(doc_id)
                        if store:
                            self.vector_stores[doc_id] = store
                            stores_to_search[doc_id] = store
        
        # Perform search on each store
        for doc_id, store in stores_to_search.items():
            docs_with_scores = store.similarity_search_with_score(query, k=top_k)
            
            for doc, score in docs_with_scores:
                results.append(RAGDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=float(score),
                    doc_id=doc.metadata.get("doc_id"),
                    source=doc.metadata.get("source")
                ))
        
        # Sort by score (lower is better)
        results.sort(key=lambda x: x.score)
        
        logger.info(f"[RAGRetrieverCore] Retrieved {len(results[:top_k])} documents for query: {query[:50]}")
        return results[:top_k]
    
    def _load_vector_store(self, doc_id: str):
        """Load vector store from disk"""
        store_path = self.vector_db_path / doc_id
        
        if not store_path.exists():
            return None
        
        try:
            if settings.vector_db_type == "faiss":
                return FAISS.load_local(str(store_path), self.embeddings, allow_dangerous_deserialization=True)
            elif settings.vector_db_type == "chroma":
                return Chroma(persist_directory=str(store_path), embedding_function=self.embeddings)
        except Exception as e:
            logger.error(f"Failed to load vector store {doc_id}: {e}")
            return None
