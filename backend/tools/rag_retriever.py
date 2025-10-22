"""
RAG Retriever Tool
Processes documents and performs semantic search
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    JSONLoader,
)

# Text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector stores
from langchain_community.vectorstores import FAISS, Chroma

from backend.config.settings import settings
from backend.models.schemas import RAGResult


class RAGRetriever:
    """Document processing and retrieval system"""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
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

    def _get_document_loader(self, file_path: Path):
        """Get appropriate document loader based on file extension"""
        extension = file_path.suffix.lower()

        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".json": lambda path: JSONLoader(path, jq_schema="."),
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
        Process and index a document
        Returns document ID
        """
        # Load document
        loader = self._get_document_loader(file_path)
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
            # Update existing store
            self.vector_stores[doc_id].add_documents(chunks)
        else:
            # Create new store
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

            # Save FAISS index
            if settings.vector_db_type == "faiss":
                vector_store.save_local(str(self.vector_db_path / doc_id))

        return doc_id

    async def retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[RAGResult]:
        """
        Retrieve relevant chunks from indexed documents
        """
        results = []

        # Determine which stores to search
        stores_to_search = {}

        if document_ids:
            for doc_id in document_ids:
                if doc_id in self.vector_stores:
                    stores_to_search[doc_id] = self.vector_stores[doc_id]
                else:
                    # Try to load from disk
                    store = self._load_vector_store(doc_id)
                    if store:
                        self.vector_stores[doc_id] = store
                        stores_to_search[doc_id] = store
        else:
            # Search all available stores
            stores_to_search = self.vector_stores.copy()

            # Load any stores from disk
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
                results.append(RAGResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=float(score)
                ))

        # Sort by score (lower is better for most distance metrics)
        results.sort(key=lambda x: x.score)

        return results[:top_k]

    def _load_vector_store(self, doc_id: str):
        """Load vector store from disk"""
        store_path = self.vector_db_path / doc_id

        if not store_path.exists():
            return None

        try:
            if settings.vector_db_type == "faiss":
                return FAISS.load_local(
                    str(store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            elif settings.vector_db_type == "chroma":
                return Chroma(
                    persist_directory=str(store_path),
                    embedding_function=self.embeddings
                )
        except Exception as e:
            print(f"Error loading vector store {doc_id}: {e}")
            return None

    def format_results(self, results: List[RAGResult]) -> str:
        """Format RAG results as context text"""
        if not results:
            return "No relevant information found in documents."

        formatted = "Retrieved Information:\n\n"

        for i, result in enumerate(results, 1):
            formatted += f"--- Excerpt {i} ---\n"
            formatted += f"{result.content}\n"
            if "source" in result.metadata:
                formatted += f"Source: {result.metadata['source']}\n"
            formatted += "\n"

        return formatted


# Global RAG retriever instance
rag_retriever = RAGRetriever()
