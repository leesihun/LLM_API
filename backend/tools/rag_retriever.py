"""
RAG Retriever Tool
==================
Consolidated RAG tool for document indexing and retrieval.
Combines tool interface and core retrieval logic using FAISS/Chroma.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

from backend.config.settings import settings
from backend.core import BaseTool, ToolResult
from backend.utils.logging_utils import get_logger
from backend.models.tool_metadata import RAGDocument

logger = get_logger(__name__)


class RAGRetrieverTool(BaseTool):
    """
    RAG (Retrieval-Augmented Generation) tool.
    Handles document indexing and semantic search.
    """

    def __init__(self):
        super().__init__()
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_host
        )
        self.vector_db_path = Path(settings.vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.vector_stores: Dict[str, Any] = {}
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        logger.info("[RAGRetrieverTool] Initialized")

    def validate_inputs(self, **kwargs) -> bool:
        """Validate RAG inputs."""
        query = kwargs.get("query", "")
        return bool(query and query.strip())

    async def execute(
        self, query: str, context: Optional[str] = None,
        document_ids: Optional[List[str]] = None, top_k: int = 5,
        file_paths: Optional[List[str]] = None, **kwargs
    ) -> ToolResult:
        """Execute RAG retrieval."""
        try:
            if not query or not query.strip():
                return self._handle_validation_error("Query cannot be empty")

            if file_paths:
                logger.info(f"[RAG] Indexing {len(file_paths)} files")
                new_ids = []
                for fp in file_paths:
                    try:
                        doc_id = await self.index_document(Path(fp))
                        new_ids.append(doc_id)
                    except Exception as e:
                        logger.warning(f"Failed to index {fp}: {e}")
                if not document_ids and new_ids:
                    document_ids = new_ids

            results = await self.retrieve(query, document_ids, top_k)
            return self._format_tool_result(query, results)

        except Exception as e:
            return self._handle_error(e, "execute")

    async def index_document(self, file_path: Path) -> str:
        """Index a document."""
        loader = self._get_document_loader(file_path)
        documents = loader() if file_path.suffix.lower() == ".json" else loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["source"] = str(file_path)

        self._update_vector_store(doc_id, chunks)
        return doc_id

    async def retrieve(self, query: str, document_ids: Optional[List[str]] = None, top_k: int = 5) -> List[RAGDocument]:
        """Retrieve relevant chunks."""
        stores = self._get_stores_to_search(document_ids)
        results = []
        
        for _, store in stores.items():
            docs = store.similarity_search_with_score(query, k=top_k)
            for doc, score in docs:
                results.append(RAGDocument(
                    content=doc.page_content, metadata=doc.metadata,
                    score=float(score), doc_id=doc.metadata.get("doc_id"),
                    source=doc.metadata.get("source")
                ))
        
        results.sort(key=lambda x: x.score)
        return results[:top_k]

    def _get_document_loader(self, file_path: Path):
        ext = file_path.suffix.lower()
        if ext == ".json": return lambda: self._load_json(file_path)
        loaders = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader, ".xlsx": UnstructuredExcelLoader}
        if ext not in loaders: raise ValueError(f"Unsupported file type: {ext}")
        return loaders[ext](str(file_path))

    def _load_json(self, path: Path) -> List[Document]:
        with open(path, "r", encoding="utf-8") as f:
            return [Document(page_content=json.dumps(json.load(f), indent=2), metadata={"source": str(path)})]

    def _update_vector_store(self, doc_id: str, chunks: List[Document]):
        if doc_id in self.vector_stores:
            self.vector_stores[doc_id].add_documents(chunks)
        else:
            if settings.vector_db_type == "faiss":
                vs = FAISS.from_documents(chunks, self.embeddings)
                vs.save_local(str(self.vector_db_path / doc_id))
            else:
                vs = Chroma.from_documents(chunks, self.embeddings, persist_directory=str(self.vector_db_path / doc_id))
            self.vector_stores[doc_id] = vs

    def _get_stores_to_search(self, doc_ids: Optional[List[str]]) -> Dict[str, Any]:
        stores = {}
        target_ids = doc_ids if doc_ids else [p.name for p in self.vector_db_path.iterdir() if p.is_dir()]
        
        for did in target_ids:
            if did in self.vector_stores:
                stores[did] = self.vector_stores[did]
            else:
                store = self._load_store(did)
                if store: 
                    self.vector_stores[did] = store
                    stores[did] = store
        return stores

    def _load_store(self, doc_id: str):
        path = self.vector_db_path / doc_id
        if not path.exists(): return None
        try:
            if settings.vector_db_type == "faiss":
                return FAISS.load_local(str(path), self.embeddings, allow_dangerous_deserialization=True)
            return Chroma(persist_directory=str(path), embedding_function=self.embeddings)
        except Exception as e:
            logger.error(f"Failed to load store {doc_id}: {e}")
            return None

    def _format_tool_result(self, query: str, results: List[RAGDocument]) -> ToolResult:
        if not results:
            return ToolResult(success=True, output=f"No results for '{query}'", metadata={"query": query, "count": 0})
        
        output = f"Found {len(results)} documents for '{query}':\n\n"
        for i, doc in enumerate(results, 1):
            output += f"{i}. (Score: {doc.score:.2f})\n   Source: {doc.source}\n   {doc.content[:300]}...\n\n"
            
        return ToolResult(success=True, output=output, metadata={"query": query, "count": len(results)})

    def format_results(self, results: List[RAGDocument]) -> str:
        """Backward compatibility."""
        if not results: return "No relevant info."
        return "\n".join([f"--- Excerpt {i} ---\n{r.content}\nSource: {r.source}\n" for i, r in enumerate(results, 1)])

rag_retriever_tool = RAGRetrieverTool()

