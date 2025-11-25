# Tool Extraction & SDK Architecture Plan

**Document Version:** 1.0.0
**Created:** November 25, 2024
**Author:** AI Architecture Analysis
**Status:** Planning Phase

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Proposed Architecture](#proposed-architecture)
4. [Detailed Design Specifications](#detailed-design-specifications)
5. [Migration Strategy](#migration-strategy)
6. [Implementation Timeline](#implementation-timeline)
7. [Testing Strategy](#testing-strategy)
8. [Deployment & Distribution](#deployment--distribution)
9. [Risk Assessment & Mitigation](#risk-assessment--mitigation)
10. [Success Metrics](#success-metrics)
11. [Appendices](#appendices)

---

## Executive Summary

### Problem Statement

The current `LLM_API` project has well-designed, modular tools (Python coder, web search, RAG retriever, file analyzer) that are **tightly coupled** to:
- Specific Ollama LLM configuration
- LangChain/LangGraph workflow orchestration
- Project-specific authentication and storage systems
- FastAPI application lifecycle

This coupling prevents:
- **Reuse** in other projects
- **Independent testing** of tools
- **Flexible LLM provider switching** (e.g., Ollama → OpenAI)
- **Community contribution** and distribution

### Proposed Solution

Extract tools into a **standalone, reusable SDK** (`huni-tools-sdk`) with:
- **Dependency injection** for LLM providers, storage, and configuration
- **Abstract base classes** for extensibility
- **Zero coupling** to orchestration logic
- **PyPI distribution** for easy installation
- **Comprehensive documentation** and examples

### Expected Benefits

| Benefit | Impact | Timeline |
|---------|--------|----------|
| **Reusability** | Use tools in any Python project | Immediate after Phase 4 |
| **Testability** | 90% test coverage with mocking | Phase 2-3 |
| **Flexibility** | Swap LLM providers in 5 lines of code | Immediate after Phase 1 |
| **Distribution** | PyPI package with semantic versioning | Phase 5 |
| **Community** | Enable external contributions | Phase 6+ |
| **Maintainability** | Independent versioning of tools vs orchestration | Ongoing |

---

## Current State Analysis

### Project Structure (As-Is)

```
LLM_API/
├── backend/
│   ├── api/                          # FastAPI routes (tightly coupled)
│   ├── config/
│   │   └── settings.py              # Hardcoded Ollama config
│   ├── tasks/
│   │   ├── react/                   # ReAct agent (orchestration)
│   │   └── Plan_execute.py          # Plan-Execute agent
│   └── tools/                        # EXTRACTION TARGET
│       ├── python_coder/            # 🎯 Well-modularized (v2.0.0)
│       ├── web_search/              # 🎯 Well-modularized (v2.0.0)
│       ├── rag_retriever/           # 🎯 Well-modularized (v2.0.0)
│       ├── file_analyzer/           # 🎯 Well-modularized (v2.0.0)
│       └── notepad.py               # Session memory (orchestration-specific)
└── data/                             # Project-specific storage
```

### Coupling Analysis

#### 1. **Python Coder Tool** (`backend/tools/python_coder/`)

**Current Dependencies:**
```python
# tool.py - Lines 15-25 (approximate)
from backend.config.settings import settings  # ❌ Tight coupling
from backend.utils.llm_factory import create_llm  # ❌ Tight coupling
from langchain_ollama import ChatOllama  # ❌ LangChain dependency

class PythonCoderTool:
    def __init__(self):
        self.llm = create_llm(  # ❌ Fixed to Ollama
            model=settings.ollama_coder_model,
            temperature=settings.ollama_coder_model_temperature
        )
        self.execution_dir = Path(settings.python_code_execution_dir)  # ❌ Fixed path
```

**Coupling Issues:**
- ❌ Hardcoded to `settings.py` global configuration
- ❌ Requires `llm_factory` (orchestration utility)
- ❌ Assumes LangChain `ChatOllama` interface
- ❌ Fixed execution directory structure
- ❌ No way to inject custom LLM provider
- ❌ No way to inject custom storage backend

**Desired State:**
```python
# After refactoring
from huni_tools.base import BaseTool, BaseLLMProvider, BaseStorage

class PythonCoderTool(BaseTool):
    def __init__(
        self,
        llm: BaseLLMProvider,        # ✅ Injected dependency
        storage: BaseStorage,         # ✅ Injected dependency
        timeout: int = 3000,
        max_iterations: int = 5
    ):
        self.llm = llm
        self.storage = storage
        self.timeout = timeout
        self.max_iterations = max_iterations
```

#### 2. **Web Search Tool** (`backend/tools/web_search/`)

**Current Dependencies:**
```python
# tool.py - Lines 10-20 (approximate)
from backend.config.settings import settings  # ❌ Tight coupling
from tavily import TavilyClient  # ✅ Already abstracted!

class WebSearchTool:
    def __init__(self):
        self.client = TavilyClient(api_key=settings.tavily_api_key)  # ❌ Fixed API key
```

**Coupling Issues:**
- ❌ Hardcoded to `settings.tavily_api_key`
- ✅ Already has clean Tavily client abstraction
- ❌ No way to inject custom search provider (e.g., Google, Bing)

**Desired State:**
```python
# After refactoring
from huni_tools.base import BaseSearchProvider

class WebSearchTool(BaseTool):
    def __init__(self, search_provider: BaseSearchProvider):
        self.search_provider = search_provider
```

#### 3. **RAG Retriever Tool** (`backend/tools/rag_retriever/`)

**Current Dependencies:**
```python
# tool.py - Lines 15-30 (approximate)
from backend.config.settings import settings  # ❌ Tight coupling
from backend.utils.llm_factory import create_llm  # ❌ Tight coupling
from langchain_community.vectorstores import FAISS  # ❌ Fixed to FAISS
from langchain_ollama import OllamaEmbeddings  # ❌ Fixed to Ollama

class RAGRetrieverTool:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model  # ❌ Fixed embedding model
        )
        self.vector_store = FAISS.load_local(
            settings.vector_db_path  # ❌ Fixed storage path
        )
```

**Coupling Issues:**
- ❌ Hardcoded to FAISS vector database
- ❌ Hardcoded to Ollama embeddings
- ❌ Fixed storage path
- ❌ No way to inject custom vector store (e.g., Pinecone, Weaviate)
- ❌ No way to inject custom embedding provider (e.g., OpenAI embeddings)

**Desired State:**
```python
# After refactoring
from huni_tools.base import BaseVectorStore, BaseEmbeddings

class RAGRetrieverTool(BaseTool):
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
```

#### 4. **File Analyzer Tool** (`backend/tools/file_analyzer/`)

**Current Dependencies:**
```python
# tool.py - Lines 10-25 (approximate)
from backend.config.settings import settings  # ❌ Tight coupling
from backend.utils.llm_factory import create_llm  # ❌ Tight coupling

class FileAnalyzer:
    def __init__(self):
        self.llm = create_llm()  # ❌ Fixed LLM provider
        self.supported_formats = ['.pdf', '.docx', '.csv', ...]  # ✅ Already flexible
```

**Coupling Issues:**
- ❌ Hardcoded to `llm_factory`
- ✅ Format handlers are already well-abstracted
- ❌ LLM-powered analysis tied to Ollama

**Desired State:**
```python
# After refactoring
from huni_tools.base import BaseLLMProvider

class FileAnalyzer(BaseTool):
    def __init__(
        self,
        llm: BaseLLMProvider,
        supported_formats: List[str] = None
    ):
        self.llm = llm
        self.supported_formats = supported_formats or DEFAULT_FORMATS
```

### Dependency Graph (Current State)

```
┌─────────────────────────────────────────────┐
│          FastAPI Application                │
│         (backend/api/app.py)                │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│       Configuration Layer                   │
│     (backend/config/settings.py)            │
│  - Hardcoded Ollama config                  │
│  - Hardcoded API keys                       │
│  - Hardcoded storage paths                  │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│        Orchestration Layer                  │
│     (backend/tasks/react/)                  │
│  - ReAct agent                              │
│  - Plan-Execute agent                       │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│          Tool Layer  🎯                     │
│     (backend/tools/)                        │
│  - python_coder/                            │
│  - web_search/                              │
│  - rag_retriever/                           │
│  - file_analyzer/                           │
└─────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         External Services                   │
│  - Ollama (http://127.0.0.1:11434)         │
│  - Tavily API                               │
│  - FAISS vector store                       │
└─────────────────────────────────────────────┘
```

**Problem:** Arrows point downward only = tight coupling, no inversion of control.

---

## Proposed Architecture

### High-Level Overview

```
┌───────────────────────────────────────────────────────────┐
│                    LLM_API Project                        │
│                 (Orchestration Layer)                     │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │         FastAPI Application                      │    │
│  │      (backend/api/, backend/tasks/)             │    │
│  └─────────────────┬───────────────────────────────┘    │
│                    │                                      │
│                    │ Depends on ⬇                        │
│                    │                                      │
│  ┌─────────────────▼───────────────────────────────┐    │
│  │      huni-tools-sdk (installed via pip)        │    │
│  │                                                  │    │
│  │  ┌────────────────────────────────────────┐   │    │
│  │  │   Base Abstractions                    │   │    │
│  │  │  - BaseTool                           │   │    │
│  │  │  - BaseLLMProvider                    │   │    │
│  │  │  - BaseStorage                        │   │    │
│  │  │  - BaseVectorStore                    │   │    │
│  │  └────────────────────────────────────────┘   │    │
│  │                                                  │    │
│  │  ┌────────────────────────────────────────┐   │    │
│  │  │   Concrete Tools                       │   │    │
│  │  │  - PythonCoderTool                    │   │    │
│  │  │  - WebSearchTool                      │   │    │
│  │  │  - RAGRetrieverTool                   │   │    │
│  │  │  - FileAnalyzerTool                   │   │    │
│  │  └────────────────────────────────────────┘   │    │
│  │                                                  │    │
│  │  ┌────────────────────────────────────────┐   │    │
│  │  │   Provider Implementations             │   │    │
│  │  │  - OllamaProvider                     │   │    │
│  │  │  - OpenAIProvider                     │   │    │
│  │  │  - LocalStorage                       │   │    │
│  │  │  - FAISSVectorStore                   │   │    │
│  │  └────────────────────────────────────────┘   │    │
│  └──────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────┘
```

### Repository Structure (After Migration)

#### **Repository 1: `huni-tools-sdk`** (New, standalone)

```
huni-tools-sdk/
├── README.md                          # Comprehensive SDK documentation
├── CHANGELOG.md                       # Semantic versioning history
├── LICENSE                            # MIT or Apache 2.0
├── pyproject.toml                     # Poetry/setuptools config
├── setup.py                           # For backward compatibility
├── requirements.txt                   # Core dependencies
├── requirements-dev.txt               # Development dependencies
│
├── huni_tools/                        # Main package
│   ├── __init__.py                   # Public API exports
│   ├── version.py                    # Version string
│   │
│   ├── base/                          # Abstract base classes
│   │   ├── __init__.py
│   │   ├── tool.py                   # BaseTool abstract class
│   │   ├── llm.py                    # BaseLLMProvider
│   │   ├── storage.py                # BaseStorage
│   │   ├── vector_store.py           # BaseVectorStore
│   │   ├── embeddings.py             # BaseEmbeddings
│   │   ├── search.py                 # BaseSearchProvider
│   │   └── exceptions.py             # Custom exceptions
│   │
│   ├── providers/                     # Concrete provider implementations
│   │   ├── __init__.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── ollama.py            # OllamaProvider
│   │   │   ├── openai.py            # OpenAIProvider
│   │   │   ├── anthropic.py         # AnthropicProvider
│   │   │   └── azure.py             # AzureOpenAIProvider
│   │   │
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   ├── local.py             # LocalStorage
│   │   │   ├── s3.py                # S3Storage
│   │   │   └── gcs.py               # GoogleCloudStorage
│   │   │
│   │   ├── vector_stores/
│   │   │   ├── __init__.py
│   │   │   ├── faiss.py             # FAISSVectorStore
│   │   │   ├── pinecone.py          # PineconeVectorStore
│   │   │   ├── weaviate.py          # WeaviateVectorStore
│   │   │   └── qdrant.py            # QdrantVectorStore
│   │   │
│   │   ├── embeddings/
│   │   │   ├── __init__.py
│   │   │   ├── ollama.py            # OllamaEmbeddings
│   │   │   ├── openai.py            # OpenAIEmbeddings
│   │   │   └── sentence_transformers.py  # HuggingFaceEmbeddings
│   │   │
│   │   └── search/
│   │       ├── __init__.py
│   │       ├── tavily.py            # TavilySearchProvider
│   │       ├── google.py            # GoogleSearchProvider
│   │       └── bing.py              # BingSearchProvider
│   │
│   ├── tools/                         # Concrete tool implementations
│   │   ├── __init__.py
│   │   │
│   │   ├── python_coder/             # Python code generation tool
│   │   │   ├── __init__.py
│   │   │   ├── tool.py              # Main PythonCoderTool class
│   │   │   ├── generator.py         # Code generation logic
│   │   │   ├── executor.py          # Code execution engine
│   │   │   ├── verifier.py          # Code verification
│   │   │   ├── variable_storage.py  # Variable persistence
│   │   │   ├── models.py            # Data models
│   │   │   ├── prompts.py           # Prompt templates
│   │   │   └── config.py            # Default configuration
│   │   │
│   │   ├── web_search/               # Web search tool
│   │   │   ├── __init__.py
│   │   │   ├── tool.py              # Main WebSearchTool class
│   │   │   ├── result_processor.py  # Result processing
│   │   │   ├── models.py            # Data models
│   │   │   ├── prompts.py           # Prompt templates
│   │   │   └── config.py            # Default configuration
│   │   │
│   │   ├── rag_retriever/            # RAG retrieval tool
│   │   │   ├── __init__.py
│   │   │   ├── tool.py              # Main RAGRetrieverTool class
│   │   │   ├── retriever.py         # Retrieval logic
│   │   │   ├── indexer.py           # Document indexing
│   │   │   ├── chunker.py           # Text chunking strategies
│   │   │   ├── models.py            # Data models
│   │   │   ├── prompts.py           # Prompt templates
│   │   │   └── config.py            # Default configuration
│   │   │
│   │   └── file_analyzer/            # File analysis tool
│   │       ├── __init__.py
│   │       ├── tool.py              # Main FileAnalyzerTool class
│   │       ├── handlers/            # Format-specific handlers
│   │       │   ├── __init__.py
│   │       │   ├── pdf.py           # PDF handler
│   │       │   ├── docx.py          # Word document handler
│   │       │   ├── csv.py           # CSV handler
│   │       │   ├── excel.py         # Excel handler
│   │       │   ├── image.py         # Image handler
│   │       │   └── json.py          # JSON handler
│   │       ├── analyzers.py         # Format analysis logic
│   │       ├── llm_analyzer.py      # LLM-powered analysis
│   │       ├── models.py            # Data models
│   │       ├── prompts.py           # Prompt templates
│   │       └── config.py            # Default configuration
│   │
│   └── utils/                         # Shared utilities
│       ├── __init__.py
│       ├── logging.py                # Logging utilities
│       ├── validation.py             # Input validation
│       └── serialization.py          # Data serialization
│
├── examples/                          # Usage examples
│   ├── 01_quickstart.py              # Basic usage
│   ├── 02_python_coder_example.py    # Python coder tool
│   ├── 03_web_search_example.py      # Web search tool
│   ├── 04_rag_example.py             # RAG retrieval tool
│   ├── 05_file_analyzer_example.py   # File analyzer tool
│   ├── 06_custom_llm_provider.py     # Custom provider
│   └── 07_advanced_workflow.py       # Complex workflow
│
├── tests/                             # Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   │
│   ├── base/                          # Base class tests
│   │   ├── test_tool.py
│   │   ├── test_llm.py
│   │   └── test_storage.py
│   │
│   ├── providers/                     # Provider tests
│   │   ├── test_ollama.py
│   │   ├── test_openai.py
│   │   └── test_storage.py
│   │
│   ├── tools/                         # Tool tests
│   │   ├── test_python_coder.py
│   │   ├── test_web_search.py
│   │   ├── test_rag_retriever.py
│   │   └── test_file_analyzer.py
│   │
│   └── integration/                   # Integration tests
│       ├── test_end_to_end.py
│       └── test_provider_switching.py
│
├── docs/                              # Sphinx documentation
│   ├── conf.py
│   ├── index.rst
│   ├── quickstart.rst
│   ├── api/                          # API reference
│   │   ├── base.rst
│   │   ├── providers.rst
│   │   └── tools.rst
│   ├── guides/                       # User guides
│   │   ├── installation.rst
│   │   ├── custom_providers.rst
│   │   └── best_practices.rst
│   └── examples/                     # Example notebooks
│       └── *.ipynb
│
├── .github/                           # GitHub Actions CI/CD
│   ├── workflows/
│   │   ├── test.yml                 # Run tests on PR
│   │   ├── lint.yml                 # Code quality checks
│   │   ├── publish.yml              # PyPI publishing
│   │   └── docs.yml                 # Documentation build
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
│
├── .gitignore
├── .pre-commit-config.yaml           # Pre-commit hooks
├── mypy.ini                          # Type checking config
├── pytest.ini                        # Pytest configuration
└── tox.ini                           # Tox configuration
```

#### **Repository 2: `LLM_API`** (Your current project, updated)

```
LLM_API/
├── backend/
│   ├── api/                          # FastAPI routes
│   ├── tasks/                        # Orchestration agents
│   │   ├── react/                   # ReAct agent
│   │   ├── Plan_execute.py          # Plan-Execute agent
│   │   └── smart_agent_task.py      # Dual agent strategy
│   ├── config/
│   │   ├── settings.py              # Configuration (now imports huni_tools)
│   │   └── prompts.py               # Prompts registry
│   ├── storage/
│   │   └── conversation_store.py    # Conversation persistence
│   ├── utils/
│   │   ├── auth.py                  # JWT authentication
│   │   └── logging_utils.py         # Logging utilities
│   └── adapters/                    # NEW: Adapter layer for huni-tools
│       ├── __init__.py
│       ├── llm_adapter.py          # Wraps settings.py config → LLMProvider
│       ├── storage_adapter.py      # Wraps settings.py config → Storage
│       └── tool_factory.py         # Factory for creating configured tools
│
├── requirements.txt                  # Now includes: huni-tools-sdk>=1.0.0
└── ...
```

---

## Detailed Design Specifications

### 1. Base Abstractions

#### **1.1. BaseTool** (`huni_tools/base/tool.py`)

```python
"""
Base abstract class for all tools in the huni-tools-sdk.

All tools must inherit from this class and implement the execute() method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ToolMetadata(BaseModel):
    """Metadata for tool execution tracking."""
    tool_name: str
    version: str
    execution_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    status: str = "pending"  # pending, running, success, failed
    error: Optional[str] = None


class ToolResult(BaseModel):
    """Standardized tool execution result."""
    success: bool
    output: Any
    metadata: ToolMetadata
    raw_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return self.model_dump()


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    All tools must implement:
    - execute(): Main execution method
    - validate_input(): Input validation

    Optional overrides:
    - get_metadata(): Return tool metadata
    - cleanup(): Cleanup resources after execution
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize the tool.

        Args:
            name: Tool name (e.g., "python_coder")
            version: Tool version (semantic versioning)
        """
        self.name = name
        self.version = version

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given inputs.

        Args:
            **kwargs: Tool-specific input parameters

        Returns:
            ToolResult: Standardized result object

        Raises:
            ToolExecutionError: If execution fails
        """
        pass

    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters before execution.

        Args:
            **kwargs: Input parameters to validate

        Returns:
            bool: True if valid, raises exception otherwise

        Raises:
            ValidationError: If validation fails
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return tool metadata.

        Returns:
            Dict containing tool name, version, capabilities, etc.
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.__doc__,
        }

    def cleanup(self):
        """
        Cleanup resources after execution.

        Override this method if your tool needs cleanup
        (e.g., closing file handles, database connections).
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False
```

#### **1.2. BaseLLMProvider** (`huni_tools/base/llm.py`)

```python
"""
Abstract base class for LLM providers.

Supports multiple LLM backends: Ollama, OpenAI, Anthropic, Azure, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from enum import Enum


class LLMMessage(BaseModel):
    """Standardized message format for LLM interaction."""
    role: str  # "system", "user", "assistant"
    content: str


class LLMResponse(BaseModel):
    """Standardized LLM response."""
    content: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement:
    - generate(): Generate text from prompt
    - generate_chat(): Generate text from conversation history
    - count_tokens(): Estimate token count
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the LLM provider.

        Args:
            model: Model name/identifier
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from a single prompt.

        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse: Generated text and metadata
        """
        pass

    @abstractmethod
    def generate_chat(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from conversation history.

        Args:
            messages: List of conversation messages
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse: Generated text and metadata
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for given text.

        Args:
            text: Text to count tokens for

        Returns:
            int: Estimated token count
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Return provider information."""
        return {
            "provider": self.__class__.__name__,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
```

#### **1.3. BaseStorage** (`huni_tools/base/storage.py`)

```python
"""
Abstract base class for storage backends.

Supports local filesystem, S3, Google Cloud Storage, etc.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime


class StorageMetadata(BaseModel):
    """Metadata for stored objects."""
    key: str
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    content_type: Optional[str] = None
    custom_metadata: Optional[Dict[str, Any]] = None


class BaseStorage(ABC):
    """
    Abstract base class for storage backends.

    All storage backends must implement:
    - save(): Save data
    - load(): Load data
    - exists(): Check if key exists
    - delete(): Delete data
    - list(): List keys
    """

    @abstractmethod
    def save(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save data to storage.

        Args:
            key: Storage key/path
            data: Binary data to save
            content_type: MIME type (e.g., "text/plain")
            metadata: Custom metadata

        Returns:
            str: Full path/URL to saved object
        """
        pass

    @abstractmethod
    def load(self, key: str) -> bytes:
        """
        Load data from storage.

        Args:
            key: Storage key/path

        Returns:
            bytes: Loaded binary data

        Raises:
            FileNotFoundError: If key doesn't exist
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in storage.

        Args:
            key: Storage key/path

        Returns:
            bool: True if exists, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete data from storage.

        Args:
            key: Storage key/path

        Returns:
            bool: True if deleted, False if key didn't exist
        """
        pass

    @abstractmethod
    def list(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[StorageMetadata]:
        """
        List objects in storage.

        Args:
            prefix: Filter by key prefix
            limit: Maximum number of results

        Returns:
            List[StorageMetadata]: List of object metadata
        """
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> StorageMetadata:
        """
        Get metadata for stored object.

        Args:
            key: Storage key/path

        Returns:
            StorageMetadata: Object metadata

        Raises:
            FileNotFoundError: If key doesn't exist
        """
        pass
```

#### **1.4. BaseVectorStore** (`huni_tools/base/vector_store.py`)

```python
"""
Abstract base class for vector databases.

Supports FAISS, Pinecone, Weaviate, Qdrant, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
import numpy as np


class VectorDocument(BaseModel):
    """Document with vector embedding."""
    id: str
    text: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """Vector search result."""
    document: VectorDocument
    score: float
    rank: int


class BaseVectorStore(ABC):
    """
    Abstract base class for vector databases.

    All vector stores must implement:
    - add(): Add documents
    - search(): Similarity search
    - delete(): Delete documents
    - update(): Update documents
    """

    @abstractmethod
    def add(
        self,
        documents: List[VectorDocument]
    ) -> List[str]:
        """
        Add documents to vector store.

        Args:
            documents: List of documents with embeddings

        Returns:
            List[str]: List of document IDs
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter: Metadata filter conditions

        Returns:
            List[SearchResult]: Search results sorted by similarity
        """
        pass

    @abstractmethod
    def delete(self, document_ids: List[str]) -> int:
        """
        Delete documents from vector store.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            int: Number of documents deleted
        """
        pass

    @abstractmethod
    def update(
        self,
        document_id: str,
        document: VectorDocument
    ) -> bool:
        """
        Update document in vector store.

        Args:
            document_id: Document ID to update
            document: Updated document

        Returns:
            bool: True if updated, False if not found
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Count total documents in vector store.

        Returns:
            int: Total document count
        """
        pass
```

#### **1.5. BaseEmbeddings** (`huni_tools/base/embeddings.py`)

```python
"""
Abstract base class for embedding models.

Supports Ollama, OpenAI, Sentence Transformers, etc.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddings(ABC):
    """
    Abstract base class for embedding models.

    All embedding models must implement:
    - embed_text(): Generate embedding for single text
    - embed_batch(): Generate embeddings for multiple texts
    """

    def __init__(self, model: str, **kwargs):
        """
        Initialize embedding model.

        Args:
            model: Model name/identifier
            **kwargs: Model-specific parameters
        """
        self.model = model
        self.extra_params = kwargs

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text.

        Args:
            text: Input text

        Returns:
            List[float]: Embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List[List[float]]: List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            int: Embedding vector dimension
        """
        pass
```

#### **1.6. BaseSearchProvider** (`huni_tools/base/search.py`)

```python
"""
Abstract base class for web search providers.

Supports Tavily, Google, Bing, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime


class SearchResultItem(BaseModel):
    """Single search result."""
    title: str
    url: str
    snippet: str
    published_date: Optional[datetime] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Search response with results."""
    query: str
    results: List[SearchResultItem]
    total_results: int
    search_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


class BaseSearchProvider(ABC):
    """
    Abstract base class for search providers.

    All search providers must implement:
    - search(): Perform search query
    """

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 5,
        **kwargs
    ) -> SearchResponse:
        """
        Perform search query.

        Args:
            query: Search query
            max_results: Maximum number of results
            **kwargs: Provider-specific parameters

        Returns:
            SearchResponse: Search results
        """
        pass
```

#### **1.7. Custom Exceptions** (`huni_tools/base/exceptions.py`)

```python
"""
Custom exceptions for huni-tools-sdk.
"""


class HuniToolsError(Exception):
    """Base exception for all huni-tools errors."""
    pass


class ToolExecutionError(HuniToolsError):
    """Raised when tool execution fails."""
    pass


class ValidationError(HuniToolsError):
    """Raised when input validation fails."""
    pass


class LLMProviderError(HuniToolsError):
    """Raised when LLM provider fails."""
    pass


class StorageError(HuniToolsError):
    """Raised when storage operation fails."""
    pass


class VectorStoreError(HuniToolsError):
    """Raised when vector store operation fails."""
    pass


class SearchProviderError(HuniToolsError):
    """Raised when search provider fails."""
    pass
```

### 2. Provider Implementations

#### **2.1. OllamaProvider** (`huni_tools/providers/llm/ollama.py`)

```python
"""
Ollama LLM provider implementation.

Connects to local Ollama instance via HTTP API.
"""

from typing import List, Optional
import requests
from huni_tools.base.llm import BaseLLMProvider, LLMMessage, LLMResponse
from huni_tools.base.exceptions import LLMProviderError


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider.

    Requires Ollama running at specified host (default: http://127.0.0.1:11434)
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 3000,
        **kwargs
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name (e.g., "qwen3:8b", "llama2")
            host: Ollama API endpoint
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional Ollama parameters (top_p, top_k, etc.)
        """
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.host = host.rstrip('/')
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text from prompt using Ollama."""
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
                **self.extra_params,
                **kwargs
            }
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data.get("response", ""),
                model=self.model,
                tokens_used=data.get("eval_count"),
                metadata={
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                }
            )
        except requests.RequestException as e:
            raise LLMProviderError(f"Ollama API request failed: {e}")

    def generate_chat(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text from conversation using Ollama chat API."""
        url = f"{self.host}/api/chat"

        payload = {
            "model": self.model,
            "messages": [msg.model_dump() for msg in messages],
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
                **self.extra_params,
                **kwargs
            }
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=self.model,
                tokens_used=data.get("eval_count"),
                finish_reason=data.get("done_reason"),
                metadata={
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                }
            )
        except requests.RequestException as e:
            raise LLMProviderError(f"Ollama chat API request failed: {e}")

    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Ollama doesn't provide tokenizer, use approximation
        # Average ~4 characters per token for English text
        return len(text) // 4

    def check_connection(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
```

#### **2.2. OpenAIProvider** (`huni_tools/providers/llm/openai.py`)

```python
"""
OpenAI LLM provider implementation.

Uses official OpenAI Python SDK.
"""

from typing import List, Optional
from openai import OpenAI
from huni_tools.base.llm import BaseLLMProvider, LLMMessage, LLMResponse
from huni_tools.base.exceptions import LLMProviderError


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider.

    Supports GPT-4, GPT-3.5, and other OpenAI models.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters
        """
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text from prompt using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs
            )

            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens,
                finish_reason=choice.finish_reason,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
            )
        except Exception as e:
            raise LLMProviderError(f"OpenAI API request failed: {e}")

    def generate_chat(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text from conversation using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs
            )

            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens,
                finish_reason=choice.finish_reason,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
            )
        except Exception as e:
            raise LLMProviderError(f"OpenAI API request failed: {e}")

    def count_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback to approximation
            return len(text) // 4
```

#### **2.3. LocalStorage** (`huni_tools/providers/storage/local.py`)

```python
"""
Local filesystem storage implementation.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from huni_tools.base.storage import BaseStorage, StorageMetadata
from huni_tools.base.exceptions import StorageError


class LocalStorage(BaseStorage):
    """
    Local filesystem storage backend.

    Stores files in specified base directory.
    """

    def __init__(self, base_path: str):
        """
        Initialize local storage.

        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save data to local filesystem."""
        try:
            file_path = self.base_path / key
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(data)

            # Save metadata if provided
            if metadata or content_type:
                meta_path = file_path.with_suffix(file_path.suffix + '.meta')
                import json
                meta_data = {
                    "content_type": content_type,
                    "metadata": metadata,
                    "created_at": datetime.now().isoformat(),
                }
                meta_path.write_text(json.dumps(meta_data))

            return str(file_path)
        except Exception as e:
            raise StorageError(f"Failed to save file: {e}")

    def load(self, key: str) -> bytes:
        """Load data from local filesystem."""
        try:
            file_path = self.base_path / key
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {key}")
            return file_path.read_bytes()
        except FileNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to load file: {e}")

    def exists(self, key: str) -> bool:
        """Check if file exists."""
        file_path = self.base_path / key
        return file_path.exists()

    def delete(self, key: str) -> bool:
        """Delete file from local filesystem."""
        try:
            file_path = self.base_path / key
            if not file_path.exists():
                return False

            file_path.unlink()

            # Delete metadata if exists
            meta_path = file_path.with_suffix(file_path.suffix + '.meta')
            if meta_path.exists():
                meta_path.unlink()

            return True
        except Exception as e:
            raise StorageError(f"Failed to delete file: {e}")

    def list(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[StorageMetadata]:
        """List files in local storage."""
        try:
            if prefix:
                pattern = prefix + "*"
                files = list(self.base_path.glob(pattern))
            else:
                files = list(self.base_path.rglob("*"))

            # Filter out metadata files
            files = [f for f in files if f.is_file() and not f.name.endswith('.meta')]

            # Sort by modification time (newest first)
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            if limit:
                files = files[:limit]

            return [self.get_metadata(str(f.relative_to(self.base_path))) for f in files]
        except Exception as e:
            raise StorageError(f"Failed to list files: {e}")

    def get_metadata(self, key: str) -> StorageMetadata:
        """Get metadata for file."""
        try:
            file_path = self.base_path / key
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {key}")

            stat = file_path.stat()

            # Load custom metadata if exists
            meta_path = file_path.with_suffix(file_path.suffix + '.meta')
            custom_metadata = None
            content_type = None
            if meta_path.exists():
                import json
                meta_data = json.loads(meta_path.read_text())
                custom_metadata = meta_data.get("metadata")
                content_type = meta_data.get("content_type")

            return StorageMetadata(
                key=key,
                size_bytes=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                content_type=content_type,
                custom_metadata=custom_metadata
            )
        except FileNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to get metadata: {e}")
```

#### **2.4. FAISSVectorStore** (`huni_tools/providers/vector_stores/faiss.py`)

```python
"""
FAISS vector store implementation.

Uses Facebook AI Similarity Search for efficient vector retrieval.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from pathlib import Path
import json
from huni_tools.base.vector_store import (
    BaseVectorStore,
    VectorDocument,
    SearchResult
)
from huni_tools.base.exceptions import VectorStoreError


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS vector store implementation.

    Supports local persistence and efficient similarity search.
    """

    def __init__(
        self,
        dimension: int,
        index_path: Optional[str] = None,
        index_type: str = "Flat"  # "Flat", "IVF", "HNSW"
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Vector embedding dimension
            index_path: Path to save/load index (optional)
            index_type: FAISS index type
        """
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None

        # Create FAISS index
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Document storage (ID -> VectorDocument)
        self.documents: Dict[str, VectorDocument] = {}

        # Load existing index if path provided
        if self.index_path and self.index_path.exists():
            self.load()

    def add(self, documents: List[VectorDocument]) -> List[str]:
        """Add documents to FAISS index."""
        try:
            # Convert vectors to numpy array
            vectors = np.array([doc.vector for doc in documents]).astype('float32')

            # Add to FAISS index
            self.index.add(vectors)

            # Store documents
            doc_ids = []
            for doc in documents:
                self.documents[doc.id] = doc
                doc_ids.append(doc.id)

            # Save if path provided
            if self.index_path:
                self.save()

            return doc_ids
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents: {e}")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        try:
            # Convert query to numpy array
            query_array = np.array([query_vector]).astype('float32')

            # Perform search
            distances, indices = self.index.search(query_array, top_k)

            # Convert results
            results = []
            doc_list = list(self.documents.values())

            for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx < len(doc_list):
                    doc = doc_list[idx]

                    # Apply metadata filter if provided
                    if filter:
                        if not self._matches_filter(doc, filter):
                            continue

                    # Convert distance to similarity score (0-1)
                    score = 1.0 / (1.0 + distance)

                    results.append(SearchResult(
                        document=doc,
                        score=score,
                        rank=rank
                    ))

            return results
        except Exception as e:
            raise VectorStoreError(f"Failed to search: {e}")

    def delete(self, document_ids: List[str]) -> int:
        """Delete documents (rebuild index)."""
        try:
            # Remove from documents dict
            deleted_count = 0
            for doc_id in document_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                    deleted_count += 1

            # Rebuild index
            if deleted_count > 0:
                self._rebuild_index()

            return deleted_count
        except Exception as e:
            raise VectorStoreError(f"Failed to delete documents: {e}")

    def update(self, document_id: str, document: VectorDocument) -> bool:
        """Update document (delete + add)."""
        try:
            if document_id not in self.documents:
                return False

            # Update document
            self.documents[document_id] = document

            # Rebuild index
            self._rebuild_index()

            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to update document: {e}")

    def count(self) -> int:
        """Count total documents."""
        return len(self.documents)

    def save(self):
        """Save index to disk."""
        if not self.index_path:
            return

        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))

            # Save documents
            docs_path = self.index_path.with_suffix('.docs.json')
            with open(docs_path, 'w') as f:
                json.dump(
                    {doc_id: doc.model_dump() for doc_id, doc in self.documents.items()},
                    f
                )
        except Exception as e:
            raise VectorStoreError(f"Failed to save index: {e}")

    def load(self):
        """Load index from disk."""
        if not self.index_path or not self.index_path.exists():
            return

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Load documents
            docs_path = self.index_path.with_suffix('.docs.json')
            if docs_path.exists():
                with open(docs_path, 'r') as f:
                    docs_data = json.load(f)
                    self.documents = {
                        doc_id: VectorDocument(**doc_data)
                        for doc_id, doc_data in docs_data.items()
                    }
        except Exception as e:
            raise VectorStoreError(f"Failed to load index: {e}")

    def _rebuild_index(self):
        """Rebuild FAISS index from documents."""
        # Reset index
        self.index.reset()

        # Re-add all documents
        if self.documents:
            vectors = np.array([doc.vector for doc in self.documents.values()]).astype('float32')
            self.index.add(vectors)

        # Save if path provided
        if self.index_path:
            self.save()

    def _matches_filter(self, doc: VectorDocument, filter: Dict[str, Any]) -> bool:
        """Check if document matches metadata filter."""
        if not doc.metadata:
            return False

        for key, value in filter.items():
            if key not in doc.metadata or doc.metadata[key] != value:
                return False

        return True
```

#### **2.5. TavilySearchProvider** (`huni_tools/providers/search/tavily.py`)

```python
"""
Tavily search provider implementation.

Uses Tavily API for web search.
"""

from typing import Optional
from datetime import datetime
from tavily import TavilyClient
from huni_tools.base.search import (
    BaseSearchProvider,
    SearchResponse,
    SearchResultItem
)
from huni_tools.base.exceptions import SearchProviderError


class TavilySearchProvider(BaseSearchProvider):
    """
    Tavily search provider.

    Requires Tavily API key.
    """

    def __init__(self, api_key: str):
        """
        Initialize Tavily provider.

        Args:
            api_key: Tavily API key
        """
        self.client = TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
        **kwargs
    ) -> SearchResponse:
        """Perform web search using Tavily."""
        try:
            import time
            start_time = time.time()

            # Perform search
            response = self.client.search(
                query=query,
                max_results=max_results,
                **kwargs
            )

            search_time_ms = (time.time() - start_time) * 1000

            # Convert results
            results = []
            for item in response.get("results", []):
                results.append(SearchResultItem(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    score=item.get("score"),
                    metadata={
                        "raw_content": item.get("raw_content"),
                        "published_date": item.get("published_date"),
                    }
                ))

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time_ms=search_time_ms,
                metadata={
                    "answer": response.get("answer"),
                    "query_analyzed": response.get("query"),
                }
            )
        except Exception as e:
            raise SearchProviderError(f"Tavily search failed: {e}")
```

### 3. Tool Implementations

#### **3.1. PythonCoderTool** (`huni_tools/tools/python_coder/tool.py`)

```python
"""
Python code generation and execution tool.

Generates Python code using LLM and executes it in a sandboxed environment.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import uuid
from datetime import datetime

from huni_tools.base.tool import BaseTool, ToolResult, ToolMetadata
from huni_tools.base.llm import BaseLLMProvider
from huni_tools.base.storage import BaseStorage
from huni_tools.base.exceptions import ToolExecutionError, ValidationError

from .generator import CodeGenerator
from .executor import CodeExecutor
from .verifier import CodeVerifier
from .models import (
    CodeGenerationRequest,
    CodeGenerationResult,
    CodeExecutionResult
)


class PythonCoderTool(BaseTool):
    """
    Python code generation and execution tool.

    Uses LLM to generate Python code based on user query,
    then executes code in a sandboxed subprocess.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        storage: BaseStorage,
        timeout: int = 3000,
        max_verification_iterations: int = 3,
        max_execution_attempts: int = 5,
        working_directory: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Python coder tool.

        Args:
            llm: LLM provider for code generation
            storage: Storage backend for saving code/results
            timeout: Code execution timeout in seconds
            max_verification_iterations: Max verification attempts
            max_execution_attempts: Max execution retry attempts
            working_directory: Directory for code execution
            **kwargs: Additional configuration
        """
        super().__init__(name="python_coder", version="2.0.0")

        self.llm = llm
        self.storage = storage
        self.timeout = timeout
        self.max_verification_iterations = max_verification_iterations
        self.max_execution_attempts = max_execution_attempts

        # Set working directory
        if working_directory:
            self.working_dir = Path(working_directory)
        else:
            self.working_dir = Path.cwd() / "scratch"
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-components
        self.generator = CodeGenerator(llm=llm)
        self.executor = CodeExecutor(
            timeout=timeout,
            working_directory=str(self.working_dir)
        )
        self.verifier = CodeVerifier(llm=llm)

    def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute Python coder tool.

        Args:
            query: User query/task description
            context: Additional context (conversation history, etc.)
            file_paths: List of file paths to process
            session_id: Session ID for persistence
            **kwargs: Additional parameters

        Returns:
            ToolResult: Execution result with generated code and output
        """
        # Validate input
        self.validate_input(query=query, context=context, file_paths=file_paths)

        # Create metadata
        execution_id = str(uuid.uuid4())
        metadata = ToolMetadata(
            tool_name=self.name,
            version=self.version,
            execution_id=execution_id,
            started_at=datetime.now(),
            status="running"
        )

        try:
            # Step 1: Generate code
            generation_request = CodeGenerationRequest(
                query=query,
                context=context or {},
                file_paths=file_paths or [],
                session_id=session_id
            )

            generation_result = self.generator.generate(generation_request)

            # Step 2: Verify code
            verified_code = generation_result.code
            verification_history = []

            for i in range(self.max_verification_iterations):
                is_valid, issues = self.verifier.verify(
                    code=verified_code,
                    query=query,
                    context=context or {}
                )

                verification_history.append({
                    "iteration": i + 1,
                    "valid": is_valid,
                    "issues": issues
                })

                if is_valid:
                    break

                # Fix code based on issues
                verified_code = self.verifier.fix_code(
                    code=verified_code,
                    issues=issues
                )

            # Step 3: Execute code with retry
            execution_result = None
            execution_history = []

            for attempt in range(self.max_execution_attempts):
                execution_result = self.executor.execute(verified_code)

                execution_history.append({
                    "attempt": attempt + 1,
                    "success": execution_result.success,
                    "output": execution_result.output,
                    "error": execution_result.error
                })

                if execution_result.success:
                    break

                # Fix code based on error
                verified_code = self.generator.fix_code_from_error(
                    code=verified_code,
                    error=execution_result.error,
                    query=query
                )

            # Step 4: Save code and results
            if session_id:
                code_key = f"{session_id}/{execution_id}.py"
                self.storage.save(
                    key=code_key,
                    data=verified_code.encode('utf-8'),
                    content_type="text/x-python"
                )

            # Create final result
            metadata.completed_at = datetime.now()
            metadata.duration_seconds = (
                metadata.completed_at - metadata.started_at
            ).total_seconds()
            metadata.status = "success" if execution_result.success else "failed"

            if not execution_result.success:
                metadata.error = execution_result.error

            return ToolResult(
                success=execution_result.success,
                output={
                    "code": verified_code,
                    "execution_output": execution_result.output,
                    "execution_error": execution_result.error,
                    "verification_history": verification_history,
                    "execution_history": execution_history,
                },
                metadata=metadata,
                raw_data={
                    "generation_result": generation_result.model_dump(),
                    "execution_result": execution_result.model_dump(),
                }
            )

        except Exception as e:
            metadata.completed_at = datetime.now()
            metadata.duration_seconds = (
                metadata.completed_at - metadata.started_at
            ).total_seconds()
            metadata.status = "failed"
            metadata.error = str(e)

            return ToolResult(
                success=False,
                output=None,
                metadata=metadata
            )

    def validate_input(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        file_paths: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """Validate input parameters."""
        if not query or not isinstance(query, str):
            raise ValidationError("Query must be a non-empty string")

        if context is not None and not isinstance(context, dict):
            raise ValidationError("Context must be a dictionary")

        if file_paths is not None:
            if not isinstance(file_paths, list):
                raise ValidationError("File paths must be a list")

            for path in file_paths:
                if not isinstance(path, str):
                    raise ValidationError("Each file path must be a string")
                if not Path(path).exists():
                    raise ValidationError(f"File not found: {path}")

        return True
```

### 4. Usage Examples

#### **4.1. Basic Usage** (`examples/01_quickstart.py`)

```python
"""
Quickstart example for huni-tools-sdk.

Shows basic setup and usage of Python coder tool.
"""

from huni_tools.providers.llm import OllamaProvider
from huni_tools.providers.storage import LocalStorage
from huni_tools.tools.python_coder import PythonCoderTool


def main():
    # Step 1: Configure LLM provider
    llm = OllamaProvider(
        model="qwen3:8b",
        host="http://127.0.0.1:11434",
        temperature=0.7
    )

    # Step 2: Configure storage
    storage = LocalStorage(
        base_path="./data/scratch"
    )

    # Step 3: Create tool
    python_tool = PythonCoderTool(
        llm=llm,
        storage=storage,
        timeout=30,
        max_verification_iterations=3,
        max_execution_attempts=5
    )

    # Step 4: Execute tool
    result = python_tool.execute(
        query="Calculate the average of numbers [1, 2, 3, 4, 5]",
        session_id="demo-session"
    )

    # Step 5: Check result
    if result.success:
        print("✅ Success!")
        print(f"Generated code:\n{result.output['code']}")
        print(f"Output: {result.output['execution_output']}")
    else:
        print("❌ Failed!")
        print(f"Error: {result.metadata.error}")


if __name__ == "__main__":
    main()
```

#### **4.2. Custom LLM Provider** (`examples/06_custom_llm_provider.py`)

```python
"""
Example of creating a custom LLM provider.

Shows how to extend BaseLLMProvider for unsupported LLM backends.
"""

from typing import List, Optional
from huni_tools.base.llm import BaseLLMProvider, LLMMessage, LLMResponse


class CustomLLMProvider(BaseLLMProvider):
    """
    Custom LLM provider example.

    Replace this with your own LLM backend logic.
    """

    def __init__(self, model: str, api_endpoint: str, **kwargs):
        super().__init__(model, **kwargs)
        self.api_endpoint = api_endpoint

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text from prompt."""
        # Implement your custom logic here
        # Example: call your proprietary LLM API

        response_text = f"Response to: {prompt}"  # Replace with actual API call

        return LLMResponse(
            content=response_text,
            model=self.model,
            tokens_used=100,
            metadata={"custom_field": "value"}
        )

    def generate_chat(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text from conversation."""
        # Implement chat logic
        last_message = messages[-1].content
        return self.generate(last_message, temperature, max_tokens, **kwargs)

    def count_tokens(self, text: str) -> int:
        """Count tokens."""
        return len(text) // 4  # Simple approximation


# Usage
def main():
    llm = CustomLLMProvider(
        model="my-custom-model",
        api_endpoint="https://api.example.com/llm"
    )

    response = llm.generate("Hello, world!")
    print(response.content)


if __name__ == "__main__":
    main()
```

---

## Migration Strategy

### Phase 1: Foundation Setup (Days 1-3)

**Goal:** Create new repository with base abstractions.

#### Tasks:

1. **Create Repository**
   - Create new GitHub repository: `huni-tools-sdk`
   - Initialize with README, LICENSE (MIT), .gitignore
   - Set up branch protection for `main` branch

2. **Project Structure**
   ```bash
   mkdir -p huni-tools-sdk/huni_tools/{base,providers,tools,utils}
   mkdir -p huni-tools-sdk/{examples,tests,docs}
   ```

3. **Create Base Abstractions**
   - `base/tool.py` - BaseTool, ToolResult, ToolMetadata
   - `base/llm.py` - BaseLLMProvider, LLMMessage, LLMResponse
   - `base/storage.py` - BaseStorage, StorageMetadata
   - `base/vector_store.py` - BaseVectorStore, VectorDocument
   - `base/embeddings.py` - BaseEmbeddings
   - `base/search.py` - BaseSearchProvider
   - `base/exceptions.py` - Custom exceptions

4. **Setup Package Configuration**
   - Create `pyproject.toml`:
     ```toml
     [build-system]
     requires = ["setuptools>=61.0", "wheel"]
     build-backend = "setuptools.build_meta"

     [project]
     name = "huni-tools-sdk"
     version = "1.0.0"
     description = "Reusable AI tool library with LLM abstraction"
     readme = "README.md"
     requires-python = ">=3.9"
     license = {text = "MIT"}

     dependencies = [
         "pydantic>=2.0.0",
         "requests>=2.28.0",
         "numpy>=1.24.0",
     ]

     [project.optional-dependencies]
     ollama = ["ollama>=0.1.0"]
     openai = ["openai>=1.0.0"]
     faiss = ["faiss-cpu>=1.7.4"]
     tavily = ["tavily-python>=0.3.0"]
     all = [
         "ollama>=0.1.0",
         "openai>=1.0.0",
         "faiss-cpu>=1.7.4",
         "tavily-python>=0.3.0",
     ]
     dev = [
         "pytest>=7.4.0",
         "pytest-cov>=4.1.0",
         "black>=23.0.0",
         "mypy>=1.5.0",
         "ruff>=0.0.290",
     ]
     ```

5. **Write Unit Tests for Base Classes**
   - `tests/base/test_tool.py`
   - `tests/base/test_llm.py`
   - `tests/base/test_storage.py`

6. **Documentation**
   - Write comprehensive README.md
   - Create `docs/architecture.md`
   - Create `docs/quickstart.md`

**Deliverables:**
- ✅ Repository created with base abstractions
- ✅ Unit tests passing (90%+ coverage)
- ✅ Documentation complete
- ✅ Package installable locally: `pip install -e .`

### Phase 2: Provider Implementations (Days 4-6)

**Goal:** Implement concrete providers for LLM, storage, etc.

#### Tasks:

1. **LLM Providers**
   - `providers/llm/ollama.py` - OllamaProvider
   - `providers/llm/openai.py` - OpenAIProvider
   - Tests: `tests/providers/test_ollama.py`, `test_openai.py`

2. **Storage Providers**
   - `providers/storage/local.py` - LocalStorage
   - `providers/storage/s3.py` - S3Storage (optional)
   - Tests: `tests/providers/test_local_storage.py`

3. **Vector Store Providers**
   - `providers/vector_stores/faiss.py` - FAISSVectorStore
   - Tests: `tests/providers/test_faiss.py`

4. **Embeddings Providers**
   - `providers/embeddings/ollama.py` - OllamaEmbeddings
   - `providers/embeddings/openai.py` - OpenAIEmbeddings
   - Tests: `tests/providers/test_embeddings.py`

5. **Search Providers**
   - `providers/search/tavily.py` - TavilySearchProvider
   - Tests: `tests/providers/test_tavily.py`

6. **Integration Tests**
   - Test provider switching (Ollama → OpenAI)
   - Test end-to-end workflows

**Deliverables:**
- ✅ All providers implemented
- ✅ Provider tests passing (85%+ coverage)
- ✅ Provider documentation complete

### Phase 3: Migrate Python Coder Tool (Days 7-10)

**Goal:** Extract and refactor Python coder tool.

#### Tasks:

1. **Copy Current Implementation**
   ```bash
   cp -r LLM_API/backend/tools/python_coder huni-tools-sdk/huni_tools/tools/
   ```

2. **Refactor Dependencies**
   - Replace `from backend.config.settings import settings`
     With: Dependency injection via `__init__()`
   - Replace `from backend.utils.llm_factory import create_llm`
     With: Accept `BaseLLMProvider` in constructor
   - Replace hardcoded paths with `BaseStorage`

3. **Update Code**
   ```python
   # Before
   class PythonCoderTool:
       def __init__(self):
           self.llm = create_llm(model=settings.ollama_coder_model)
           self.execution_dir = Path(settings.python_code_execution_dir)

   # After
   class PythonCoderTool(BaseTool):
       def __init__(
           self,
           llm: BaseLLMProvider,
           storage: BaseStorage,
           timeout: int = 3000
       ):
           self.llm = llm
           self.storage = storage
           self.timeout = timeout
   ```

4. **Create Tests**
   - `tests/tools/test_python_coder.py`
   - Test code generation
   - Test code execution
   - Test verification loop
   - Test retry logic

5. **Create Examples**
   - `examples/02_python_coder_example.py`
   - Show basic usage
   - Show file processing
   - Show variable persistence

6. **Update Main Project**
   - In `LLM_API/backend/adapters/tool_factory.py`:
     ```python
     from huni_tools.tools.python_coder import PythonCoderTool
     from huni_tools.providers.llm import OllamaProvider
     from huni_tools.providers.storage import LocalStorage

     def create_python_coder_tool(settings):
         """Factory for creating configured Python coder tool."""
         llm = OllamaProvider(
             model=settings.ollama_coder_model,
             host=settings.ollama_host,
             temperature=settings.ollama_coder_model_temperature
         )

         storage = LocalStorage(
             base_path=settings.python_code_execution_dir
         )

         return PythonCoderTool(
             llm=llm,
             storage=storage,
             timeout=settings.python_code_timeout,
             max_verification_iterations=settings.python_code_max_iterations
         )
     ```

**Deliverables:**
- ✅ Python coder tool migrated and working
- ✅ Tests passing (90%+ coverage)
- ✅ Main project updated to use SDK
- ✅ Backward compatibility maintained

### Phase 4: Migrate Remaining Tools (Days 11-15)

**Goal:** Migrate web search, RAG retriever, and file analyzer.

#### Tasks:

1. **Web Search Tool**
   - Migrate `backend/tools/web_search/` → `huni_tools/tools/web_search/`
   - Refactor to use `BaseSearchProvider`
   - Write tests
   - Create examples

2. **RAG Retriever Tool**
   - Migrate `backend/tools/rag_retriever/` → `huni_tools/tools/rag_retriever/`
   - Refactor to use `BaseVectorStore` and `BaseEmbeddings`
   - Write tests
   - Create examples

3. **File Analyzer Tool**
   - Migrate `backend/tools/file_analyzer/` → `huni_tools/tools/file_analyzer/`
   - Refactor to use `BaseLLMProvider`
   - Write tests
   - Create examples

4. **Update Main Project**
   - Update `backend/adapters/tool_factory.py` to create all tools
   - Update `backend/tasks/react/tool_executor.py` to use SDK tools
   - Test end-to-end workflows

**Deliverables:**
- ✅ All tools migrated
- ✅ Tests passing for all tools
- ✅ Main project fully using SDK
- ✅ No breaking changes to API

### Phase 5: Documentation & Distribution (Days 16-18)

**Goal:** Prepare for public release.

#### Tasks:

1. **Documentation**
   - Write comprehensive API reference (Sphinx)
   - Create user guides:
     - Installation guide
     - Quickstart tutorial
     - Custom provider guide
     - Best practices
   - Create example notebooks (Jupyter)

2. **Testing & Quality**
   - Run full test suite
   - Achieve 90%+ code coverage
   - Run type checking with mypy
   - Run linting with ruff/black
   - Test on multiple Python versions (3.9, 3.10, 3.11)

3. **CI/CD Setup**
   - GitHub Actions for:
     - Running tests on PR
     - Running linters
     - Building documentation
     - Publishing to PyPI (on release tag)

4. **Package Distribution**
   - Build package: `python -m build`
   - Test installation: `pip install dist/*.whl`
   - Publish to TestPyPI first
   - Publish to PyPI: `twine upload dist/*`

5. **Release**
   - Tag version: `git tag v1.0.0`
   - Create GitHub release with changelog
   - Announce on relevant forums/communities

**Deliverables:**
- ✅ Package published to PyPI
- ✅ Documentation hosted (GitHub Pages or Read the Docs)
- ✅ CI/CD pipeline working
- ✅ Public release announced

### Phase 6: Maintenance & Growth (Ongoing)

**Goal:** Support community and add features.

#### Tasks:

1. **Community Support**
   - Monitor GitHub issues
   - Respond to questions
   - Review pull requests
   - Update documentation based on feedback

2. **Feature Additions**
   - Add new LLM providers (Anthropic, Azure, etc.)
   - Add new vector stores (Pinecone, Weaviate, etc.)
   - Add new tools based on community requests

3. **Versioning**
   - Follow semantic versioning
   - Maintain backward compatibility
   - Document breaking changes clearly

---

## Implementation Timeline

### Gantt Chart

```
Week 1 (Days 1-7):
│ Phase 1: Foundation Setup     [████████████] Days 1-3
│ Phase 2: Provider Impl (Start) [███████.....] Days 4-6
│ Phase 3: Python Coder (Start)  [...........█] Day 7

Week 2 (Days 8-14):
│ Phase 3: Python Coder (Complete) [███████████] Days 7-10
│ Phase 4: Remaining Tools         [████████...] Days 11-14

Week 3 (Days 15-21):
│ Phase 4: Remaining Tools (Complete) [██.........] Days 14-15
│ Phase 5: Documentation & Dist      [███████████] Days 16-18
│ Phase 6: Launch & Support          [......█████] Days 19-21
```

### Milestones

| Milestone | Target Date | Deliverable |
|-----------|-------------|-------------|
| M1: Base abstractions complete | Day 3 | Repository with base classes |
| M2: Providers implemented | Day 6 | All provider classes working |
| M3: First tool migrated | Day 10 | Python coder in SDK |
| M4: All tools migrated | Day 15 | Complete SDK with all tools |
| M5: Package published | Day 18 | PyPI package available |
| M6: Public launch | Day 21 | Community announcement |

---

## Testing Strategy

### Unit Tests

**Coverage Target:** 90%+

#### Base Classes (`tests/base/`)
- `test_tool.py` - Test BaseTool contract
- `test_llm.py` - Test BaseLLMProvider interface
- `test_storage.py` - Test BaseStorage interface
- `test_vector_store.py` - Test BaseVectorStore interface

#### Providers (`tests/providers/`)
- `test_ollama.py` - Test OllamaProvider
  - Mock HTTP requests
  - Test error handling
  - Test token counting
- `test_openai.py` - Test OpenAIProvider
  - Mock API calls
  - Test rate limiting
- `test_local_storage.py` - Test LocalStorage
  - Create temp directory
  - Test save/load/delete
  - Test metadata

#### Tools (`tests/tools/`)
- `test_python_coder.py` - Test PythonCoderTool
  - Mock LLM responses
  - Test code generation
  - Test verification loop
  - Test execution retry
- `test_web_search.py` - Test WebSearchTool
  - Mock search API
  - Test result processing
- `test_rag_retriever.py` - Test RAGRetrieverTool
  - Mock vector store
  - Test retrieval
- `test_file_analyzer.py` - Test FileAnalyzerTool
  - Test multiple file formats

### Integration Tests (`tests/integration/`)

#### End-to-End Workflows
```python
def test_python_coder_end_to_end():
    """Test complete Python coder workflow."""
    # Setup real Ollama (if available) or mock
    llm = OllamaProvider(model="qwen3:8b")
    storage = LocalStorage(base_path="/tmp/test")

    tool = PythonCoderTool(llm=llm, storage=storage)

    result = tool.execute(
        query="Calculate sum of [1,2,3]",
        session_id="test-session"
    )

    assert result.success
    assert "6" in result.output["execution_output"]
```

#### Provider Switching
```python
def test_switch_llm_provider():
    """Test switching from Ollama to OpenAI."""
    storage = LocalStorage(base_path="/tmp/test")

    # Create tool with Ollama
    ollama_llm = OllamaProvider(model="qwen3:8b")
    tool_ollama = PythonCoderTool(llm=ollama_llm, storage=storage)

    # Create same tool with OpenAI
    openai_llm = OpenAIProvider(model="gpt-4", api_key="test")
    tool_openai = PythonCoderTool(llm=openai_llm, storage=storage)

    # Both should work identically
    query = "Print hello world"
    result1 = tool_ollama.execute(query=query)
    result2 = tool_openai.execute(query=query)

    assert result1.success
    assert result2.success
```

### Performance Tests (`tests/performance/`)

#### Load Testing
```python
import concurrent.futures

def test_concurrent_execution():
    """Test tool under concurrent load."""
    llm = OllamaProvider(model="qwen3:8b")
    storage = LocalStorage(base_path="/tmp/test")
    tool = PythonCoderTool(llm=llm, storage=storage)

    queries = [f"Calculate {i} * 2" for i in range(10)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(tool.execute, query=q)
            for q in queries
        ]
        results = [f.result() for f in futures]

    assert all(r.success for r in results)
```

### Test Configuration (`pytest.ini`)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --verbose
    --cov=huni_tools
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=90
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow tests (skip by default)
```

---

## Deployment & Distribution

### Package Building

#### 1. **Version Management** (`huni_tools/version.py`)

```python
"""Version information for huni-tools-sdk."""

__version__ = "1.0.0"

def get_version():
    """Return the version string."""
    return __version__
```

#### 2. **Build Package**

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# This creates:
# - dist/huni_tools_sdk-1.0.0-py3-none-any.whl
# - dist/huni_tools_sdk-1.0.0.tar.gz
```

#### 3. **Test Installation Locally**

```bash
# Install in virtual environment
python -m venv test_env
source test_env/bin/activate
pip install dist/huni_tools_sdk-1.0.0-py3-none-any.whl

# Test import
python -c "from huni_tools import PythonCoderTool; print('Success!')"
```

### PyPI Publishing

#### 1. **Create PyPI Account**
- Sign up at https://pypi.org/
- Create API token
- Save token to `~/.pypirc`

#### 2. **Test on TestPyPI First**

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ huni-tools-sdk
```

#### 3. **Publish to PyPI**

```bash
# Upload to PyPI
twine upload dist/*

# Now anyone can install:
pip install huni-tools-sdk
```

### GitHub Actions CI/CD

#### `.github/workflows/test.yml`

```yaml
name: Tests

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run linters
        run: |
          black --check huni_tools tests
          ruff check huni_tools tests
          mypy huni_tools

      - name: Run tests
        run: |
          pytest --cov=huni_tools --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

#### `.github/workflows/publish.yml`

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

### Documentation Hosting

#### **Option 1: GitHub Pages**

```yaml
# .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [main]

jobs:
  build-docs:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install sphinx sphinx-rtd-theme
          pip install -e .

      - name: Build docs
        run: |
          cd docs
          make html

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_build/html
```

#### **Option 2: Read the Docs**

- Connect repository to https://readthedocs.org/
- Auto-builds on each commit
- Supports versioned documentation

---

## Risk Assessment & Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Breaking changes in main project** | Medium | High | • Maintain backward compatibility via adapters<br>• Comprehensive integration tests<br>• Feature flags for gradual migration |
| **Performance regression** | Low | Medium | • Performance benchmarks in CI<br>• Load testing before release<br>• Profiling of critical paths |
| **Dependency conflicts** | Medium | Low | • Minimal core dependencies<br>• Optional dependencies for providers<br>• Pin major versions only |
| **LLM provider API changes** | Medium | Medium | • Abstract provider interface<br>• Provider-specific tests<br>• Version pinning for stability |
| **Security vulnerabilities** | Low | High | • Code execution sandboxing maintained<br>• Input validation at all boundaries<br>• Security audit before v1.0 |

### Organizational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Scope creep** | High | Medium | • Strict phase boundaries<br>• MVP-first approach<br>• Defer nice-to-have features |
| **Time overrun** | Medium | Medium | • Buffer time in schedule (20%)<br>• Daily progress tracking<br>• Early escalation of blockers |
| **Adoption resistance** | Low | Low | • Clear migration guides<br>• Side-by-side comparison examples<br>• Maintain legacy compatibility |
| **Maintenance burden** | Medium | Medium | • Comprehensive documentation<br>• Automated testing & CI/CD<br>• Clear contribution guidelines |

### Quality Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Inadequate test coverage** | Low | High | • 90%+ coverage requirement<br>• Automated coverage reporting<br>• Integration tests for all workflows |
| **Poor documentation** | Medium | High | • Documentation as deliverable<br>• Example-driven docs<br>• Community feedback loop |
| **Inconsistent APIs** | Low | Medium | • Design review before implementation<br>• API consistency checklist<br>• Type hints everywhere |

---

## Success Metrics

### Development Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Test Coverage** | ≥ 90% | Pytest coverage report |
| **Code Quality** | A grade | Ruff/SonarQube |
| **Type Safety** | 100% typed | Mypy strict mode |
| **Documentation Coverage** | 100% of public APIs | Sphinx coverage plugin |
| **Build Success Rate** | ≥ 95% | GitHub Actions |

### Adoption Metrics (Post-Launch)

| Metric | 1 Month | 3 Months | 6 Months |
|--------|---------|----------|----------|
| **PyPI Downloads** | 100+ | 500+ | 2000+ |
| **GitHub Stars** | 20+ | 100+ | 300+ |
| **Community PRs** | 2+ | 10+ | 30+ |
| **Issues Resolved** | 80%+ | 85%+ | 90%+ |
| **Documentation Views** | 500+ | 2000+ | 5000+ |

### Technical Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Package Size** | < 5MB | Without optional dependencies |
| **Installation Time** | < 30s | On standard connection |
| **Import Time** | < 500ms | For core modules |
| **LLM Provider Switch Time** | < 5 lines of code | Ease of switching |
| **Backward Compatibility** | 100% | Main project still works |

---

## Appendices

### Appendix A: Comparison Matrix

**Before vs After Migration**

| Aspect | Before (Monolithic) | After (SDK) |
|--------|---------------------|-------------|
| **Reusability** | Tools tied to LLM_API project | Tools work in any Python project |
| **LLM Provider** | Hardcoded Ollama | Configurable (Ollama, OpenAI, custom) |
| **Storage** | Hardcoded local filesystem | Configurable (Local, S3, GCS) |
| **Testing** | Difficult (need full app context) | Easy (mock dependencies) |
| **Distribution** | Git clone + setup | `pip install huni-tools-sdk` |
| **Versioning** | Monolithic versioning | Independent tool versioning |
| **Documentation** | Project-specific | Public API docs |
| **Community** | Internal only | Public contributions welcome |

### Appendix B: File Mapping

**Migration Path for Each File**

| Current Location | New Location | Changes Required |
|------------------|--------------|------------------|
| `backend/tools/python_coder/tool.py` | `huni_tools/tools/python_coder/tool.py` | • Remove `settings` import<br>• Add dependency injection<br>• Use `BaseTool` |
| `backend/tools/python_coder/generator.py` | `huni_tools/tools/python_coder/generator.py` | • Remove `llm_factory` import<br>• Accept `BaseLLMProvider` |
| `backend/tools/python_coder/executor.py` | `huni_tools/tools/python_coder/executor.py` | • Minimal changes (already isolated) |
| `backend/tools/web_search/tool.py` | `huni_tools/tools/web_search/tool.py` | • Remove `settings.tavily_api_key`<br>• Accept `BaseSearchProvider` |
| `backend/tools/rag_retriever/tool.py` | `huni_tools/tools/rag_retriever/tool.py` | • Remove `settings` import<br>• Accept `BaseVectorStore`, `BaseEmbeddings` |
| `backend/tools/file_analyzer/tool.py` | `huni_tools/tools/file_analyzer/tool.py` | • Remove `llm_factory`<br>• Accept `BaseLLMProvider` |

### Appendix C: Configuration Examples

#### **Example 1: Using Ollama (Current Setup)**

```python
# In LLM_API project after migration
from huni_tools.providers.llm import OllamaProvider
from huni_tools.providers.storage import LocalStorage
from huni_tools.tools.python_coder import PythonCoderTool

# This replicates your current setup
llm = OllamaProvider(
    model="qwen3:8b",
    host="http://127.0.0.1:11434",
    temperature=1.0
)

storage = LocalStorage(base_path="./data/scratch")

python_tool = PythonCoderTool(
    llm=llm,
    storage=storage,
    timeout=3000,
    max_verification_iterations=3
)
```

#### **Example 2: Switching to OpenAI**

```python
# Just change the LLM provider - everything else stays the same!
from huni_tools.providers.llm import OpenAIProvider
from huni_tools.providers.storage import LocalStorage
from huni_tools.tools.python_coder import PythonCoderTool

llm = OpenAIProvider(
    model="gpt-4",
    api_key="sk-...",
    temperature=0.7
)

storage = LocalStorage(base_path="./data/scratch")

python_tool = PythonCoderTool(
    llm=llm,  # Only this line changed!
    storage=storage,
    timeout=3000
)
```

#### **Example 3: Using S3 Storage**

```python
from huni_tools.providers.llm import OllamaProvider
from huni_tools.providers.storage import S3Storage
from huni_tools.tools.python_coder import PythonCoderTool

llm = OllamaProvider(model="qwen3:8b")

storage = S3Storage(
    bucket="my-code-executions",
    region="us-west-2"
)

python_tool = PythonCoderTool(
    llm=llm,
    storage=storage,  # Now using S3 instead of local!
    timeout=3000
)
```

### Appendix D: Breaking Changes Checklist

**Before Making Breaking Changes:**

- [ ] Is there a non-breaking alternative?
- [ ] Have we documented the breaking change?
- [ ] Have we provided migration guide?
- [ ] Have we bumped major version (semver)?
- [ ] Have we deprecated old API first (if possible)?
- [ ] Have we updated all examples?
- [ ] Have we updated tests?
- [ ] Have we communicated to users?

### Appendix E: Release Checklist

**Before Each Release:**

- [ ] All tests passing (unit + integration)
- [ ] Code coverage ≥ 90%
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped (semver)
- [ ] No critical security issues
- [ ] Performance benchmarks stable
- [ ] Example code tested
- [ ] Migration guide written (if breaking)
- [ ] Git tag created
- [ ] GitHub release notes written
- [ ] PyPI package uploaded
- [ ] Documentation deployed
- [ ] Community announcement posted

---

## Conclusion

This plan provides a **comprehensive roadmap** for extracting your well-designed tools into a reusable SDK. The migration follows **industry best practices**:

1. **Dependency injection** for flexibility
2. **Abstract base classes** for extensibility
3. **Comprehensive testing** for reliability
4. **Semantic versioning** for stability
5. **Public distribution** for community growth

**Key Advantages:**
- ✅ Tools become **portable** across projects
- ✅ **Swap LLM providers** with 5 lines of code
- ✅ **Independent testing** without full application
- ✅ **Community contributions** welcome
- ✅ **Professional distribution** via PyPI

**Next Steps:**
1. Review this plan and approve strategy
2. Create `huni-tools-sdk` repository
3. Begin Phase 1: Foundation setup
4. Follow migration phases sequentially
5. Launch public beta by Day 18

Would you like me to help you get started with any specific phase? I can:
- Create the initial repository structure
- Write the base abstraction classes
- Set up the first provider implementation
- Create migration scripts for existing tools

Let me know how you'd like to proceed!