"""
LLM API Configuration
All settings are configurable here
"""
from pathlib import Path
from typing import Literal

# ============================================================================
# Server Settings
# ============================================================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 1007  # Main API server (chat, auth, etc.)
TOOLS_PORT = 1006   # Tools API server (websearch, python_coder, rag) - separate to avoid deadlock
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# ============================================================================
# LLM Backend Settings
# ============================================================================
# Which LLM backend to use: "ollama", "llamacpp", or "auto" (tries ollama first, falls back to llamacpp)
LLM_BACKEND: Literal["ollama", "llamacpp", "auto"] = "auto"

# Ollama Settings
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"  # Default model

# Llama.cpp Settings
LLAMACPP_HOST = "http://localhost:8080"
LLAMACPP_MODEL = "default"  # Model loaded in llama.cpp server

# ============================================================================
# Model Parameters (Default LLM Inference Settings)
# ============================================================================
DEFAULT_TEMPERATURE = 0.7  # Randomness (0.0 = deterministic, 2.0 = very random)
DEFAULT_TOP_P = 0.9  # Nucleus sampling
DEFAULT_TOP_K = 40  # Top-k sampling
DEFAULT_MAX_TOKENS = 2048  # Maximum tokens in response

# ============================================================================
# Database Settings
# ============================================================================
DATABASE_PATH = "data/app.db"  # SQLite database path

# ============================================================================
# Authentication Settings
# ============================================================================
JWT_SECRET_KEY = "your-secret-key-change-in-production"  # CHANGE THIS IN PRODUCTION!
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 7 days

# Default admin credentials
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "administrator"

# ============================================================================
# File Storage Settings
# ============================================================================
UPLOAD_DIR = Path("data/uploads")  # User persistent uploads: data/uploads/{username}/
SCRATCH_DIR = Path("data/scratch")  # Session temporary files: data/scratch/{session_id}/
MAX_FILE_SIZE_MB = 100  # Maximum file upload size in MB

# ============================================================================
# Prompts Settings
# ============================================================================
PROMPTS_DIR = Path("prompts")  # System prompts directory

# ============================================================================
# Logging Settings
# ============================================================================
LOG_DIR = Path("data/logs")  # Logs directory
PROMPTS_LOG_PATH = LOG_DIR / "prompts.log"  # LLM interaction logs

# ============================================================================
# Agent Settings
# ============================================================================
# Available agent types
AVAILABLE_AGENTS = ["auto", "react", "plan_execute", "chat"]
DEFAULT_AGENT = "auto"

# ReAct Agent Settings
REACT_FORMAT = "prompt"  # "prompt" or "native" (Ollama native tool calling)
REACT_MAX_ITERATIONS = 10  # Maximum reasoning iterations
REACT_RETRY_ON_ERROR = True  # Retry failed tool calls

# Plan-Execute Agent Settings
PLAN_MAX_STEPS = 10  # Maximum plan steps
PLAN_REPLAN_ON_FAILURE = True  # Re-plan when a step fails
PLAN_SHARE_CONTEXT = True  # Share context across plan execution steps

# ============================================================================
# Tools Settings (for future implementation)
# ============================================================================
# Available tools - these will be implemented later
AVAILABLE_TOOLS = [
    "websearch",
    "python_coder",
    "rag",
]

# Tool-specific Model Configuration
# Different tools can use different models optimized for their tasks
TOOL_MODELS = {
    "websearch": "gpt-oss:20b",  # Web search summarization
    "python_coder": "gpt-oss:20b",  # Code generation
    "rag": "gpt-oss:20b",  # Document retrieval
}

# Tool-specific Model Parameters
# Each tool can have customized inference parameters
TOOL_PARAMETERS = {
    "websearch": {
        "temperature": 0.7,
        "max_tokens": 2048,
        "timeout": 360,  # 1 minute for web search (includes 2 LLM calls + Tavily API call)
    },
    "python_coder": {
        "temperature": 0.3,  # Lower temp for more deterministic code
        "max_tokens": 2048,
        "timeout": 120,  # 2 minutes for code execution
    },
    "rag": {
        "temperature": 0.5,
        "max_tokens": 2048,
        "timeout": 120,  # 2 minutes for RAG retrieval
    },
}

# Default tool timeout (seconds)
DEFAULT_TOOL_TIMEOUT = 60

# ============================================================================
# Web Search Tool Settings
# ============================================================================
WEBSEARCH_PROVIDER = "tavily"  # Options: "tavily", "serper", "mock"
TAVILY_API_KEY = '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'
TAVILY_MAX_RESULTS = 5  # Maximum search results to retrieve
TAVILY_SEARCH_DEPTH = "advanced"  # "basic" or "advanced" - use "basic" for faster responses
TAVILY_INCLUDE_DOMAINS = []  # List of domains to include (empty = all)
TAVILY_EXCLUDE_DOMAINS = []  # List of domains to exclude
WEBSEARCH_MAX_RESULTS = 5  # Maximum search results for ReAct agent

# ============================================================================
# Python Coder Tool Settings
# ============================================================================
PYTHON_EXECUTOR_TIMEOUT = 30  # Execution timeout in seconds
PYTHON_EXECUTOR_MAX_OUTPUT_SIZE = 1024 * 1024  # 1MB max output
PYTHON_WORKSPACE_DIR = SCRATCH_DIR  # Uses session scratch directory
PYTHON_CODER_TIMEOUT = 30  # Timeout for ReAct agent tool call

# ============================================================================
# RAG Tool Settings
# ============================================================================
RAG_DOCUMENTS_DIR = Path("data/rag_documents")  # Document storage
RAG_INDEX_DIR = Path("data/rag_indices")  # FAISS indices storage
RAG_METADATA_DIR = Path("data/rag_metadata")  # Metadata storage

# Embedding Model Settings
RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace model
RAG_EMBEDDING_DEVICE = "cpu"  # "cpu" or "cuda"
RAG_EMBEDDING_BATCH_SIZE = 32

# FAISS Settings
RAG_INDEX_TYPE = "Flat"  # "Flat", "IVF", or "HNSW"
RAG_SIMILARITY_METRIC = "cosine"  # "cosine", "l2", or "ip" (inner product)

# Chunking Settings
RAG_CHUNK_SIZE = 512  # Characters per chunk
RAG_CHUNK_OVERLAP = 50  # Overlap between chunks
RAG_MAX_RESULTS = 5  # Maximum documents to retrieve
RAG_DEFAULT_COLLECTION = "default"  # Default collection name

# Supported document formats
RAG_SUPPORTED_FORMATS = [".txt", ".pdf", ".docx", ".md", ".json", ".csv"]

# ============================================================================
# Session Settings
# ============================================================================
MAX_CONVERSATION_HISTORY = 50  # Maximum messages to keep in conversation history
SESSION_CLEANUP_DAYS = 7  # Delete sessions older than this

# ============================================================================
# Streaming Settings
# ============================================================================
STREAM_CHUNK_SIZE = 1  # How many tokens to send per SSE chunk
STREAM_TIMEOUT = 3600  # Streaming timeout in seconds (1 hour)

# ============================================================================
# CORS Settings
# ============================================================================
CORS_ORIGINS = [
    "http://localhost:1007",
    "http://127.0.0.1:1007",
]

# ============================================================================
# Ensure directories exist
# ============================================================================
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RAG_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
RAG_METADATA_DIR.mkdir(parents=True, exist_ok=True)
