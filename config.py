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
SERVER_PORT = 10007  # Main API server (chat, auth, etc.)
TOOLS_HOST = "127.0.0.1"  # Tools server host (change if on different machine)
TOOLS_PORT = 10006   # Tools API server (websearch, python_coder, rag) - separate to avoid deadlock
WEBSEARCH_SERVER_HOST = "10.252.38.241"  # Remote websearch server (fallback: local tools server)
WEBSEARCH_SERVER_PORT = 10006            # Remote websearch server port
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

SERVER_WORKERS = 4  # Number of parallel workers for main API
TOOLS_SERVER_WORKERS = 4  # Number of parallel workers for tools API

LLM_BACKEND: Literal["ollama", "llamacpp", "auto"] = "llamacpp"


OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"

# Model Preloading Settings
PRELOAD_MODEL_ON_STARTUP = False  # Preload default model to GPU on server startup
PRELOAD_KEEP_ALIVE = -1  # Keep model in memory: -1 = indefinitely, "5m" = 5 minutes, 0 = unload immediately

# Llama.cpp Settings
LLAMACPP_HOST = "http://localhost:5904"
LLAMACPP_MODEL = "default"  # Model loaded in llama.cpp server

# ============================================================================
# Model Parameters (Default LLM Inference Settings)
# ============================================================================
DEFAULT_TEMPERATURE = 0.7  # Randomness (0.0 = deterministic, 2.0 = very random)
DEFAULT_TOP_P = 0.9  # Nucleus sampling
DEFAULT_TOP_K = 40  # Top-k sampling
DEFAULT_MAX_TOKENS = 128000  # Maximum tokens in response

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
# Stop Signal Settings
# ============================================================================
STOP_FILE = Path("data/STOP")  # Create this file to halt all running inference

# ============================================================================
# Agent Settings
# ============================================================================
# Available agent types
AVAILABLE_AGENTS = ["auto", "react", "plan_execute", "ultrawork", "chat"]
DEFAULT_AGENT = "auto"

# ReAct Agent Settings
REACT_FORMAT = "prompt"  # "prompt" or "native" (Ollama native tool calling)
REACT_MAX_ITERATIONS = 5  # Maximum reasoning iterations
REACT_RETRY_ON_ERROR = True  # Let LLM see errors and decide next action (intelligent retry)
REACT_MAX_PARSE_RETRIES = 3  # Maximum retries for parsing Thought/Action/Action Input (if LLM fails to follow format)

# Plan-Execute Agent Settings
PLAN_MAX_STEPS = 5  # Maximum plan steps
PLAN_MAX_ITERATIONS = 3  # Maximum iterations of plan-execute-replan cycle
PLAN_REPLAN_ON_FAILURE = True  # Re-plan when a step fails
PLAN_MIN_REACT_ITERATIONS_FOR_REPLAN = REACT_MAX_ITERATIONS  # Minimum React iterations before allowing step-level replanning
PLAN_SHARE_CONTEXT = True  # Share context across plan execution steps
PLAN_INCLUDE_FULL_HISTORY = True  # Include all conversation history in planning
PLAN_MAX_HISTORY_MESSAGES = 0  # Maximum history messages to include (0 = unlimited)
PLAN_HISTORY_IN_SYNTHESIS = True  # Include conversation history in final synthesis

# Ultrawork Agent Settings (replaces plan_execute when PYTHON_EXECUTOR_MODE="opencode")
ULTRAWORK_MAX_ITERATIONS = 5  # Maximum refinement iterations
ULTRAWORK_VERIFY_TEMPERATURE = 0.3  # Temperature for verification (low for consistency)

# ============================================================================
# Tools Settings (for future implementation)
# ============================================================================
# Available tools - these will be implemented later
AVAILABLE_TOOLS = [
    "websearch",
    "python_coder",
    "rag",
]

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
        "max_tokens": 30000,
        "timeout": 864000,  # 10 days for web search
    },
    "python_coder": {
        "temperature": 1.0,  # Lower temp for more deterministic code
        "max_tokens": 128000,
        "timeout": 864000,  # 10 days for code execution
    },
    "rag": {
        "temperature": 0.2,  # Low temperature for factual synthesis from retrieved documents
        "max_tokens": 30000,
        "timeout": 864000,  # 10 days for RAG retrieval
    },
}

# Default tool timeout (seconds) - 10 days
DEFAULT_TOOL_TIMEOUT = 864000

# ============================================================================
# Web Search Tool Settings
# ============================================================================
WEBSEARCH_PROVIDER = "tavily"  # Options: "tavily", "serper", "mock"
TAVILY_API_KEY = "your-secret-key-change-in-production"
TAVILY_MAX_RESULTS = 5  # Maximum search results to retrieve
TAVILY_SEARCH_DEPTH = "advanced"  # "basic" or "advanced" - use "basic" for faster responses
TAVILY_INCLUDE_DOMAINS = []  # List of domains to include (empty = all)
TAVILY_EXCLUDE_DOMAINS = []  # List of domains to exclude
WEBSEARCH_MAX_RESULTS = 5  # Maximum search results for ReAct agent

# ============================================================================
# Python Coder Tool Settings
# ============================================================================
# Execution mode: "native" or "opencode"
# - native: Direct Python code execution (fast, no LLM overhead)
# - opencode: Natural language to code using opencode AI coding agent (recommended)
PYTHON_EXECUTOR_MODE: Literal["native", "opencode"] = "opencode"

# Native executor settings
PYTHON_EXECUTOR_TIMEOUT = 864000  # Execution timeout in seconds (10 days)
PYTHON_EXECUTOR_MAX_OUTPUT_SIZE = 1024 * 1024 * 10  # 10MB max output
PYTHON_WORKSPACE_DIR = SCRATCH_DIR  # Uses session scratch directory
PYTHON_CODER_TIMEOUT = 864000  # Timeout for ReAct agent tool call (10 days)

# OpenCode settings
OPENCODE_PATH: str = "opencode"  # Path to opencode binary (globally installed via npm)
OPENCODE_SERVER_PORT: int = 37254  # Server port
OPENCODE_SERVER_HOST: str = "127.0.0.1"  # Server host
OPENCODE_TIMEOUT: int = 864000  # Execution timeout in seconds (10 days)
OPENCODE_PROVIDER: str = "llama.cpp2"  # Provider: "ollama", "llama.cpp", or "opencode" (free)
OPENCODE_MODEL: str = "GPT-OSS:120b"  # Model name within the provider

# Smart Edit Settings (Context-aware code generation)
PYTHON_CODER_SMART_EDIT = True  # Enable LLM-based smart editing (merges with existing .py files when beneficial)

# ============================================================================
# RAG Tool Settings
# ============================================================================
RAG_DOCUMENTS_DIR = Path("data/rag_documents")  # Document storage
RAG_INDEX_DIR = Path("data/rag_indices")  # FAISS indices storage
RAG_METADATA_DIR = Path("data/rag_metadata")  # Metadata storage

# Embedding Model Settings
# RECOMMENDED MODELS (ranked by accuracy, 2026):
#
# Multilingual (Korean + English + 100 languages):
# 1. "BAAI/bge-m3" - Best multilingual, cross-lingual retrieval (1024 dim)
# 2. "intfloat/multilingual-e5-large" - Strong multilingual (1024 dim)
#
# English-only:
# 3. "BAAI/bge-large-en-v1.5" - Best accuracy for English-only (1024 dim)
# 4. "BAAI/bge-base-en-v1.5" - Good English accuracy, faster (768 dim)
# 5. "sentence-transformers/all-MiniLM-L6-v2" - Lightweight, decent (384 dim)
#
# NOTE: Switching models requires rebuilding all FAISS indices (re-upload documents).
# Use HuggingFace model name (requires internet) or absolute path to local model directory.
RAG_EMBEDDING_MODEL = "/scratch0/LLM_models/offline_models/bge-m3"  # Multilingual: Korean + English cross-lingual retrieval
RAG_EMBEDDING_DEVICE = "cuda"  # "cpu" or "cuda"
RAG_EMBEDDING_BATCH_SIZE = 16

# FAISS Settings
RAG_INDEX_TYPE = "Flat"  # "Flat", "IVF", or "HNSW"
RAG_SIMILARITY_METRIC = "cosine"  # "cosine", "l2", or "ip" (inner product)

# Chunking Settings
RAG_CHUNK_SIZE = 512  # Characters per chunk (optimal: 200-500 for general docs)
RAG_CHUNK_OVERLAP = 50  # Overlap between chunks
RAG_CHUNKING_STRATEGY = "semantic"  # "fixed", "semantic" (best), "recursive", "sentence"
RAG_MAX_RESULTS = 100  # Maximum documents to retrieve
RAG_MIN_SCORE_THRESHOLD = 0.5  # Minimum relevance score (0.0-1.0) - chunks below this are discarded
                               # Applies to FAISS cosine, RRF, and sigmoid-normalized rerank scores
RAG_CONTEXT_WINDOW = 1  # Number of neighboring chunks to include around each match (0 = matched chunk only)

# Hybrid Search Settings (RECOMMENDED for 15-20% accuracy improvement)
RAG_USE_HYBRID_SEARCH = True  # Enable hybrid dense + sparse retrieval
RAG_HYBRID_ALPHA = 0.5  # Weight: 0.0=pure keyword, 1.0=pure semantic, 0.5=balanced

# Reranking Settings (RECOMMENDED for 15-20% additional accuracy improvement)
RAG_USE_RERANKING = True  # Enable two-stage retrieval with reranking
RAG_RERANKER_MODEL = "/scratch0/LLM_models/offline_models/mmarco-mMiniLMv2-L12-H384-v1"  # Multilingual cross-encoder for reranking
RAG_RERANKING_TOP_K = 500  # Retrieve more initially, then rerank to top-k

# Query Optimization
RAG_QUERY_PREFIX = ""  # bge-m3 uses built-in instruction handling; no manual prefix needed
RAG_USE_MULTI_QUERY = True  # Generate multiple query variants and merge results via RRF
RAG_MULTI_QUERY_COUNT = 6  # Number of bilingual query variants (3 angles x 2 languages)
RAG_QUERY_EXPANSION = False  # Expand query with synonyms/related terms (experimental)

RAG_DEFAULT_COLLECTION = "default"  # Default collection name

# Supported document formats
RAG_SUPPORTED_FORMATS = [".txt", ".pdf", ".docx", ".xlsx", ".xls", ".md", ".json", ".csv"]

# ============================================================================
# Session Settings
# ============================================================================
MAX_CONVERSATION_HISTORY = 50  # Maximum messages to keep in conversation history
SESSION_CLEANUP_DAYS = 7  # Delete sessions older than this

# ============================================================================
# Streaming Settings
# ============================================================================
STREAM_CHUNK_SIZE = 1  # How many tokens to send per SSE chunk
STREAM_TIMEOUT = 864000  # Streaming timeout in seconds (10 days)

# ============================================================================
# CORS Settings
# ============================================================================
CORS_ORIGINS = [
    "http://localhost:10007",
    "http://127.0.0.1:10007",
    "*",  # Allow all origins (change to specific IP if you want to restrict)
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
