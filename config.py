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
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# Concurrency Settings
# Number of worker processes for handling concurrent requests
# Workers are lightweight (~50MB each) since llama.cpp is on remote server
# Set to 1 to disable multi-processing (for debugging)
SERVER_WORKERS = 4  # Number of parallel workers for main API
TOOLS_SERVER_WORKERS = 4  # Number of parallel workers for tools API

# ============================================================================
# LLM Backend Settings
# ============================================================================
# Which LLM backend to use: "ollama", "llamacpp", or "auto" (tries ollama first, falls back to llamacpp)
LLM_BACKEND: Literal["ollama", "llamacpp", "auto"] = "llamacpp"

# Ollama Settings
# IMPORTANT FOR LINUX: If running on Linux and getting "Access denied" errors,
# you need to configure Ollama to accept network connections:
#   1. Set environment variable: export OLLAMA_HOST=0.0.0.0:11434
#   2. Then start Ollama: ollama serve
#   3. Or run: OLLAMA_HOST=0.0.0.0:11434 ollama serve
# For same-machine deployments, use http://127.0.0.1:11434
# For different machines, update to http://<ollama-server-ip>:11434
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"  # Default model (changed from non-existent GLM46)

# Model Preloading Settings
PRELOAD_MODEL_ON_STARTUP = False  # Preload default model to GPU on server startup
PRELOAD_KEEP_ALIVE = -1  # Keep model in memory: -1 = indefinitely, "5m" = 5 minutes, 0 = unload immediately

# Llama.cpp Settings
LLAMACPP_HOST = "http://localhost:5905"
LLAMACPP_MODEL = "default"  # Model loaded in llama.cpp server

# ============================================================================
# Model Parameters (Default LLM Inference Settings)
# ============================================================================
DEFAULT_TEMPERATURE = 1.0  # Randomness (0.0 = deterministic, 2.0 = very random)
DEFAULT_TOP_P = 0.9  # Nucleus sampling
DEFAULT_TOP_K = 40  # Top-k sampling
DEFAULT_MAX_TOKENS = 30000  # Maximum tokens in response

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
REACT_MAX_ITERATIONS = 5  # Maximum reasoning iterations
REACT_RETRY_ON_ERROR = True  # Let LLM see errors and decide next action (intelligent retry)

# Plan-Execute Agent Settings
PLAN_MAX_STEPS = 10  # Maximum plan steps
PLAN_MAX_ITERATIONS = 3  # Maximum iterations of plan-execute-replan cycle
PLAN_REPLAN_ON_FAILURE = True  # Re-plan when a step fails
PLAN_MIN_REACT_ITERATIONS_FOR_REPLAN = REACT_MAX_ITERATIONS  # Minimum React iterations before allowing step-level replanning
PLAN_SHARE_CONTEXT = True  # Share context across plan execution steps
PLAN_INCLUDE_FULL_HISTORY = True  # Include all conversation history in planning
PLAN_MAX_HISTORY_MESSAGES = 0  # Maximum history messages to include (0 = unlimited)
PLAN_HISTORY_IN_SYNTHESIS = True  # Include conversation history in final synthesis

# ============================================================================
# Tools Settings (for future implementation)
# ============================================================================
# Available tools - these will be implemented later
AVAILABLE_TOOLS = [
    "websearch",
    "python_coder",
    "rag",
    "ppt_maker",
]

# Tool-specific Model Configuration
# Different tools can use different models optimized for their tasks
TOOL_MODELS = {
    "websearch": "gpt-oss:20b",  # Web search summarization
    "python_coder": "gpt-oss:20b",  # Code generation
    "rag": "gpt-oss:20b",  # Document retrieval
    "ppt_maker": "gpt-oss:20b",  # Presentation markdown generation
}

# Tool-specific Model Parameters
# Each tool can have customized inference parameters
TOOL_PARAMETERS = {
    "websearch": {
        "temperature": 0.7,
        "max_tokens": 30000,
        "timeout": 360,  # 1 minute for web search (includes 2 LLM calls + Tavily API call)
    },
    "python_coder": {
        "temperature": 1.0,  # Lower temp for more deterministic code
        "max_tokens": 30000,
        "timeout": 1200,  # 2 minutes for code execution
    },
    "rag": {
        "temperature": 1.0,
        "max_tokens": 30000,
        "timeout": 120,  # 2 minutes for RAG retrieval
    },
    "ppt_maker": {
        "temperature": 0.7,
        "max_tokens": 30000,
        "timeout": 120,  # 2 minutes for generation + export
    },
}

# Default tool timeout (seconds)
DEFAULT_TOOL_TIMEOUT = 1200

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
# Execution mode: "native" or "nanocoder"
# - native: Direct Python code execution (fast, no LLM overhead)
# - nanocoder: Natural language to code using nanocoder CLI (autonomous coding)
PYTHON_EXECUTOR_MODE: Literal["native", "nanocoder"] = "native"

# Native executor settings
PYTHON_EXECUTOR_TIMEOUT = 1200  # Execution timeout in seconds
PYTHON_EXECUTOR_MAX_OUTPUT_SIZE = 1024 * 1024 * 10  # 10MB max output
PYTHON_WORKSPACE_DIR = SCRATCH_DIR  # Uses session scratch directory
PYTHON_CODER_TIMEOUT = 3000  # Timeout for ReAct agent tool call

# Nanocoder settings
NANOCODER_PATH = "nanocoder"  # Path to nanocoder binary (globally installed)
NANOCODER_CONFIG_DIR = Path(".nanocoder")  # Nanocoder config directory
NANOCODER_TIMEOUT = 1200  # Nanocoder execution timeout in seconds

# Smart Edit Settings (Context-aware code generation)
PYTHON_CODER_SMART_EDIT = True  # Enable LLM-based smart editing (merges with existing .py files when beneficial)

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
# PPT Maker Tool Settings
# ============================================================================
# Marp CLI path (use npx if not globally installed)
PPT_MAKER_MARP_CLI = "npx -y @marp-team/marp-cli"  # -y for auto-install

# Marp Theme Settings
PPT_MAKER_DEFAULT_THEME = "gaia"  # "default", "gaia", "uncover"
PPT_MAKER_THEMES = ["default", "gaia", "uncover"]  # Available themes

# Presentation Settings
PPT_MAKER_PAGINATE = True  # Show page numbers
PPT_MAKER_DEFAULT_FOOTER = ""  # Default footer text (empty = no footer)
PPT_MAKER_DEFAULT_HEADER = ""  # Default header text (empty = no header)

# Export Settings
PPT_MAKER_EXPORT_PDF = True  # Export to PDF
PPT_MAKER_EXPORT_PPTX = True  # Export to PPTX
PPT_MAKER_ALLOW_LOCAL_FILES = True  # Allow local file embedding

# Limits
PPT_MAKER_MAX_SLIDES = 100  # Maximum slides per presentation
PPT_MAKER_TIMEOUT = 120  # Marp CLI execution timeout (seconds)

# Workspace (uses session scratch directory)
PPT_MAKER_WORKSPACE_DIR = SCRATCH_DIR  # Same as python_coder

# LLM Settings for markdown generation
PPT_MAKER_MODEL = "gpt-oss:20b"  # Model for generating markdown
PPT_MAKER_TEMPERATURE = 0.7  # Temperature for generation
PPT_MAKER_MAX_TOKENS = 4096  # Max tokens for markdown generation

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
NANOCODER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
