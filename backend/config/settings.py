"""
Configuration Management System
All settings are defined in this file with defaults
.env file is OPTIONAL - can override settings if present
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with all defaults defined here
    .env file is optional - use it only if you need to override specific values
    """

    # ============================================================================
    # Server Configuration
    # ============================================================================

    server_host: str = '0.0.0.0'
    server_port: int = 1007
    secret_key: str = 'dev-secret-key-change-in-production-please'

    # ============================================================================
    # LLM Backend Configuration
    # ============================================================================

    # Backend selection: 'ollama' or 'llamacpp'
    llm_backend: str = 'ollama'  # Default to Ollama for backward compatibility

    # ============================================================================
    # Ollama Configuration - Optimized for Performance
    # ============================================================================

    # Ollama service endpoint
    ollama_host: str = 'http://127.0.0.1:11434'
    ollama_model: str = 'qwen3-coder:30b'  # Available: gpt-oss:20b (13GB), deepseek-r1:1.5b (1.1GB), llama3.2-vision:11b (7.8GB)
    agentic_classifier_model: str = 'qwen3-coder:30b'  # Use lightweight model for classification
    ollama_coder_model: str = 'qwen3-coder:30b'  # Use lightweight model for code generation
    ollama_vision_model: str = 'llama3.2-vision:11b'  # Vision model for image analysis
    ollama_timeout: int = 3000000
    ollama_num_ctx: int = 2048  # Optimized for faster processing (reduced from 4096)
    # Sampling parameters - optimized for coherent responses
    ollama_temperature: float = 0.6  # 0.1=conservative, 1.0=creative
    ollama_top_p: float = 0.95      # Nucleus sampling
    ollama_top_k: int = 20          # Top-k sampling (higher = more diverse)
    ollama_coder_model_temperature: float = 1.0

    # ============================================================================
    # Llama.cpp Configuration
    # ============================================================================

    # Model file paths (GGUF format)
    llamacpp_model_path: str = '../models/gpt-oss-120b.gguf'
    llamacpp_classifier_model_path: str = '../models/gpt-oss-20b.gguf'
    llamacpp_coder_model_path: str = '../models/GLM-4.6-REAP.gguf'
    llamacpp_vision_model_path: str = '../models/gemma3-12b-it-q8_0.gguf'

    # Hardware acceleration
    llamacpp_n_gpu_layers: int = -1  # -1 = offload all layers to GPU, 0 = CPU only
    llamacpp_n_threads: int = 8      # CPU threads for computation
    llamacpp_n_batch: int = 512      # Batch size for prompt processing

    # Context and generation settings
    llamacpp_n_ctx: int = 2048      # Context window size (optimized for faster processing)
    llamacpp_temperature: float = 1.0
    llamacpp_top_p: float = 0.95
    llamacpp_top_k: int = 40
    llamacpp_max_tokens: int = 4096  # Max tokens to generate
    llamacpp_rope_freq_base: float = 10000.0  # RoPE frequency base (adjust for extended context)
    llamacpp_rope_freq_scale: float = 1.0     # RoPE frequency scaling

    # Memory optimization
    llamacpp_use_mmap: bool = True   # Use memory mapping for model loading
    llamacpp_use_mlock: bool = False # Lock model in RAM (requires privileges)
    llamacpp_low_vram: bool = False  # Reduce VRAM usage (slower)

    # Verbose logging for debugging
    llamacpp_verbose: bool = True

    # ============================================================================
    # ReAct Agent Configuration - Optimized Iteration Limits
    # ============================================================================

    react_max_iterations: int = 10  # Maximum iterations for ReAct loop (optimized from 10)
    react_step_max_retries: int = 5  # Maximum retries per step in plan execution (optimized from 5)
    python_code_max_iterations: int = 5  # Single-shot code execution; iteration handled by agent
    
    # Tool calling mode: 'react' (text-based prompting) or 'native' (Ollama/OpenAI function calling)
    # - 'react': Works with all backends (Ollama, llama.cpp), uses THOUGHT/ACTION/OBSERVATION format
    # - 'native': Uses structured JSON tool calling, best with Ollama (llama.cpp support is limited)
    tool_calling_mode: str = 'react'

    # ============================================================================
    # Vision/Multimodal Configuration
    # ============================================================================

    # Vision model for image understanding tasks
    vision_enabled: bool = False
    vision_max_image_size: int = 2048  # Max dimension (width/height) for image resizing
    vision_temperature: float = 0.3  # Lower temp for more focused vision analysis

    # ============================================================================
    # API Keys - SECURE THESE IN PRODUCTION
    # ============================================================================

    tavily_api_key: str = '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'

    # ============================================================================
    # Vector Database - Optimized for Performance
    # ============================================================================

    vector_db_type: str = 'faiss'
    vector_db_path: str = './data/vector_db'
    embedding_model: str = 'bge-m3:latest'

    # ============================================================================
    # Storage Paths - Organized Data Structure
    # ============================================================================

    users_path: str = './data/users/users.json'
    sessions_path: str = './data/sessions/sessions.json'
    conversations_path: str = './data/conversations'
    uploads_path: str = './data/uploads'

    # ============================================================================
    # Authentication - Security Optimized
    # ============================================================================

    # JWT algorithm - HS256 is secure and widely supported
    jwt_algorithm: str = 'HS256'
    jwt_expiration_hours: int = 2400

    # ============================================================================
    # Logging - Comprehensive and Structured
    # ============================================================================

    log_level: str = 'INFO'
    log_file: str = './data/logs/app.log'

    # ============================================================================
    # AGENTIC FLOW - Model Selection
    # ============================================================================


    # ============================================================================
    # ReAct Agent Configuration
    # ============================================================================

    # Available tools
    available_tools: list[str] = ['web_search', 'rag', 'python_coder', 'vision_analyzer', 'shell']

    # ============================================================================
    # Python Code Execution - Sandbox Configuration
    # ============================================================================

    python_code_enabled: bool = True
    python_code_timeout: int = 3000
    python_code_max_memory: int = 5120
    python_code_execution_dir: str = './data/scratch'
    python_code_allow_partial_execution: bool = False
    python_code_max_file_size: int = 500
    python_code_use_persistent_repl: bool = True
    python_code_output_max_llm_chars: int = 8000000  # Max chars from result files to send to LLM

    # Pre-import libraries in REPL for faster execution (reduces 20-30s import time to <1s)
    # Libraries are imported once when REPL starts, then reused across all executions
    python_code_preload_libraries: list[str] = [
        'pandas as pd',
        'numpy as np',
        'matplotlib.pyplot as plt',
        'matplotlib',  # For matplotlib.use('Agg') configuration
    ]

    # ============================================================================
    # Shell Tool - Safe command execution for navigation and inspection
    # ============================================================================
    shell_tool_enabled: bool = True
    shell_tool_timeout: int = 30  # seconds
    shell_windows_mode: bool = True  # If True on Windows, map to native commands (dir/cd/type/findstr)


    model_config = SettingsConfigDict(
        # Allow environment variables to override defaults (12-factor app compliant)
        env_file='.env',  # Optional .env file
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
        validate_assignment=True
    )


def load_settings() -> Settings:
    """
    Load settings from settings.py defaults.
    Environment variables can override defaults if needed (12-factor app compliant).
    """
    settings = Settings()
    
    # Create necessary directories
    directories_to_create = [
        settings.conversations_path,
        settings.uploads_path,
        settings.vector_db_path,
        settings.python_code_execution_dir,
        Path(settings.log_file).parent,
        Path(settings.users_path).parent,
        Path(settings.sessions_path).parent,
    ]

    for directory in directories_to_create:
        Path(directory).mkdir(parents=True, exist_ok=True)

    return settings


# Global settings instance
settings = load_settings()
