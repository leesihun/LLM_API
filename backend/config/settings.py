"""
Configuration Management System
All settings are defined in this file with defaults
.env file is OPTIONAL - can override settings if present
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os
import secrets
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with all defaults defined here
    .env file is optional - use it only if you need to override specific values
    """

    # ============================================================================
    # Server Configuration
    # ============================================================================

    # Host binding - use localhost in production, 0.0.0.0 for development
    server_host: str = '0.0.0.0'

    # Port - standard HTTP port, change if needed
    server_port: int = 1007

    # SECRET KEY - CRITICAL: Generate a secure 32+ character key for production
    # Use: python -c "import secrets; print(secrets.token_urlsafe(32))"
    secret_key: str = 'dev-secret-key-change-in-production-please'

    # ============================================================================
    # Ollama Configuration - Optimized for Performance
    # ============================================================================

    # Ollama service endpoint
    ollama_host: str = 'http://127.0.0.1:11434'

    # Model selection - gpt-oss:20b
    ollama_model: str = 'gemma3:12b'

    # Request timeout - 5 minutes for most requests, adjust based on model size
    ollama_timeout: int = 3000000  # 50 minutes

    # Context window - balance between capability and memory usage
    ollama_num_ctx: int = 16384  # Good balance for most use cases

    # Sampling parameters - optimized for coherent responses
    ollama_temperature: float = 1.0  # 0.1=conservative, 1.0=creative
    ollama_top_p: float = 0.95      # Nucleus sampling
    ollama_top_k: int = 64          # Top-k sampling

    # ============================================================================
    # API Keys - SECURE THESE IN PRODUCTION
    # ============================================================================

    # Tavily Search API - Get from https://tavily.com/
    tavily_api_key: str = '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'

    # ============================================================================
    # Vector Database - Optimized for Performance
    # ============================================================================

    # Vector DB type - 'faiss' for speed, 'chroma' for persistence
    vector_db_type: str = 'faiss'

    # Vector database storage path
    vector_db_path: str = './data/vector_db'

    # Embedding model - nomic-embed-text:latest is fast and good quality
    embedding_model: str = 'bge-m3:latest'

    # ============================================================================
    # Storage Paths - Organized Data Structure
    # ============================================================================

    # User data and authentication
    users_path: str = './data/users/users.json'

    # Session management
    sessions_path: str = './data/sessions/sessions.json'

    # Conversation history storage
    conversations_path: str = './data/conversations'

    # File uploads directory
    uploads_path: str = './data/uploads'

    # ============================================================================
    # Authentication - Security Optimized
    # ============================================================================

    # JWT algorithm - HS256 is secure and widely supported
    jwt_algorithm: str = 'HS256'

    # JWT expiration - 24 hours is reasonable for most applications
    jwt_expiration_hours: int = 24

    # ============================================================================
    # Logging - Comprehensive and Structured
    # ============================================================================

    # Log level - INFO for production, DEBUG for development
    log_level: str = 'INFO'

    # Log file location
    log_file: str = './data/logs/app.log'



    # AGENTIC FLOW

    # Agentic flow model
    agentic_classifier_model: str = 'gemma3:12b'

    # NOTE: Agentic classifier prompt is now centralized in config/prompts/task_classification.py
    # Use: from backend.config.prompts import get_agentic_classifier_prompt
    # This property provides backward compatibility
    @property
    def agentic_classifier_prompt(self) -> str:
        """Get agentic classifier prompt from centralized prompts module."""
        from backend.config.prompts import get_agentic_classifier_prompt
        return get_agentic_classifier_prompt()

    # Available tools
    available_tools: list[str] = ['web_search', 'rag', 'python_coder', 'chat']






    # ============================================================================
    # Python Code Execution - Sandbox Configuration
    # ============================================================================

    # Enable/disable Python code execution feature
    python_code_enabled: bool = True

    # Maximum execution time for generated code (seconds)
    python_code_timeout: int = 3000

    # Maximum memory usage (MB) - future: cgroups for true enforcement
    python_code_max_memory: int = 5120

    # Temporary execution directory for code execution
    python_code_execution_dir: str = './data/scratch'

    # Maximum verification-modification loop iterations
    python_code_max_iterations: int = 5

    # Execute code even if minor issues remain after max iterations
    python_code_allow_partial_execution: bool = True

    # Maximum input file size (MB) for code execution
    python_code_max_file_size: int = 500

    model_config = SettingsConfigDict(
        # DO NOT read from system environment variables - only use defaults from this file
        env_prefix='NONEXISTENT_PREFIX_',  # Ignore all env vars
        env_file=None,  # Don't load .env file
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
        validate_assignment=True
    )


def load_settings() -> Settings:
    """
    Load settings from settings.py defaults
    .env file is optional - only used if present to override defaults
    """
    # FORCE: Remove problematic environment variables that interfere with settings
    problematic_vars = ['OLLAMA_HOST', 'OLLAMA_MODEL', 'SERVER_HOST', 'SERVER_PORT']
    for var in problematic_vars:
        if var in os.environ:
            print(f"WARNING: Removing system env var {var}={os.environ[var]} to use settings.py defaults")
            del os.environ[var]

    # Check if .env exists (optional)
    if os.path.exists(".env"):
        print("Info: Using settings from settings.py with .env overrides")
    else:
        print("Info: Using settings from settings.py (no .env file found, which is OK)")

    try:
        settings = Settings()
        # Debug: Print loaded settings
        print(f"DEBUG: ollama_host = '{settings.ollama_host}'")
        print(f"DEBUG: ollama_model = '{settings.ollama_model}'")
        print(f"DEBUG: server_host = '{settings.server_host}'")

        # FORCE FIX: Override if wrong value loaded
        if settings.ollama_host == '0.0.0.0' or not settings.ollama_host.startswith('http'):
            print("WARNING: ollama_host has wrong value, forcing to http://127.0.0.1:11434")
            settings.ollama_host = 'http://127.0.0.1:11434'

    except Exception as e:
        raise ValueError(
            f"Configuration error: {e}\n"
            "Please check your backend/config/settings.py file."
        )

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
