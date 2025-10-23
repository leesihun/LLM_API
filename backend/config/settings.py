"""
Configuration Management System
All settings are defined in this file with defaults
.env file is OPTIONAL - can override settings if present
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
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
    server_port: int = 8000

    # SECRET KEY - CRITICAL: Generate a secure 32+ character key for production
    # Use: python -c "import secrets; print(secrets.token_urlsafe(32))"
    secret_key: str = 'dev-secret-key-change-in-production-please'

    # ============================================================================
    # Ollama Configuration - Optimized for Performance
    # ============================================================================

    # Ollama service endpoint
    ollama_host: str = Field(default='http://127.0.0.1:11434', env='OLLAMA_HOST')

    # Model selection - gpt-oss:20b
    ollama_model: str = Field(default='gpt-oss:20b', env='OLLAMA_MODEL')

    # Request timeout - 5 minutes for most requests, adjust based on model size
    ollama_timeout: int = 3000000  # 50 minutes

    # Context window - balance between capability and memory usage
    ollama_num_ctx: int = 4096  # Good balance for most use cases

    # Sampling parameters - optimized for coherent responses
    ollama_temperature: float = 0.7  # 0.1=conservative, 1.0=creative
    ollama_top_p: float = 0.9       # Nucleus sampling
    ollama_top_k: int = 40          # Top-k sampling

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

    model_config = SettingsConfigDict(
        env_file=".env" if os.path.exists(".env") else None,
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
        Path(settings.log_file).parent,
        Path(settings.users_path).parent,
        Path(settings.sessions_path).parent,
    ]

    for directory in directories_to_create:
        Path(directory).mkdir(parents=True, exist_ok=True)

    return settings


def create_env_file(env_path: str = ".env") -> None:
    """
    Create a .env file that mirrors the current Settings defaults.
    Pulling values from Settings ensures the template stays in sync.
    """
    defaults = Settings()
    env_content = f"""# ==============================================================================
# LLM API Configuration
# ==============================================================================
# Copy this file to .env and customize for your environment
# CRITICAL: Update sensitive values (API keys, secret keys) for production!

# ==============================================================================
# Server Configuration
# ==============================================================================
SERVER_HOST={defaults.server_host}
SERVER_PORT={defaults.server_port}
SECRET_KEY={defaults.secret_key}

# ==============================================================================
# Ollama Configuration
# ==============================================================================
OLLAMA_HOST={defaults.ollama_host}
OLLAMA_MODEL={defaults.ollama_model}
OLLAMA_TIMEOUT={defaults.ollama_timeout}
OLLAMA_NUM_CTX={defaults.ollama_num_ctx}
OLLAMA_TEMPERATURE={defaults.ollama_temperature}
OLLAMA_TOP_P={defaults.ollama_top_p}
OLLAMA_TOP_K={defaults.ollama_top_k}

# ==============================================================================
# API Keys (Get these from respective services)
# ==============================================================================
TAVILY_API_KEY={defaults.tavily_api_key}

# ==============================================================================
# Vector Database
# ==============================================================================
VECTOR_DB_TYPE={defaults.vector_db_type}
VECTOR_DB_PATH={defaults.vector_db_path}
EMBEDDING_MODEL={defaults.embedding_model}

# ==============================================================================
# Storage Paths
# ==============================================================================
USERS_PATH={defaults.users_path}
SESSIONS_PATH={defaults.sessions_path}
CONVERSATIONS_PATH={defaults.conversations_path}
UPLOADS_PATH={defaults.uploads_path}

# ==============================================================================
# Authentication
# ==============================================================================
JWT_ALGORITHM={defaults.jwt_algorithm}
JWT_EXPIRATION_HOURS={defaults.jwt_expiration_hours}

# ==============================================================================
# Logging
# ==============================================================================
LOG_LEVEL={defaults.log_level}
LOG_FILE={defaults.log_file}

# ==============================================================================
# Production Checklist
# ==============================================================================
# - Generate secure SECRET_KEY:
#   python -c "import secrets; print(secrets.token_urlsafe(32))"
# - Get Tavily API key from https://tavily.com/
# - Set SERVER_HOST=localhost for production security
# - Adjust OLLAMA_MODEL based on your hardware (7b, 13b, 70b)
# - Set LOG_LEVEL=WARNING for production environments
"""

    try:
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print(".env file created successfully!")
        print("Next steps:")
        print("   1. Edit .env file with your actual values")
        print("   2. Get API keys from respective services")
        print("   3. Run the application")
    except OSError as exc:
        raise ValueError(f"Failed to create .env file: {exc}") from exc


# Global settings instance
settings = load_settings()
