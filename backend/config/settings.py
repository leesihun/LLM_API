"""
Configuration Management System
Optimized settings with security and performance best practices
Loads from environment variables with secure defaults
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os
import secrets
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with optimized defaults
    All sensitive values should be overridden in production via .env file
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
    ollama_host: str = 'http://localhost:11434'

    # Model selection - gpt-oss:20b
    ollama_model: str = 'gpt-oss:20b'

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

    # Embedding model - all-MiniLM-L6-v2 is fast and good quality
    embedding_model: str = 'all-MiniLM-L6-v2'

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

    class Config:
        env_file = ".env"
        case_sensitive = False


def load_settings() -> Settings:
    """
    Load settings from environment variables with validation
    Creates .env file if missing and provides helpful setup instructions
    """
    if not os.path.exists(".env"):
        print("Warning: .env file not found!")
        print("Creating .env file with default settings...")
        print("Please edit .env file and update sensitive values for production!")
        create_env_file()

    try:
        settings = Settings()
    except Exception as e:
        raise ValueError(
            f"Configuration error: {e}\n"
            "Please check your .env file and ensure all required variables are set.\n"
            "Refer to .env.example for guidance."
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
