"""
Configuration Management System
Loads settings from environment variables with NO fallbacks
All required settings must be explicitly configured
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings - all must be configured in .env file"""

    # Server Configuration
    server_host: '0.0.0.0'
    server_port: 8000
    secret_key: 'change-this-secret-key-in-production'

    # Ollama Configuration
    ollama_host: 'http://localhost:11434'
    ollama_model: 'gpt-oss:20b'
    ollama_timeout: 600000
    ollama_num_ctx: 8192
    ollama_temperature: 0.1
    ollama_top_p: 0.9
    ollama_top_k: 40

    # Tavily Search API
    tavily_api_key: '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'

    # Vector Database
    vector_db_type: str
    vector_db_path: str
    embedding_model: str

    # Storage Paths
    users_path: str
    sessions_path: str
    conversations_path: str
    uploads_path: str

    # Authentication
    jwt_algorithm: str
    jwt_expiration_hours: int

    # Logging
    log_level: str
    log_file: str

    class Config:
        env_file = ".env"
        case_sensitive = False


def load_settings() -> Settings:
    """
    Load settings from environment variables
    Raises error if required variables are missing
    """
    if not os.path.exists(".env"):
        raise FileNotFoundError(
            ".env file not found. Copy .env.example to .env and configure all required variables"
        )

    try:
        settings = Settings()
    except Exception as e:
        raise ValueError(
            f"Configuration error: {e}\n"
            "Please ensure all required environment variables are set in .env file"
        )

    # Create necessary directories
    Path(settings.conversations_path).mkdir(parents=True, exist_ok=True)
    Path(settings.uploads_path).mkdir(parents=True, exist_ok=True)
    Path(settings.vector_db_path).mkdir(parents=True, exist_ok=True)
    Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.users_path).parent.mkdir(parents=True, exist_ok=True)

    return settings


# Global settings instance
settings = load_settings()
