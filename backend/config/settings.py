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

    server_host: str = '0.0.0.0'
    server_port: int = 1007
    secret_key: str = 'dev-secret-key-change-in-production-please'

    # ============================================================================
    # Ollama Configuration - Optimized for Performance
    # ============================================================================

    # Ollama service endpoint
    ollama_host: str = 'http://127.0.0.1:11434'
    ollama_model: str = 'qwen3-coder:30b'#'gpt-oss:20b'
    agentic_classifier_model: str = 'qwen3-coder:30b'#'gpt-oss:20b'
    ollama_coder_model: str = 'qwen3-coder:30b'#'gpt-oss:20b'
    ollama_vision_model: str = 'gemma3:12b'
    ollama_timeout: int = 3000000
    ollama_num_ctx: int = 16384
    # Sampling parameters - optimized for coherent responses
    ollama_temperature: float = 0.6  # 0.1=conservative, 1.0=creative
    ollama_top_p: float = 0.95      # Nucleus sampling
    ollama_top_k: int = 20          # Top-k sampling (higher = more diverse)
    ollama_coder_model_temperature: float = 0.6

    react_max_iterations: int = 10  # Maximum iterations for ReAct loop
    react_step_max_retries: int = 5  # Maximum retries per step in plan execution
    python_code_max_iterations: int = 5

    # ============================================================================
    # Qwen Thinking Effort Configuration
    # ============================================================================

    # Thinking effort levels with severity-based instructions for Qwen models
    # - none: /no_think (instant responses, no reasoning)
    # - low: /think + brief reasoning guidance (1-2 steps)
    # - mid: /think + moderate reasoning guidance (2-4 steps, default)
    # - high: /think + deep reasoning guidance (4+ steps, thorough analysis)

    # Global default thinking effort
    thinking_effort_default: str = 'none'

    # Task-specific thinking effort settings
    thinking_effort_classifier: str = 'none'     # Fast classification
    thinking_effort_coder: str = 'high'          # Deep code reasoning
    thinking_effort_react: str = 'mid'           # Balanced agent reasoning
    thinking_effort_planner: str = 'high'        # Thorough planning
    thinking_effort_vision: str = 'low'          # Light visual analysis

    def get_thinking_prompt_prefix(self, effort: str) -> str:
        """
        Get Qwen thinking mode prompt prefix with severity-based instructions.

        Severity levels control the depth of reasoning:
        - none: No thinking (/no_think)
        - low: Quick reasoning (1-2 steps)
        - mid: Balanced reasoning (2-4 steps)
        - high: Deep reasoning (4+ steps with thorough analysis)

        Args:
            effort: Thinking effort level ('none', 'low', 'mid', 'high')

        Returns:
            Prompt prefix string with thinking control and severity guidance

        Examples:
            >>> settings.get_thinking_prompt_prefix('none')
            '/no_think'

            >>> settings.get_thinking_prompt_prefix('low')
            '/think Brief reasoning: consider 1-2 key steps, then respond concisely.'

            >>> settings.get_thinking_prompt_prefix('high')
            '/think Deep reasoning: analyze thoroughly with 4+ steps. Break down the problem, consider alternatives, verify your approach, then provide a comprehensive answer.'
        """
        prefixes = {
            'none': '/no_think',

            'low': '/think Brief reasoning: consider 1-2 key steps, then respond concisely.',

            'mid': '/think Moderate reasoning: think through 2-4 steps. Analyze the problem, plan your approach, then respond.',

            'high': '/think Deep reasoning: analyze thoroughly with 4+ steps. Break down the problem, consider alternatives, verify your approach, then provide a comprehensive answer.'
        }

        return prefixes.get(effort, prefixes['mid'])

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
    available_tools: list[str] = ['web_search', 'rag', 'python_coder', 'vision_analyzer']

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

    settings = Settings()
    # Debug: Print loaded settings
    print(f"DEBUG: ollama_host = '{settings.ollama_host}'")
    print(f"DEBUG: ollama_model = '{settings.ollama_model}'")
    print(f"DEBUG: server_host = '{settings.server_host}'")

    # FORCE FIX: Override if wrong value loaded
    if settings.ollama_host == '0.0.0.0' or not settings.ollama_host.startswith('http'):
        print("WARNING: ollama_host has wrong value, forcing to http://127.0.0.1:11434")
        settings.ollama_host = 'http://127.0.0.1:11434'

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
