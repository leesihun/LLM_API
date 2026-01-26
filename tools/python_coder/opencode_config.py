"""
OpenCode Configuration Generator
Generates opencode config.json based on config.py settings
"""
import json
import sys
from pathlib import Path

import config


def get_opencode_config_path() -> Path:
    """Get platform-specific opencode config directory"""
    if sys.platform == "win32":
        return Path.home() / ".config" / "opencode" / "config.json"
    else:
        return Path.home() / ".config" / "opencode" / "config.json"


def generate_opencode_config() -> Path:
    """
    Generate opencode config.json based on config.py LLM backend settings

    Returns:
        Path to generated config file

    Raises:
        ValueError: If LLM_BACKEND is invalid
    """
    config_path = get_opencode_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    providers = {}

    # Add Ollama provider
    if config.LLM_BACKEND in ["ollama", "auto"]:
        ollama_base = config.OLLAMA_HOST.rstrip("/")
        if not ollama_base.endswith("/v1"):
            ollama_base = f"{ollama_base}/v1"

        providers["ollama"] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Ollama",
            "options": {
                "baseURL": ollama_base
            },
            "models": {
                config.OLLAMA_MODEL: {
                    "name": config.OLLAMA_MODEL
                }
            }
        }

    # Add llama.cpp provider
    if config.LLM_BACKEND in ["llamacpp", "auto"]:
        llamacpp_base = config.LLAMACPP_HOST.rstrip("/")
        if not llamacpp_base.endswith("/v1"):
            llamacpp_base = f"{llamacpp_base}/v1"

        providers["llama.cpp"] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "llama.cpp",
            "options": {
                "baseURL": llamacpp_base
            },
            "models": {
                config.LLAMACPP_MODEL: {
                    "name": config.LLAMACPP_MODEL
                }
            }
        }

    if not providers:
        raise ValueError(
            f"No providers configured. LLM_BACKEND={config.LLM_BACKEND} "
            f"must be 'ollama', 'llamacpp', or 'auto'"
        )

    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": providers
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(opencode_config, f, indent=2)

    print(f"[OPENCODE] Config generated: {config_path}")
    return config_path


def ensure_opencode_config() -> Path:
    """
    Ensure opencode config exists

    Always regenerates to stay in sync with config.py

    Returns:
        Path to config file
    """
    return generate_opencode_config()
