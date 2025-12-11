"""
LLM Backend Modules
==================
Modular LLM backend implementations for flexible model deployment.

Supports:
- Ollama: LangChain ChatOllama integration
- llama.cpp: Direct GGUF model loading with ChatOllama-compatible wrapper

Version: 1.0.0
Created: 2025-12-03
"""

from backend.core.llm_backends.llamacpp_wrapper import LlamaCppWrapper
from backend.core.llm_backends.interceptor import LLMInterceptor, LogFormat, LogEntry, LogMessage

__all__ = [
    'LlamaCppWrapper',
    'LLMInterceptor',
    'LogFormat',
    'LogEntry',
    'LogMessage',
]
