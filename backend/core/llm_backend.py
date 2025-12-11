"""
LLM Backend abstraction for Ollama and llama.cpp
Provides unified interface for both backends with auto-fallback
"""
from typing import Iterator, List, Dict, Optional
import httpx
from abc import ABC, abstractmethod

import config


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> str:
        """Non-streaming chat completion"""
        pass

    @abstractmethod
    def chat_stream(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> Iterator[str]:
        """Streaming chat completion"""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass


class OllamaBackend(LLMBackend):
    """Ollama backend implementation"""

    def __init__(self, host: str = None):
        self.host = (host or config.OLLAMA_HOST).rstrip("/")

    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception:
            return []

    def chat(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> str:
        """Non-streaming chat completion"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        response = httpx.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=config.STREAM_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    def chat_stream(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> Iterator[str]:
        """Streaming chat completion"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }

        with httpx.stream(
            "POST",
            f"{self.host}/api/chat",
            json=payload,
            timeout=config.STREAM_TIMEOUT
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    import json
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue


class LlamaCppBackend(LLMBackend):
    """llama.cpp backend implementation (OpenAI-compatible API)"""

    def __init__(self, host: str = None):
        self.host = (host or config.LLAMACPP_HOST).rstrip("/")

    def is_available(self) -> bool:
        """Check if llama.cpp server is running"""
        try:
            response = httpx.get(f"{self.host}/v1/models", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models (llama.cpp typically has one loaded model)"""
        try:
            response = httpx.get(f"{self.host}/v1/models", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception:
            return []

    def chat(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> str:
        """Non-streaming chat completion (OpenAI-compatible)"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }

        response = httpx.post(
            f"{self.host}/v1/chat/completions",
            json=payload,
            timeout=config.STREAM_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def chat_stream(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> Iterator[str]:
        """Streaming chat completion (OpenAI-compatible)"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }

        with httpx.stream(
            "POST",
            f"{self.host}/v1/chat/completions",
            json=payload,
            timeout=config.STREAM_TIMEOUT
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    import json
                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue


class AutoLLMBackend(LLMBackend):
    """
    Auto-selecting backend that tries Ollama first, falls back to llama.cpp
    """

    def __init__(self):
        self.ollama = OllamaBackend()
        self.llamacpp = LlamaCppBackend()
        self._active_backend: Optional[LLMBackend] = None

    def _get_backend(self) -> LLMBackend:
        """Get the active backend, detecting availability if needed"""
        if self._active_backend is None:
            if self.ollama.is_available():
                self._active_backend = self.ollama
                print("Using Ollama backend")
            elif self.llamacpp.is_available():
                self._active_backend = self.llamacpp
                print("Using llama.cpp backend")
            else:
                raise RuntimeError("No LLM backend available (tried Ollama and llama.cpp)")

        return self._active_backend

    def is_available(self) -> bool:
        """Check if any backend is available"""
        return self.ollama.is_available() or self.llamacpp.is_available()

    def list_models(self) -> List[str]:
        """List models from active backend"""
        backend = self._get_backend()
        return backend.list_models()

    def chat(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> str:
        """Non-streaming chat using active backend"""
        backend = self._get_backend()
        return backend.chat(messages, model, temperature)

    def chat_stream(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> Iterator[str]:
        """Streaming chat using active backend"""
        backend = self._get_backend()
        return backend.chat_stream(messages, model, temperature)


# Factory function to get the configured backend
def get_llm_backend() -> LLMBackend:
    """Get LLM backend based on config"""
    if config.LLM_BACKEND == "ollama":
        return OllamaBackend()
    elif config.LLM_BACKEND == "llamacpp":
        return LlamaCppBackend()
    elif config.LLM_BACKEND == "auto":
        return AutoLLMBackend()
    else:
        raise ValueError(f"Unknown LLM backend: {config.LLM_BACKEND}")


# Global backend instance with interceptor
from backend.core.llm_interceptor import LLMInterceptor

_backend = get_llm_backend()
llm_backend = LLMInterceptor(_backend)
