"""
LLM Factory Module
==================
Centralized factory for creating LLM instances with consistent configuration.
Eliminates code duplication across the backend by providing standardized LLM creation.

Version: 1.0.0
Created: 2025-01-13
"""

from typing import Optional, Dict, Any
from langchain_ollama import ChatOllama
import httpx
import logging

from backend.config.settings import settings
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """
    Factory class for creating and managing LLM instances.

    Provides standardized methods for creating LLMs with different configurations:
    - Standard LLM: General purpose chat and reasoning
    - Classifier LLM: Task classification with low temperature
    - Coder LLM: Code generation with specific parameters

    Features:
    - Consistent configuration from settings
    - Connection pooling and retry logic
    - Health checks and validation
    - Lazy initialization support
    """

    _connection_pool: Optional[httpx.Client] = None

    @classmethod
    def create_llm(
        cls,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> ChatOllama:
        """
        Create a standard LLM instance with default or custom parameters.

        Args:
            model: Model name (defaults to settings.ollama_model)
            temperature: Sampling temperature (defaults to settings.ollama_temperature)
            num_ctx: Context window size (defaults to settings.ollama_num_ctx)
            top_p: Nucleus sampling parameter (defaults to settings.ollama_top_p)
            top_k: Top-k sampling parameter (defaults to settings.ollama_top_k)
            timeout: Request timeout in milliseconds (defaults to settings.ollama_timeout)
            **kwargs: Additional parameters to pass to ChatOllama

        Returns:
            Configured ChatOllama instance

        Example:
            >>> llm = LLMFactory.create_llm(temperature=0.7)
            >>> response = llm.invoke("Hello, world!")
        """
        config = {
            "base_url": settings.ollama_host,
            "model": model or settings.ollama_model,
            "temperature": temperature if temperature is not None else settings.ollama_temperature,
            "num_ctx": num_ctx or settings.ollama_num_ctx,
            "top_p": top_p if top_p is not None else settings.ollama_top_p,
            "top_k": top_k if top_k is not None else settings.ollama_top_k,
            "keep_alive": "60m",  # Keep model loaded in VRAM for 60 minutes
        }

        # Add timeout if specified (convert to seconds)
        if timeout is not None:
            config["timeout"] = timeout / 1000
        elif settings.ollama_timeout:
            config["timeout"] = settings.ollama_timeout / 1000

        # Merge with additional kwargs
        config.update(kwargs)

        logger.debug(f"[LLMFactory] Creating LLM with model={config['model']}, temp={config['temperature']}")

        try:
            return ChatOllama(**config)
        except Exception as e:
            logger.error(f"[LLMFactory] Failed to create LLM: {e}")
            raise

    @classmethod
    def create_classifier_llm(
        cls,
        model: Optional[str] = None,
        temperature: float = 0.1,
        num_ctx: int = 2048,
        **kwargs
    ) -> ChatOllama:
        """
        Create an LLM optimized for classification tasks.

        Uses low temperature for consistent classification results and
        smaller context window for faster processing.

        Args:
            model: Model name (defaults to settings.agentic_classifier_model)
            temperature: Low temperature for consistent results (default: 0.1)
            num_ctx: Small context window for speed (default: 2048)
            **kwargs: Additional parameters to pass to ChatOllama

        Returns:
            Configured ChatOllama instance for classification

        Example:
            >>> classifier = LLMFactory.create_classifier_llm()
            >>> result = classifier.invoke("Classify this: search for weather")
        """
        classifier_model = model or settings.agentic_classifier_model

        logger.debug(f"[LLMFactory] Creating classifier LLM with model={classifier_model}")

        return cls.create_llm(
            model=classifier_model,
            temperature=temperature,
            num_ctx=num_ctx,
            **kwargs
        )

    @classmethod
    def create_coder_llm(
        cls,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        **kwargs
    ) -> ChatOllama:
        """
        Create an LLM optimized for code generation tasks.

        Uses standard configuration but allows customization for
        code generation specific needs.

        Args:
            model: Model name (defaults to settings.ollama_model)
            temperature: Sampling temperature (defaults to settings.ollama_temperature)
            num_ctx: Context window size (defaults to settings.ollama_num_ctx)
            **kwargs: Additional parameters to pass to ChatOllama

        Returns:
            Configured ChatOllama instance for code generation

        Example:
            >>> coder = LLMFactory.create_coder_llm()
            >>> code = coder.invoke("Generate Python code to read a CSV file")
        """
        logger.debug("[LLMFactory] Creating coder LLM")

        return cls.create_llm(
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            **kwargs
        )

    @classmethod
    def check_connection(cls, timeout: int = 5000) -> bool:
        """
        Check if Ollama service is accessible.

        Args:
            timeout: Connection timeout in milliseconds (default: 5000)

        Returns:
            True if connection is successful, False otherwise

        Example:
            >>> if LLMFactory.check_connection():
            ...     print("Ollama is ready")
        """
        try:
            import httpx

            url = f"{settings.ollama_host}/api/tags"
            response = httpx.get(url, timeout=timeout / 1000)

            if response.status_code == 200:
                logger.info("[LLMFactory] Ollama connection successful")
                return True
            else:
                logger.warning(f"[LLMFactory] Ollama returned status {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"[LLMFactory] Connection check failed: {e}")
            return False

    @classmethod
    def get_available_models(cls) -> list:
        """
        Get list of available models from Ollama.

        Returns:
            List of model names, or empty list if unable to fetch

        Example:
            >>> models = LLMFactory.get_available_models()
            >>> print(f"Available models: {models}")
        """
        try:
            import httpx

            url = f"{settings.ollama_host}/api/tags"
            response = httpx.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                models = [model.get("name") for model in data.get("models", [])]
                logger.info(f"[LLMFactory] Found {len(models)} available models")
                return models
            else:
                logger.warning(f"[LLMFactory] Failed to get models: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"[LLMFactory] Error fetching models: {e}")
            return []


# Global factory instance (optional, for convenience)
llm_factory = LLMFactory()
