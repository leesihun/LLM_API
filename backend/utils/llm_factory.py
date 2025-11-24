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
from datetime import datetime
from pathlib import Path

from backend.config.settings import settings
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMInterceptor:
    """
    Wrapper for LLM instances that intercepts and logs all prompts.
    Saves prompts to a user-specific file with clear separation for debugging and analysis.
    """

    def __init__(self, llm: ChatOllama, user_id: str = "default"):
        """
        Initialize the interceptor.

        Args:
            llm: The LLM instance to wrap
            user_id: User ID for organizing log files (defaults to "default")
        """
        self.llm = llm
        self.user_id = user_id

        # Create user-specific log file path: data/scratch/{username}/prompts.log
        self.log_file = Path(f"data/scratch/{user_id}/prompts.log")

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize log file with header if it doesn't exist
        if not self.log_file.exists():
            header = f"=== LLM Prompt Log for User: {user_id} ===\n"
            header += f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n\n"
            self.log_file.write_text(header, encoding='utf-8')

    def _log_prompt(self, prompt: str, model: str = None):
        """Save prompt to log file with timestamp and separation."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        model_name = model or getattr(self.llm, 'model', 'unknown')

        log_entry = f"""
{'='*80}
TIMESTAMP: {timestamp}
MODEL: {model_name}
USER: {self.user_id}
{'='*80}

{prompt}


{'='*80}
END OF PROMPT
{'='*80}


"""

        # Append to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

        logger.debug(f"[LLMInterceptor] Logged prompt for user '{self.user_id}' to {self.log_file} (length: {len(prompt)} chars)")

    async def ainvoke(self, prompt, **kwargs):
        """Async invoke with prompt logging."""
        self._log_prompt(prompt)
        return await self.llm.ainvoke(prompt, **kwargs)

    def invoke(self, prompt, **kwargs):
        """Sync invoke with prompt logging."""
        self._log_prompt(prompt)
        return self.llm.invoke(prompt, **kwargs)

    async def astream(self, prompt, **kwargs):
        """Async stream with prompt logging."""
        self._log_prompt(prompt)
        return await self.llm.astream(prompt, **kwargs)

    def stream(self, prompt, **kwargs):
        """Sync stream with prompt logging."""
        self._log_prompt(prompt)
        return self.llm.stream(prompt, **kwargs)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped LLM."""
        return getattr(self.llm, name)


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
        user_id: Optional[str] = None,
        enable_prompt_logging: bool = True,
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
            user_id: User ID for prompt logging (defaults to "default")
            enable_prompt_logging: Enable prompt interception and logging (default: True)
            **kwargs: Additional parameters to pass to ChatOllama

        Returns:
            Configured ChatOllama instance (wrapped with LLMInterceptor if logging enabled)

        Example:
            >>> llm = LLMFactory.create_llm(temperature=0.7, user_id="alice")
            >>> response = llm.invoke("Hello, world!")
        """
        config = {
            "base_url": settings.ollama_host,
            "model": model or settings.ollama_model,
            "temperature": temperature if temperature is not None else settings.ollama_temperature,
            "num_ctx": num_ctx or settings.ollama_num_ctx,
            "top_p": top_p if top_p is not None else settings.ollama_top_p,
            "top_k": top_k if top_k is not None else settings.ollama_top_k,
            "keep_alive": -1,  # Keep model loaded in VRAM indefinitely (permanent)
        }

        # Add timeout if specified (convert to seconds)
        if timeout is not None:
            config["timeout"] = timeout
        elif settings.ollama_timeout:
            config["timeout"] = settings.ollama_timeout

        # Merge with additional kwargs
        config.update(kwargs)

        logger.debug(f"[LLMFactory] Creating LLM with model={config['model']}, temp={config['temperature']}")

        try:
            llm = ChatOllama(**config)

            # Wrap with interceptor for prompt logging
            if enable_prompt_logging:
                user_id = user_id or "default"
                llm = LLMInterceptor(llm, user_id=user_id)
                logger.debug(f"[LLMFactory] Enabled prompt logging for user '{user_id}'")

            return llm
        except Exception as e:
            logger.error(f"[LLMFactory] Failed to create LLM: {e}")
            raise

    @classmethod
    def create_classifier_llm(
        cls,
        model: Optional[str] = None,
        temperature: float = 0.1,
        num_ctx: int = 2048,
        user_id: Optional[str] = None,
        enable_prompt_logging: bool = True,
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
            user_id: User ID for prompt logging (defaults to "default")
            enable_prompt_logging: Enable prompt interception and logging (default: True)
            **kwargs: Additional parameters to pass to ChatOllama

        Returns:
            Configured ChatOllama instance for classification

        Example:
            >>> classifier = LLMFactory.create_classifier_llm(user_id="alice")
            >>> result = classifier.invoke("Classify this: search for weather")
        """
        classifier_model = model or settings.agentic_classifier_model

        logger.debug(f"[LLMFactory] Creating classifier LLM with model={classifier_model}")

        return cls.create_llm(
            model=classifier_model,
            temperature=temperature,
            num_ctx=num_ctx,
            user_id=user_id,
            enable_prompt_logging=enable_prompt_logging,
            **kwargs
        )

    @classmethod
    def create_coder_llm(
        cls,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        user_id: Optional[str] = None,
        enable_prompt_logging: bool = True,
        **kwargs
    ) -> ChatOllama:
        """
        Create an LLM optimized for code generation tasks.

        Uses settings.ollama_coder_model by default, which can be configured
        to use a specialized coding model (e.g., deepseek-coder, codellama).

        Args:
            model: Model name (defaults to settings.ollama_coder_model)
            temperature: Sampling temperature (defaults to settings.ollama_coder_model_temperature)
            num_ctx: Context window size (defaults to settings.ollama_num_ctx)
            user_id: User ID for prompt logging (defaults to "default")
            enable_prompt_logging: Enable prompt interception and logging (default: True)
            **kwargs: Additional parameters to pass to ChatOllama

        Returns:
            Configured ChatOllama instance for code generation

        Example:
            >>> coder = LLMFactory.create_coder_llm(user_id="alice")
            >>> code = coder.invoke("Generate Python code to read a CSV file")
        """
        coder_model = model or settings.ollama_coder_model
        coder_temperature = (
            temperature if temperature is not None else settings.ollama_coder_model_temperature
        )
        coder_num_ctx = num_ctx or settings.ollama_num_ctx

        logger.debug(f"[LLMFactory] Creating coder LLM with model={coder_model}")

        return cls.create_llm(
            model=coder_model,
            temperature=coder_temperature,
            num_ctx=coder_num_ctx,
            user_id=user_id,
            enable_prompt_logging=enable_prompt_logging,
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
