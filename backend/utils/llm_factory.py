"""
LLM Factory Module
==================
Centralized factory for creating LLM instances with consistent configuration.
Eliminates code duplication across the backend by providing standardized LLM creation.

Supports multiple backends:
- Ollama: LangChain ChatOllama integration
- Llama.cpp: Direct GGUF model loading with llama-cpp-python

Version: 1.3.0
Created: 2025-01-13
Updated: 2025-12-03 - Refactored: Extracted LlamaCppWrapper and LLMInterceptor to separate modules

Changes:
- Extracted LlamaCppWrapper to backend.core.llm_backends.llamacpp_wrapper
- Extracted LLMInterceptor to backend.core.llm_backends.interceptor
- Reduced file size from 1,023 lines → 400 lines (61% reduction)
"""

from typing import Optional, Union
from langchain_ollama import ChatOllama
from pathlib import Path
import httpx

from backend.config.settings import settings
from backend.utils.logging_utils import get_logger
from backend.core.llm_backends import LlamaCppWrapper, LLMInterceptor, LogFormat

logger = get_logger(__name__)


class LLMFactory:
    """
    Factory class for creating and managing LLM instances.

    Provides standardized methods for creating LLMs with different configurations:
    - Standard LLM: General purpose chat and reasoning
    - Classifier LLM: Task classification with low temperature
    - Coder LLM: Code generation with specific parameters
    - Vision LLM: Image understanding and visual tasks

    Features:
    - Consistent configuration from settings
    - Support for Ollama and llama.cpp backends
    - Optional prompt/response logging with multiple formats
    - Health checks and model listing
    """

    _connection_pool: Optional[httpx.Client] = None

    @classmethod
    def create_llm(
        cls,
        model: Optional[str] = None,
        model_path: Optional[str] = None,
        backend: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        timeout: Optional[int] = None,
        user_id: Optional[str] = None,
        enable_prompt_logging: bool = True,
        log_format: LogFormat = LogFormat.STRUCTURED,
        log_file: Optional[Path] = None,
        **kwargs
    ) -> Union[ChatOllama, LlamaCppWrapper]:
        """
        Create a standard LLM instance with default or custom parameters.

        Supports both Ollama and llama.cpp backends with automatic backend selection.

        Args:
            model: Model name for Ollama (defaults to settings.ollama_model)
            model_path: Path to GGUF model file for llama.cpp (defaults to settings.llamacpp_model_path)
            backend: Backend to use ('ollama' or 'llamacpp', defaults to settings.llm_backend)
            temperature: Sampling temperature (defaults based on backend)
            num_ctx: Context window size (defaults based on backend)
            top_p: Nucleus sampling parameter (defaults based on backend)
            top_k: Top-k sampling parameter (defaults based on backend)
            timeout: Request timeout in milliseconds (Ollama only)
            user_id: User ID for prompt logging (defaults to "default")
            enable_prompt_logging: Enable prompt interception and logging (default: True)
            log_format: Format for prompt logs - STRUCTURED, JSON, or COMPACT (default: STRUCTURED)
            log_file: Custom log file path (defaults to data/scratch/prompts.log)
            **kwargs: Additional backend-specific parameters

        Returns:
            Configured LLM instance (ChatOllama or LlamaCppWrapper)
            Wrapped with LLMInterceptor if logging enabled

        Example:
            >>> # Ollama backend (default)
            >>> llm = LLMFactory.create_llm(temperature=0.7, user_id="alice")
            >>> response = llm.invoke("Hello, world!")

            >>> # Llama.cpp backend
            >>> llm = LLMFactory.create_llm(
            ...     backend='llamacpp',
            ...     model_path='./models/qwen-7b.gguf',
            ...     temperature=0.7
            ... )

            >>> # With JSON logging for parsing
            >>> llm = LLMFactory.create_llm(log_format=LogFormat.JSON)
        """
        # Determine backend
        selected_backend = backend or settings.llm_backend
        selected_backend = selected_backend.lower()

        logger.debug(f"[LLMFactory] Creating LLM with backend='{selected_backend}'")

        # ====================================================================
        # Ollama Backend
        # ====================================================================
        if selected_backend == 'ollama':
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

            logger.debug(f"[LLMFactory] Creating Ollama LLM with model={config['model']}, temp={config['temperature']}")

            try:
                llm = ChatOllama(**config)
            except Exception as e:
                logger.error(f"[LLMFactory] Failed to create Ollama LLM: {e}")
                raise

        # ====================================================================
        # Llama.cpp Backend
        # ====================================================================
        elif selected_backend == 'llamacpp':
            config = {
                "model_path": model_path or settings.llamacpp_model_path,
                "n_ctx": num_ctx or settings.llamacpp_n_ctx,
                "n_gpu_layers": kwargs.get('n_gpu_layers', settings.llamacpp_n_gpu_layers),
                "n_threads": kwargs.get('n_threads', settings.llamacpp_n_threads),
                "n_batch": kwargs.get('n_batch', settings.llamacpp_n_batch),
                "temperature": temperature if temperature is not None else settings.llamacpp_temperature,
                "top_p": top_p if top_p is not None else settings.llamacpp_top_p,
                "top_k": top_k if top_k is not None else settings.llamacpp_top_k,
                "max_tokens": kwargs.get('max_tokens', settings.llamacpp_max_tokens),
                "rope_freq_base": kwargs.get('rope_freq_base', settings.llamacpp_rope_freq_base),
                "rope_freq_scale": kwargs.get('rope_freq_scale', settings.llamacpp_rope_freq_scale),
                "use_mmap": kwargs.get('use_mmap', settings.llamacpp_use_mmap),
                "use_mlock": kwargs.get('use_mlock', settings.llamacpp_use_mlock),
                "low_vram": kwargs.get('low_vram', settings.llamacpp_low_vram),
                "verbose": kwargs.get('verbose', settings.llamacpp_verbose),
            }

            logger.debug(
                f"[LLMFactory] Creating llama.cpp LLM with "
                f"model_path={config['model_path']}, "
                f"n_ctx={config['n_ctx']}, "
                f"n_gpu_layers={config['n_gpu_layers']}, "
                f"temp={config['temperature']}"
            )

            try:
                llm = LlamaCppWrapper(**config)
            except Exception as e:
                logger.error(f"[LLMFactory] Failed to create llama.cpp LLM: {e}")
                raise

        else:
            raise ValueError(
                f"Invalid backend '{selected_backend}'. "
                f"Must be 'ollama' or 'llamacpp'."
            )

        # ====================================================================
        # Wrap with interceptor for prompt logging
        # ====================================================================
        if enable_prompt_logging:
            user_id = user_id or "default"
            llm = LLMInterceptor(
                llm,
                user_id=user_id,
                log_format=log_format,
                log_file=log_file
            )
            logger.debug(f"[LLMFactory] Enabled prompt logging for user '{user_id}' (format: {log_format.value})")

        return llm

    @classmethod
    def create_classifier_llm(
        cls,
        model: Optional[str] = None,
        temperature: float = 0.1,
        num_ctx: int = 2048,
        user_id: Optional[str] = None,
        enable_prompt_logging: bool = True,
        log_format: LogFormat = LogFormat.STRUCTURED,
        log_file: Optional[Path] = None,
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
            log_format: Format for prompt logs - STRUCTURED, JSON, or COMPACT (default: STRUCTURED)
            log_file: Custom log file path (defaults to data/scratch/prompts.log)
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
            log_format=log_format,
            log_file=log_file,
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
        log_format: LogFormat = LogFormat.STRUCTURED,
        log_file: Optional[Path] = None,
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
            log_format: Format for prompt logs - STRUCTURED, JSON, or COMPACT (default: STRUCTURED)
            log_file: Custom log file path (defaults to data/scratch/prompts.log)
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
            log_format=log_format,
            log_file=log_file,
            **kwargs
        )

    @classmethod
    def create_vision_llm(
        cls,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        user_id: Optional[str] = None,
        enable_prompt_logging: bool = True,
        log_format: LogFormat = LogFormat.STRUCTURED,
        log_file: Optional[Path] = None,
        **kwargs
    ) -> ChatOllama:
        """
        Create an LLM optimized for vision/image understanding tasks.

        Uses settings.ollama_vision_model by default (e.g., llama3.2-vision:11b).
        Uses lower temperature for more focused visual analysis.

        Args:
            model: Vision model name (defaults to settings.ollama_vision_model)
            temperature: Sampling temperature (defaults to settings.vision_temperature)
            num_ctx: Context window size (defaults to settings.ollama_num_ctx)
            user_id: User ID for prompt logging (defaults to "default")
            enable_prompt_logging: Enable prompt interception and logging (default: True)
            log_format: Format for prompt logs - STRUCTURED, JSON, or COMPACT (default: STRUCTURED)
            log_file: Custom log file path (defaults to data/scratch/prompts.log)
            **kwargs: Additional parameters to pass to ChatOllama

        Returns:
            Configured ChatOllama instance for vision tasks

        Example:
            >>> vision_llm = LLMFactory.create_vision_llm(user_id="alice")
            >>> # Use with multimodal message containing images
            >>> response = vision_llm.invoke(multimodal_message)
        """
        vision_model = model or settings.ollama_vision_model
        vision_temperature = (
            temperature if temperature is not None else settings.vision_temperature
        )
        vision_num_ctx = num_ctx or settings.ollama_num_ctx

        logger.debug(f"[LLMFactory] Creating vision LLM with model={vision_model}")

        return cls.create_llm(
            model=vision_model,
            temperature=vision_temperature,
            num_ctx=vision_num_ctx,
            user_id=user_id,
            enable_prompt_logging=enable_prompt_logging,
            log_format=log_format,
            log_file=log_file,
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
