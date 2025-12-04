"""
Llama.cpp Backend Wrapper
=========================
ChatOllama-compatible wrapper for llama-cpp-python.

Provides seamless switching between Ollama and llama.cpp backends without
modifying the rest of the codebase.

Features:
- Lazy loading (model loaded on first inference)
- Sync and async invoke/stream methods
- Message format conversion (LangChain → llama.cpp)
- Compatible with LLMInterceptor

Version: 1.0.0
Created: 2025-12-03
"""

from typing import Union, List, AsyncIterator, Iterator
from pathlib import Path
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
import asyncio
import logging

logger = logging.getLogger(__name__)


class LlamaCppWrapper:
    """
    Wrapper for llama-cpp-python that provides a ChatOllama-compatible interface.

    This allows seamless switching between Ollama and llama.cpp backends without
    modifying the rest of the codebase.

    Features:
    - Lazy loading (model loaded on first inference)
    - Sync and async invoke/stream methods
    - Message format conversion (LangChain → llama.cpp)
    - Compatible with LLMInterceptor
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 16384,
        n_gpu_layers: int = -1,
        n_threads: int = 8,
        n_batch: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 4096,
        rope_freq_base: float = 10000.0,
        rope_freq_scale: float = 1.0,
        use_mmap: bool = True,
        use_mlock: bool = False,
        low_vram: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize llama.cpp wrapper.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_gpu_layers: GPU layers to offload (-1 = all, 0 = CPU only)
            n_threads: CPU threads for computation
            n_batch: Batch size for prompt processing
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            rope_freq_base: RoPE frequency base
            rope_freq_scale: RoPE frequency scaling
            use_mmap: Use memory mapping for model loading
            use_mlock: Lock model in RAM
            low_vram: Reduce VRAM usage
            verbose: Enable verbose logging
        """
        self.model_path = model_path
        self.model_name = Path(model_path).stem

        # Store configuration for lazy loading
        self._config = {
            'model_path': model_path,
            'n_ctx': n_ctx,
            'n_gpu_layers': n_gpu_layers,
            'n_threads': n_threads,
            'n_batch': n_batch,
            'use_mmap': use_mmap,
            'use_mlock': use_mlock,
            'low_vram': low_vram,
            'verbose': verbose,
            'rope_freq_base': rope_freq_base,
            'rope_freq_scale': rope_freq_scale,
        }

        # Generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens

        # Lazy-loaded llama.cpp instance
        self._llm = None

        # Store model attribute for compatibility
        self.model = self.model_name

        logger.info(f"[LlamaCppWrapper] Initialized with model: {self.model_path}")

    def _load_model(self):
        """Lazy load the llama.cpp model."""
        if self._llm is not None:
            return

        try:
            from llama_cpp import Llama

            logger.info(f"[LlamaCppWrapper] Loading model from {self.model_path}...")

            # Check if model file exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    f"Please download a GGUF model and place it at this path."
                )

            self._llm = Llama(**self._config)

            logger.info(f"[LlamaCppWrapper] Model loaded successfully!")
            logger.info(f"[LlamaCppWrapper] Context size: {self._config['n_ctx']}")
            logger.info(f"[LlamaCppWrapper] GPU layers: {self._config['n_gpu_layers']}")

        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Please install it:\n"
                "pip install llama-cpp-python"
            )
        except Exception as e:
            logger.error(f"[LlamaCppWrapper] Failed to load model: {e}")
            raise

    def _convert_messages_to_prompt(self, messages: Union[str, List, BaseMessage]) -> str:
        """
        Convert LangChain messages to a prompt string for llama.cpp.

        Args:
            messages: String prompt or list of LangChain messages

        Returns:
            Formatted prompt string
        """
        if isinstance(messages, str):
            return messages

        # Convert list of messages to chat format
        if isinstance(messages, list):
            prompt_parts = []

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    prompt_parts.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    prompt_parts.append(f"Assistant: {msg.content}")
                elif isinstance(msg, SystemMessage):
                    prompt_parts.append(f"System: {msg.content}")
                elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = msg.type.capitalize()
                    prompt_parts.append(f"{role}: {msg.content}")
                else:
                    prompt_parts.append(str(msg))

            prompt_parts.append("Assistant:")
            return "\n\n".join(prompt_parts)

        # Single message
        if isinstance(messages, BaseMessage):
            return messages.content

        return str(messages)

    def invoke(self, prompt: Union[str, List, BaseMessage], **kwargs) -> AIMessage:
        """
        Synchronous inference (ChatOllama-compatible).

        Args:
            prompt: Input prompt (string or messages)
            **kwargs: Additional generation parameters

        Returns:
            AIMessage with generated response
        """
        self._load_model()

        # Convert to prompt string
        prompt_str = self._convert_messages_to_prompt(prompt)

        # Merge generation parameters
        gen_params = {
            'temperature': kwargs.get('temperature', self.temperature),
            'top_p': kwargs.get('top_p', self.top_p),
            'top_k': kwargs.get('top_k', self.top_k),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
        }

        logger.debug(f"[LlamaCppWrapper] Generating with temp={gen_params['temperature']}")

        # Generate response
        result = self._llm(prompt_str, **gen_params)

        # Extract text from result
        text = result['choices'][0]['text']

        return AIMessage(content=text)

    async def ainvoke(self, prompt: Union[str, List, BaseMessage], **kwargs) -> AIMessage:
        """
        Asynchronous inference (runs in thread pool).

        Args:
            prompt: Input prompt (string or messages)
            **kwargs: Additional generation parameters

        Returns:
            AIMessage with generated response
        """
        # Run sync invoke in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.invoke(prompt, **kwargs))

    def stream(self, prompt: Union[str, List, BaseMessage], **kwargs) -> Iterator[AIMessage]:
        """
        Synchronous streaming inference.

        Args:
            prompt: Input prompt (string or messages)
            **kwargs: Additional generation parameters

        Yields:
            AIMessage chunks with partial responses
        """
        self._load_model()

        # Convert to prompt string
        prompt_str = self._convert_messages_to_prompt(prompt)

        # Merge generation parameters
        gen_params = {
            'temperature': kwargs.get('temperature', self.temperature),
            'top_p': kwargs.get('top_p', self.top_p),
            'top_k': kwargs.get('top_k', self.top_k),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'stream': True,  # Enable streaming
        }

        # Stream tokens
        for chunk in self._llm(prompt_str, **gen_params):
            text = chunk['choices'][0]['text']
            yield AIMessage(content=text)

    async def astream(self, prompt: Union[str, List, BaseMessage], **kwargs) -> AsyncIterator[AIMessage]:
        """
        Asynchronous streaming inference.

        Args:
            prompt: Input prompt (string or messages)
            **kwargs: Additional generation parameters

        Yields:
            AIMessage chunks with partial responses
        """
        # Run sync stream in executor
        loop = asyncio.get_event_loop()

        for chunk in self.stream(prompt, **kwargs):
            yield chunk
            await asyncio.sleep(0)  # Allow event loop to process other tasks
