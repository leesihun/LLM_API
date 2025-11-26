"""
LLM Factory Module
==================
Centralized factory for creating LLM instances with consistent configuration.
Eliminates code duplication across the backend by providing standardized LLM creation.

Version: 1.1.0
Created: 2025-01-13
Updated: 2025-11-25 - Improved LLMInterceptor with structured logging
"""

from typing import Optional, Dict, Any, List, Literal
from langchain_ollama import ChatOllama
import httpx
import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from backend.config.settings import settings
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LogFormat(Enum):
    """Output format for LLM logs."""
    STRUCTURED = "structured"  # Human-readable structured format
    JSON = "json"              # JSON Lines format for parsing
    COMPACT = "compact"        # Minimal format for quick scanning


@dataclass
class LogMessage:
    """Represents a single message in the conversation."""
    role: str
    content: str
    
    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LogEntry:
    """Structured log entry for LLM interactions."""
    call_id: str
    timestamp: str
    entry_type: str  # "REQUEST" or "RESPONSE"
    model: str
    user_id: str
    messages: List[LogMessage]
    token_estimate: int
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "timestamp": self.timestamp,
            "type": self.entry_type,
            "model": self.model,
            "user": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
            "token_estimate": self.token_estimate,
            "duration_ms": self.duration_ms
        }


class LLMInterceptor:
    """
    Wrapper for LLM instances that intercepts and logs all prompts/responses.
    
    Features:
    - Structured, easy-to-read log format
    - Request/Response pairing with unique call IDs
    - Multiple output formats (structured, JSON, compact)
    - Token estimation for tracking usage
    - Duration tracking for performance analysis
    - Clear visual hierarchy with message role separation
    
    Log Format Options:
    - STRUCTURED: Human-readable with clear visual sections
    - JSON: JSON Lines format for programmatic parsing
    - COMPACT: Minimal format for quick scanning
    """

    def __init__(
        self, 
        llm: ChatOllama, 
        user_id: str = "default",
        log_format: LogFormat = LogFormat.STRUCTURED,
        log_file: Optional[Path] = None
    ):
        """
        Initialize the interceptor.

        Args:
            llm: The LLM instance to wrap
            user_id: User ID for organizing log files (defaults to "default")
            log_format: Output format for logs (default: STRUCTURED)
            log_file: Custom log file path (defaults to data/scratch/prompts.log)
        """
        self.llm = llm
        self.user_id = user_id
        self.log_format = log_format
        self._current_call_id: Optional[str] = None
        self._call_start_time: Optional[datetime] = None

        # Create log file path
        self.log_file = log_file or Path("data/scratch/prompts.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize log file with header if it doesn't exist
        if not self.log_file.exists():
            self._write_header()

    def _write_header(self):
        """Write log file header based on format."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if self.log_format == LogFormat.JSON:
            header = json.dumps({
                "log_type": "llm_prompt_log",
                "created": timestamp,
                "user_id": self.user_id,
                "format_version": "1.0"
            }) + "\n"
        else:
            header = f"""┌{'─'*78}┐
│{'LLM PROMPT LOG'.center(78)}│
├{'─'*78}┤
│  User: {self.user_id:<69}│
│  Created: {timestamp:<66}│
│  Format: {self.log_format.value:<67}│
└{'─'*78}┘

"""
        self.log_file.write_text(header, encoding='utf-8')

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (approx 4 chars per token)."""
        return len(text) // 4

    def _parse_messages(self, prompt) -> List[LogMessage]:
        """Parse prompt into structured messages."""
        messages = []
        
        if isinstance(prompt, str):
            messages.append(LogMessage(role="USER", content=prompt))
        elif isinstance(prompt, list):
            for msg in prompt:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = msg.type.upper()
                    messages.append(LogMessage(role=role, content=msg.content))
                else:
                    messages.append(LogMessage(role="UNKNOWN", content=str(msg)))
        else:
            messages.append(LogMessage(role="RAW", content=str(prompt)))
            
        return messages

    def _format_structured(self, entry: LogEntry) -> str:
        """Format entry in human-readable structured format."""
        lines = []
        
        # Header box
        type_icon = "📤" if entry.entry_type == "REQUEST" else "📥"
        type_color = "REQUEST " if entry.entry_type == "REQUEST" else "RESPONSE"
        
        lines.append(f"\n{'═'*80}")
        lines.append(f"  {type_icon} {type_color}  │  ID: {entry.call_id[:8]}  │  {entry.timestamp}")
        lines.append(f"{'─'*80}")
        lines.append(f"  Model: {entry.model:<30}  User: {entry.user_id}")

        # Build the tokens/duration line
        tokens_line = f"  Tokens: ~{entry.token_estimate:<25}"
        if entry.duration_ms:
            tokens_line += f"  Duration: {entry.duration_ms:.0f}ms"
        lines.append(tokens_line)

        lines.append(f"{'─'*80}")
        
        # Messages section
        for msg in entry.messages:
            role_label = f"[{msg.role}]"
            lines.append(f"\n  {role_label}")
            lines.append(f"  {'·'*40}")
            
            # Indent content lines
            content_lines = msg.content.split('\n')
            for line in content_lines:
                # Wrap long lines
                if len(line) > 74:
                    wrapped = [line[i:i+74] for i in range(0, len(line), 74)]
                    for w in wrapped:
                        lines.append(f"    {w}")
                else:
                    lines.append(f"    {line}")
        
        lines.append(f"\n{'═'*80}\n")
        
        return '\n'.join(lines)

    def _format_json(self, entry: LogEntry) -> str:
        """Format entry as JSON line."""
        return json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"

    def _format_compact(self, entry: LogEntry) -> str:
        """Format entry in compact format."""
        type_marker = ">>>" if entry.entry_type == "REQUEST" else "<<<"
        content_preview = entry.messages[0].content[:100] if entry.messages else ""
        content_preview = content_preview.replace('\n', ' ')
        if len(entry.messages[0].content if entry.messages else "") > 100:
            content_preview += "..."
            
        return f"[{entry.timestamp}] {type_marker} {entry.call_id[:8]} | {entry.model} | {content_preview}\n"

    def _format_entry(self, entry: LogEntry) -> str:
        """Format entry based on configured format."""
        if self.log_format == LogFormat.JSON:
            return self._format_json(entry)
        elif self.log_format == LogFormat.COMPACT:
            return self._format_compact(entry)
        else:
            return self._format_structured(entry)

    def _log_request(self, prompt, model: str = None) -> str:
        """Log a request and return the call ID."""
        call_id = str(uuid.uuid4())
        self._current_call_id = call_id
        self._call_start_time = datetime.now()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        model_name = model or getattr(self.llm, 'model', 'unknown')
        messages = self._parse_messages(prompt)
        
        total_content = ' '.join(m.content for m in messages)
        token_estimate = self._estimate_tokens(total_content)
        
        entry = LogEntry(
            call_id=call_id,
            timestamp=timestamp,
            entry_type="REQUEST",
            model=model_name,
            user_id=self.user_id,
            messages=messages,
            token_estimate=token_estimate
        )
        
        formatted = self._format_entry(entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted)
        
        logger.debug(f"[LLMInterceptor] Logged request {call_id[:8]} for user '{self.user_id}'")
        
        return call_id

    def _log_response(self, response, model: str = None):
        """Log a response with timing information."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        model_name = model or getattr(self.llm, 'model', 'unknown')
        
        # Calculate duration
        duration_ms = None
        if self._call_start_time:
            duration_ms = (datetime.now() - self._call_start_time).total_seconds() * 1000
        
        # Extract response content
        if hasattr(response, 'content'):
            response_content = response.content
        else:
            response_content = str(response)
        
        messages = [LogMessage(role="ASSISTANT", content=response_content)]
        token_estimate = self._estimate_tokens(response_content)
        
        entry = LogEntry(
            call_id=self._current_call_id or str(uuid.uuid4()),
            timestamp=timestamp,
            entry_type="RESPONSE",
            model=model_name,
            user_id=self.user_id,
            messages=messages,
            token_estimate=token_estimate,
            duration_ms=duration_ms
        )
        
        formatted = self._format_entry(entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted)
        
        logger.debug(f"[LLMInterceptor] Logged response for call {self._current_call_id[:8] if self._current_call_id else 'unknown'}")
        
        # Reset call tracking
        self._current_call_id = None
        self._call_start_time = None

    async def ainvoke(self, prompt, **kwargs):
        """Async invoke with prompt and response logging."""
        self._log_request(prompt)
        response = await self.llm.ainvoke(prompt, **kwargs)
        self._log_response(response)
        return response

    def invoke(self, prompt, **kwargs):
        """Sync invoke with prompt and response logging."""
        self._log_request(prompt)
        response = self.llm.invoke(prompt, **kwargs)
        self._log_response(response)
        return response

    async def astream(self, prompt, **kwargs):
        """Async stream with prompt logging (response logged on completion)."""
        call_id = self._log_request(prompt)
        
        async def stream_with_logging():
            chunks = []
            async for chunk in self.llm.astream(prompt, **kwargs):
                chunks.append(chunk)
                yield chunk
            # Log aggregated response after streaming completes
            if chunks:
                full_content = ''.join(
                    c.content if hasattr(c, 'content') else str(c) 
                    for c in chunks
                )
                # Create a mock response object for logging
                class MockResponse:
                    content = full_content
                self._current_call_id = call_id
                self._log_response(MockResponse())
        
        return stream_with_logging()

    def stream(self, prompt, **kwargs):
        """Sync stream with prompt logging (response logged on completion)."""
        call_id = self._log_request(prompt)
        
        def stream_with_logging():
            chunks = []
            for chunk in self.llm.stream(prompt, **kwargs):
                chunks.append(chunk)
                yield chunk
            # Log aggregated response after streaming completes
            if chunks:
                full_content = ''.join(
                    c.content if hasattr(c, 'content') else str(c) 
                    for c in chunks
                )
                class MockResponse:
                    content = full_content
                self._current_call_id = call_id
                self._log_response(MockResponse())
        
        return stream_with_logging()

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
        log_format: LogFormat = LogFormat.STRUCTURED,
        log_file: Optional[Path] = None,
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
            log_format: Format for prompt logs - STRUCTURED, JSON, or COMPACT (default: STRUCTURED)
            log_file: Custom log file path (defaults to data/scratch/prompts.log)
            **kwargs: Additional parameters to pass to ChatOllama

        Returns:
            Configured ChatOllama instance (wrapped with LLMInterceptor if logging enabled)

        Example:
            >>> llm = LLMFactory.create_llm(temperature=0.7, user_id="alice")
            >>> response = llm.invoke("Hello, world!")
            
            >>> # With JSON logging for parsing
            >>> llm = LLMFactory.create_llm(log_format=LogFormat.JSON)
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
                llm = LLMInterceptor(
                    llm, 
                    user_id=user_id, 
                    log_format=log_format,
                    log_file=log_file
                )
                logger.debug(f"[LLMFactory] Enabled prompt logging for user '{user_id}' (format: {log_format.value})")

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
