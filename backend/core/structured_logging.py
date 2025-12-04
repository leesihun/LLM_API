"""
Structured Logging Module

Provides structured logging with context awareness, request tracing,
and performance metrics.

Features:
- JSON output for machine parsing
- Request/session ID tracking via context vars
- Performance metrics logging
- Error context capture
- Automatic PII sanitization

Version: 1.0.0
Created: 2025-12-03
"""

import logging
import json
from typing import Any, Dict, Optional
from contextvars import ContextVar
from datetime import datetime
import time
from functools import wraps
import re

# ============================================================================
# Context Variables for Request Tracing
# ============================================================================

request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


# ============================================================================
# PII Sanitization
# ============================================================================

class PIISanitizer:
    """Sanitize potentially sensitive information from logs."""

    # Patterns for common PII
    PATTERNS = {
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'ip': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        'api_key': re.compile(r'\b[A-Za-z0-9_-]{32,}\b'),
        'jwt': re.compile(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'),
    }

    @classmethod
    def sanitize(cls, text: str) -> str:
        """
        Sanitize PII from text.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text with PII replaced
        """
        if not isinstance(text, str):
            return text

        sanitized = text

        # Replace emails
        sanitized = cls.PATTERNS['email'].sub('[EMAIL]', sanitized)

        # Replace IPs (but keep localhost)
        sanitized = cls.PATTERNS['ip'].sub(
            lambda m: m.group(0) if m.group(0).startswith('127.') else '[IP]',
            sanitized
        )

        # Replace API keys and tokens
        sanitized = cls.PATTERNS['api_key'].sub('[API_KEY]', sanitized)
        sanitized = cls.PATTERNS['jwt'].sub('[JWT_TOKEN]', sanitized)

        return sanitized


# ============================================================================
# Structured Logger
# ============================================================================

class StructuredLogger:
    """
    Logger with structured output and context awareness.

    Features:
    - JSON output for machine parsing
    - Request/session/user ID tracking
    - Performance metrics
    - Error context capture
    - PII sanitization
    """

    def __init__(self, name: str, enable_json: bool = False, sanitize_pii: bool = True):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            enable_json: If True, output as JSON
            sanitize_pii: If True, sanitize PII from logs
        """
        self.logger = logging.getLogger(name)
        self.name = name
        self.enable_json = enable_json
        self.sanitize_pii = sanitize_pii

    def _build_context(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build log context with request/session/user IDs.

        Args:
            extra: Extra context fields

        Returns:
            Complete context dictionary
        """
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "logger": self.name,
            "request_id": request_id_var.get(),
            "session_id": session_id_var.get(),
            "user_id": user_id_var.get(),
        }

        if extra:
            # Sanitize PII if enabled
            if self.sanitize_pii:
                extra = self._sanitize_dict(extra)
            context.update(extra)

        return context

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary values."""
        sanitized = {}

        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = PIISanitizer.sanitize(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    PIISanitizer.sanitize(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    def _format_message(self, level: str, message: str, context: Dict[str, Any]) -> str:
        """
        Format log message.

        Args:
            level: Log level
            message: Log message
            context: Context dictionary

        Returns:
            Formatted log message
        """
        # Sanitize message if enabled
        if self.sanitize_pii:
            message = PIISanitizer.sanitize(message)

        if self.enable_json:
            return json.dumps({
                "level": level,
                "message": message,
                **context
            })
        else:
            # Human-readable format
            return message

    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        context = self._build_context(kwargs)
        formatted = self._format_message("DEBUG", message, context)
        self.logger.debug(formatted)

    def info(self, message: str, **kwargs):
        """Log info message with context."""
        context = self._build_context(kwargs)
        formatted = self._format_message("INFO", message, context)
        self.logger.info(formatted)

    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        context = self._build_context(kwargs)
        formatted = self._format_message("WARNING", message, context)
        self.logger.warning(formatted)

    def error(self, message: str, exc: Optional[Exception] = None, **kwargs):
        """
        Log error message with exception context.

        Args:
            message: Error message
            exc: Exception object
            **kwargs: Extra context
        """
        context = self._build_context(kwargs)

        if exc:
            context["exception"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "recoverable": getattr(exc, 'recoverable', False)
            }

            # Add exception context if available
            if hasattr(exc, 'context'):
                context["exception"]["context"] = exc.context

        formatted = self._format_message("ERROR", message, context)
        self.logger.error(formatted, exc_info=exc is not None)

    def critical(self, message: str, exc: Optional[Exception] = None, **kwargs):
        """Log critical message with exception context."""
        context = self._build_context(kwargs)

        if exc:
            context["exception"] = {
                "type": type(exc).__name__,
                "message": str(exc)
            }

        formatted = self._format_message("CRITICAL", message, context)
        self.logger.critical(formatted, exc_info=exc is not None)

    def perf(self, operation: str, duration_ms: float, **kwargs):
        """
        Log performance metric.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            **kwargs: Extra context
        """
        context = self._build_context(kwargs)
        context["performance"] = {
            "operation": operation,
            "duration_ms": round(duration_ms, 2)
        }

        formatted = self._format_message("INFO", f"Performance: {operation}", context)
        self.logger.info(formatted)


# ============================================================================
# Logger Factory
# ============================================================================

_logger_cache: Dict[str, StructuredLogger] = {}


def get_logger(
    name: str,
    enable_json: bool = False,
    sanitize_pii: bool = True
) -> StructuredLogger:
    """
    Get a structured logger instance (cached).

    Args:
        name: Logger name
        enable_json: If True, output as JSON
        sanitize_pii: If True, sanitize PII from logs

    Returns:
        StructuredLogger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("User logged in", user_id=123)
    """
    cache_key = f"{name}_{enable_json}_{sanitize_pii}"

    if cache_key not in _logger_cache:
        _logger_cache[cache_key] = StructuredLogger(name, enable_json, sanitize_pii)

    return _logger_cache[cache_key]


# ============================================================================
# Context Managers
# ============================================================================

class LogContext:
    """
    Context manager for setting request/session/user IDs.

    Example:
        >>> with LogContext(request_id="abc123", session_id="xyz789"):
        ...     logger.info("Processing request")  # IDs automatically included
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        self.request_id = request_id
        self.session_id = session_id
        self.user_id = user_id
        self._tokens = []

    def __enter__(self):
        if self.request_id:
            self._tokens.append(request_id_var.set(self.request_id))
        if self.session_id:
            self._tokens.append(session_id_var.set(self.session_id))
        if self.user_id:
            self._tokens.append(user_id_var.set(self.user_id))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for token in reversed(self._tokens):
            token.var.reset(token)


# ============================================================================
# Decorators
# ============================================================================

def log_execution_time(logger: Optional[StructuredLogger] = None):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance (optional, uses function module logger if not provided)

    Example:
        >>> @log_execution_time()
        ... def slow_function():
        ...     time.sleep(1)
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.perf(f"{func.__name__}", duration_ms)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{func.__name__} failed after {duration_ms:.2f}ms",
                    exc=e
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.perf(f"{func.__name__}", duration_ms)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{func.__name__} failed after {duration_ms:.2f}ms",
                    exc=e
                )
                raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
