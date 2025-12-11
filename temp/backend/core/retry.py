"""
Retry Utility Module
===================
Retry logic with exponential backoff for resilient operations.

Provides decorators and functions for retrying operations that may fail:
- Exponential backoff
- Configurable max attempts
- Exception filtering
- Async support
- Detailed logging

Version: 2.0.0
Created: 2025-01-20
"""

import asyncio
import time
from typing import Optional, Callable, TypeVar, Any, Tuple, Type
from functools import wraps

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


async def retry_async(
    func: Callable[..., Any],
    *args,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    **kwargs
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: The async function to retry
        *args: Positional arguments for func
        max_attempts: Maximum number of attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        exceptions: Tuple of exceptions to catch (default: (Exception,))
        on_retry: Optional callback called on each retry with (exception, attempt)
        **kwargs: Keyword arguments for func

    Returns:
        Result from successful function call

    Raises:
        The last exception if all attempts fail

    Example:
        >>> async def fetch_data(url):
        ...     # May fail due to network issues
        ...     return await client.get(url)
        ...
        >>> result = await retry_async(
        ...     fetch_data,
        ...     "https://api.example.com/data",
        ...     max_attempts=5,
        ...     initial_delay=2.0,
        ...     exceptions=(httpx.RequestError, httpx.TimeoutException)
        ... )
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(f"Attempt {attempt}/{max_attempts} for {func.__name__}")
            result = await func(*args, **kwargs)

            if attempt > 1:
                logger.info(
                    f"{func.__name__} succeeded on attempt {attempt}/{max_attempts}"
                )

            return result

        except exceptions as e:
            last_exception = e

            if attempt == max_attempts:
                logger.error(
                    f"{func.__name__} failed after {max_attempts} attempts: {e}"
                )
                raise

            # Calculate delay with exponential backoff
            current_delay = min(delay, max_delay)

            logger.warning(
                f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                f"Retrying in {current_delay:.1f}s..."
            )

            # Call retry callback if provided
            if on_retry:
                try:
                    on_retry(e, attempt)
                except Exception as callback_error:
                    logger.error(f"Error in retry callback: {callback_error}")

            # Wait before retrying
            await asyncio.sleep(current_delay)

            # Increase delay for next attempt
            delay *= exponential_base

    # Should never reach here, but just in case
    raise last_exception


def retry_sync(
    func: Callable[..., Any],
    *args,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    **kwargs
) -> Any:
    """
    Retry a synchronous function with exponential backoff.

    Args:
        func: The synchronous function to retry
        *args: Positional arguments for func
        max_attempts: Maximum number of attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        exceptions: Tuple of exceptions to catch (default: (Exception,))
        on_retry: Optional callback called on each retry with (exception, attempt)
        **kwargs: Keyword arguments for func

    Returns:
        Result from successful function call

    Raises:
        The last exception if all attempts fail

    Example:
        >>> def save_to_database(data):
        ...     # May fail due to connection issues
        ...     db.insert(data)
        ...
        >>> retry_sync(
        ...     save_to_database,
        ...     my_data,
        ...     max_attempts=5,
        ...     exceptions=(DatabaseError,)
        ... )
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(f"Attempt {attempt}/{max_attempts} for {func.__name__}")
            result = func(*args, **kwargs)

            if attempt > 1:
                logger.info(
                    f"{func.__name__} succeeded on attempt {attempt}/{max_attempts}"
                )

            return result

        except exceptions as e:
            last_exception = e

            if attempt == max_attempts:
                logger.error(
                    f"{func.__name__} failed after {max_attempts} attempts: {e}"
                )
                raise

            # Calculate delay with exponential backoff
            current_delay = min(delay, max_delay)

            logger.warning(
                f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                f"Retrying in {current_delay:.1f}s..."
            )

            # Call retry callback if provided
            if on_retry:
                try:
                    on_retry(e, attempt)
                except Exception as callback_error:
                    logger.error(f"Error in retry callback: {callback_error}")

            # Wait before retrying
            time.sleep(current_delay)

            # Increase delay for next attempt
            delay *= exponential_base

    # Should never reach here, but just in case
    raise last_exception


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to catch
        on_retry: Optional callback for retry events

    Returns:
        Decorated function

    Example:
        >>> @with_retry(max_attempts=5, exceptions=(ValueError,))
        ... async def fetch_user(user_id):
        ...     # May fail, will retry automatically
        ...     return await api.get_user(user_id)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(
                func,
                *args,
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                exceptions=exceptions,
                on_retry=on_retry,
                **kwargs
            )
        return wrapper
    return decorator


def with_retry_sync(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying synchronous functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to catch
        on_retry: Optional callback for retry events

    Returns:
        Decorated function

    Example:
        >>> @with_retry_sync(max_attempts=3, exceptions=(IOError,))
        ... def save_file(path, content):
        ...     # May fail, will retry automatically
        ...     with open(path, 'w') as f:
        ...         f.write(content)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_sync(
                func,
                *args,
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                exceptions=exceptions,
                on_retry=on_retry,
                **kwargs
            )
        return wrapper
    return decorator


class RetryConfig:
    """
    Configuration for retry behavior.

    Provides a reusable configuration object for retry parameters.

    Example:
        >>> config = RetryConfig(
        ...     max_attempts=5,
        ...     initial_delay=2.0,
        ...     exceptions=(httpx.RequestError,)
        ... )
        ...
        >>> result = await retry_async(fetch_data, url, **config.to_dict())
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        """
        Initialize retry configuration.

        Args:
            max_attempts: Maximum number of attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            exceptions: Tuple of exceptions to catch
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.exceptions = exceptions

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary of retry parameters
        """
        return {
            "max_attempts": self.max_attempts,
            "initial_delay": self.initial_delay,
            "max_delay": self.max_delay,
            "exponential_base": self.exponential_base,
            "exceptions": self.exceptions
        }

    @classmethod
    def for_llm(cls) -> "RetryConfig":
        """
        Create retry configuration optimized for LLM calls.

        Returns:
            RetryConfig with LLM-appropriate settings
        """
        return cls(
            max_attempts=3,
            initial_delay=2.0,
            max_delay=30.0,
            exponential_base=2.0
        )

    @classmethod
    def for_network(cls) -> "RetryConfig":
        """
        Create retry configuration optimized for network calls.

        Returns:
            RetryConfig with network-appropriate settings
        """
        return cls(
            max_attempts=5,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0
        )

    @classmethod
    def for_file_operations(cls) -> "RetryConfig":
        """
        Create retry configuration optimized for file operations.

        Returns:
            RetryConfig with file operation-appropriate settings
        """
        return cls(
            max_attempts=3,
            initial_delay=0.5,
            max_delay=5.0,
            exponential_base=2.0
        )
