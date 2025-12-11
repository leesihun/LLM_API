"""
Async Utilities Module

Provides utilities for async operations, batch processing, and concurrency control.

Features:
- Batch processing with concurrency limits
- CPU-bound operation handling
- Timeout utilities
- Retry with exponential backoff
- Rate limiting

Version: 1.0.0
Created: 2025-12-03
"""

import asyncio
from typing import List, TypeVar, Callable, Any, Optional, Awaitable
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from backend.core.structured_logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')


# ============================================================================
# Thread Pool for CPU-bound Operations
# ============================================================================

# Global thread pool executor
_cpu_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu-worker")

# Global process pool executor (for heavy CPU work)
_process_executor = ProcessPoolExecutor(max_workers=2)


async def run_cpu_bound(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run CPU-bound function in thread pool.

    Use this for operations that are CPU-intensive but need to run
    in the async context without blocking the event loop.

    Args:
        func: Synchronous function to run
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result

    Example:
        >>> def expensive_calculation(n):
        ...     return sum(i**2 for i in range(n))
        >>> result = await run_cpu_bound(expensive_calculation, 1000000)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_cpu_executor, func, *args, **kwargs)


async def run_process_bound(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run CPU-bound function in process pool.

    Use this for very heavy CPU operations that benefit from
    true parallelism (bypassing GIL).

    Args:
        func: Synchronous function to run
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result

    Note:
        Function must be picklable (top-level function, not lambda)

    Example:
        >>> def heavy_computation(data):
        ...     # Very CPU-intensive work
        ...     return result
        >>> result = await run_process_bound(heavy_computation, data)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_process_executor, func, *args, **kwargs)


# ============================================================================
# Batch Processing
# ============================================================================

class AsyncBatchProcessor:
    """
    Process items in batches with concurrency control.

    Features:
    - Concurrent processing with semaphore limit
    - Progress tracking
    - Error handling (continue or stop on error)
    - Results aggregation
    """

    def __init__(self, max_concurrent: int = 10, stop_on_error: bool = False):
        """
        Initialize batch processor.

        Args:
            max_concurrent: Maximum concurrent operations
            stop_on_error: If True, stop on first error
        """
        self.max_concurrent = max_concurrent
        self.stop_on_error = stop_on_error
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch(
        self,
        items: List[T],
        processor: Callable[[T], Awaitable[R]],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[R]:
        """
        Process items concurrently with limit.

        Args:
            items: Items to process
            processor: Async function to process each item
            on_progress: Optional progress callback (processed, total)

        Returns:
            List of results (in same order as items)

        Example:
            >>> async def process_item(item):
            ...     # Process item
            ...     return result
            >>> processor = AsyncBatchProcessor(max_concurrent=5)
            >>> results = await processor.process_batch(items, process_item)
        """
        total = len(items)
        processed = 0
        results = [None] * total  # Preserve order
        errors = []

        async def process_with_limit(index: int, item: T):
            nonlocal processed

            async with self.semaphore:
                try:
                    result = await processor(item)
                    results[index] = result
                    processed += 1

                    if on_progress:
                        on_progress(processed, total)

                    return result

                except Exception as e:
                    logger.error(f"Error processing item {index}: {e}", exc=e)
                    errors.append((index, item, e))

                    if self.stop_on_error:
                        raise

                    return None

        # Create tasks for all items
        tasks = [
            process_with_limit(i, item)
            for i, item in enumerate(items)
        ]

        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=not self.stop_on_error)

        if errors and self.stop_on_error:
            raise errors[0][2]  # Raise first error

        return results

    async def process_stream(
        self,
        items: List[T],
        processor: Callable[[T], Awaitable[R]]
    ):
        """
        Process items as stream (yield results as they complete).

        Args:
            items: Items to process
            processor: Async function to process each item

        Yields:
            Results as they complete (order not preserved)

        Example:
            >>> async for result in processor.process_stream(items, process_item):
            ...     print(result)
        """
        async def process_with_limit(item: T):
            async with self.semaphore:
                return await processor(item)

        tasks = [
            asyncio.create_task(process_with_limit(item))
            for item in items
        ]

        for coro in asyncio.as_completed(tasks):
            try:
                yield await coro
            except Exception as e:
                logger.error(f"Error processing item: {e}", exc=e)
                if self.stop_on_error:
                    raise


# ============================================================================
# Timeout Utilities
# ============================================================================

async def with_timeout(
    coro: Awaitable[T],
    timeout: float,
    default: Optional[T] = None
) -> T:
    """
    Run coroutine with timeout, returning default on timeout.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        default: Default value to return on timeout

    Returns:
        Coroutine result or default

    Example:
        >>> result = await with_timeout(slow_operation(), timeout=5.0, default=None)
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default


# ============================================================================
# Retry with Exponential Backoff
# ============================================================================

def retry_async(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retry with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Exponential backoff base
        exceptions: Tuple of exceptions to catch

    Example:
        >>> @retry_async(max_attempts=3, initial_delay=1.0)
        ... async def unstable_operation():
        ...     # May fail
        ...     return result
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            delay = initial_delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    if attempt == max_attempts - 1:
                        # Last attempt, re-raise
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts",
                            exc=e
                        )
                        raise

                    # Calculate backoff delay
                    backoff = min(delay * (exponential_base ** attempt), max_delay)

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}), "
                        f"retrying in {backoff:.2f}s",
                        error=str(e)
                    )

                    await asyncio.sleep(backoff)

            # Should never reach here
            raise RuntimeError(f"{func.__name__} exhausted all retry attempts")

        return wrapper

    return decorator


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for async operations.

    Features:
    - Token bucket algorithm
    - Configurable rate and burst
    - Async-aware
    """

    def __init__(self, rate: float, burst: int = 1):
        """
        Initialize rate limiter.

        Args:
            rate: Tokens per second
            burst: Maximum burst size (bucket capacity)
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1):
        """
        Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire
        """
        async with self.lock:
            while True:
                # Refill tokens
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.burst,
                    self.tokens + elapsed * self.rate
                )
                self.last_update = now

                # Try to acquire
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                # Wait for next token
                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)


def rate_limit(rate: float, burst: int = 1):
    """
    Decorator for rate limiting async functions.

    Args:
        rate: Calls per second
        burst: Maximum burst size

    Example:
        >>> @rate_limit(rate=10.0, burst=5)  # Max 10 calls/sec, burst of 5
        ... async def api_call():
        ...     return await fetch_data()
    """
    limiter = RateLimiter(rate, burst)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            await limiter.acquire()
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Debounce/Throttle
# ============================================================================

def debounce_async(wait: float):
    """
    Debounce async function - only execute after wait time of inactivity.

    Args:
        wait: Wait time in seconds

    Example:
        >>> @debounce_async(wait=1.0)
        ... async def search(query):
        ...     return await api_search(query)
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        task = None
        lock = asyncio.Lock()

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            nonlocal task

            async with lock:
                # Cancel previous task
                if task and not task.done():
                    task.cancel()

                # Create new task
                async def delayed():
                    await asyncio.sleep(wait)
                    return await func(*args, **kwargs)

                task = asyncio.create_task(delayed())
                return await task

        return wrapper

    return decorator


def throttle_async(rate: float):
    """
    Throttle async function - limit execution rate.

    Args:
        rate: Minimum time between calls (seconds)

    Example:
        >>> @throttle_async(rate=1.0)  # Max once per second
        ... async def update_ui():
        ...     await refresh_display()
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        last_call = 0.0
        lock = asyncio.Lock()

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            nonlocal last_call

            async with lock:
                now = time.monotonic()
                elapsed = now - last_call

                if elapsed < rate:
                    await asyncio.sleep(rate - elapsed)

                last_call = time.monotonic()
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Gather with Limits
# ============================================================================

async def gather_with_concurrency(
    n: int,
    *coros: Awaitable[T]
) -> List[T]:
    """
    Gather coroutines with concurrency limit.

    Args:
        n: Maximum concurrent coroutines
        *coros: Coroutines to execute

    Returns:
        List of results

    Example:
        >>> results = await gather_with_concurrency(
        ...     5,  # Max 5 concurrent
        ...     fetch(1), fetch(2), fetch(3), ...
        ... )
    """
    semaphore = asyncio.Semaphore(n)

    async def with_limit(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*[with_limit(c) for c in coros])
