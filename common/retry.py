"""
Retry Mechanism with Exponential Backoff

Provides configurable retry logic with exponential backoff, jitter,
and various retry strategies for resilient service communication.
"""

import asyncio
import time
import logging
import random
import math
from typing import Any, Callable, Optional, Union, Type, Tuple, List
from dataclasses import dataclass
from enum import Enum
import functools

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


class JitterType(Enum):
    """Jitter types for retry delays."""
    NONE = "none"
    FULL = "full"
    EQUAL = "equal"
    DECORRELATED = "decorrelated"


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: JitterType = JitterType.FULL
    backoff_multiplier: float = 2.0
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()
    timeout: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.backoff_multiplier <= 1.0:
            raise ValueError("backoff_multiplier must be > 1.0")


@dataclass
class RetryStats:
    """Retry attempt statistics."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_delay: float = 0.0
    last_attempt_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    exceptions: List[Exception] = None
    
    def __post_init__(self):
        if self.exceptions is None:
            self.exceptions = []


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, attempts: int, last_exception: Exception, stats: RetryStats):
        self.attempts = attempts
        self.last_exception = last_exception
        self.stats = stats
        super().__init__(
            f"Retry exhausted after {attempts} attempts. "
            f"Last exception: {last_exception}"
        )


class RetryDelayCalculator:
    """Calculate retry delays based on strategy and jitter."""
    
    @staticmethod
    def calculate_delay(
        attempt: int,
        config: RetryConfig,
        previous_delay: Optional[float] = None
    ) -> float:
        """
        Calculate delay for retry attempt.
        
        Args:
            attempt: Current attempt number (0-based)
            config: Retry configuration
            previous_delay: Previous delay (for decorrelated jitter)
            
        Returns:
            Delay in seconds
        """
        # Calculate base delay based on strategy
        if config.strategy == RetryStrategy.FIXED_DELAY:
            base_delay = config.base_delay
        
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            base_delay = config.base_delay * (config.backoff_multiplier ** attempt)
        
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            base_delay = config.base_delay * (1 + attempt)
        
        elif config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            base_delay = config.base_delay * RetryDelayCalculator._fibonacci(attempt + 1)
        
        else:
            base_delay = config.base_delay
        
        # Apply jitter
        if config.jitter == JitterType.NONE:
            delay = base_delay
        
        elif config.jitter == JitterType.FULL:
            # Random delay between 0 and base_delay
            delay = random.uniform(0, base_delay)
        
        elif config.jitter == JitterType.EQUAL:
            # Random delay between base_delay/2 and base_delay
            delay = base_delay / 2 + random.uniform(0, base_delay / 2)
        
        elif config.jitter == JitterType.DECORRELATED:
            # Decorrelated jitter based on previous delay
            if previous_delay is None:
                delay = random.uniform(0, config.base_delay)
            else:
                delay = random.uniform(config.base_delay, previous_delay * 3)
        
        else:
            delay = base_delay
        
        # Ensure delay is within bounds
        return min(max(delay, 0), config.max_delay)
    
    @staticmethod
    def _fibonacci(n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


class Retry:
    """Retry mechanism with configurable strategies."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.stats = RetryStats()
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.async_call(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)
            return sync_wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RetryExhaustedError: When all retry attempts fail
        """
        last_exception = None
        previous_delay = None
        
        for attempt in range(self.config.max_attempts):
            self.stats.total_attempts += 1
            self.stats.last_attempt_time = time.time()
            
            try:
                # Execute function with timeout if configured
                if self.config.timeout:
                    result = self._call_with_timeout(func, self.config.timeout, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success
                self.stats.successful_attempts += 1
                self.stats.last_success_time = time.time()
                
                logger.debug(
                    f"Retry successful on attempt {attempt + 1}/{self.config.max_attempts}"
                )
                
                return result
                
            except Exception as e:
                last_exception = e
                self.stats.failed_attempts += 1
                self.stats.last_failure_time = time.time()
                self.stats.exceptions.append(e)
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    logger.debug(f"Non-retryable exception: {e}")
                    raise
                
                # Check if this was the last attempt
                if attempt == self.config.max_attempts - 1:
                    break
                
                # Calculate and apply delay
                delay = RetryDelayCalculator.calculate_delay(
                    attempt, self.config, previous_delay
                )
                previous_delay = delay
                self.stats.total_delay += delay
                
                logger.debug(
                    f"Retry attempt {attempt + 1}/{self.config.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                time.sleep(delay)
        
        # All attempts failed
        raise RetryExhaustedError(
            self.config.max_attempts, last_exception, self.stats
        )
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function with retry logic.
        
        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RetryExhaustedError: When all retry attempts fail
        """
        last_exception = None
        previous_delay = None
        
        for attempt in range(self.config.max_attempts):
            self.stats.total_attempts += 1
            self.stats.last_attempt_time = time.time()
            
            try:
                # Execute function with timeout if configured
                if self.config.timeout:
                    result = await self._async_call_with_timeout(
                        func, self.config.timeout, *args, **kwargs
                    )
                else:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                
                # Success
                self.stats.successful_attempts += 1
                self.stats.last_success_time = time.time()
                
                logger.debug(
                    f"Async retry successful on attempt {attempt + 1}/{self.config.max_attempts}"
                )
                
                return result
                
            except Exception as e:
                last_exception = e
                self.stats.failed_attempts += 1
                self.stats.last_failure_time = time.time()
                self.stats.exceptions.append(e)
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    logger.debug(f"Non-retryable exception: {e}")
                    raise
                
                # Check if this was the last attempt
                if attempt == self.config.max_attempts - 1:
                    break
                
                # Calculate and apply delay
                delay = RetryDelayCalculator.calculate_delay(
                    attempt, self.config, previous_delay
                )
                previous_delay = delay
                self.stats.total_delay += delay
                
                logger.debug(
                    f"Async retry attempt {attempt + 1}/{self.config.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        raise RetryExhaustedError(
            self.config.max_attempts, last_exception, self.stats
        )
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        # Check non-retryable exceptions first
        if any(isinstance(exception, exc_type) for exc_type in self.config.non_retryable_exceptions):
            return False
        
        # Check retryable exceptions
        return any(isinstance(exception, exc_type) for exc_type in self.config.retryable_exceptions)
    
    def _call_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Call function with timeout (sync version)."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function call timed out after {timeout}s")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel timeout
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    async def _async_call_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Call async function with timeout."""
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        else:
            # Run sync function in executor with timeout
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, func, *args, **kwargs),
                timeout=timeout
            )
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        return {
            "total_attempts": self.stats.total_attempts,
            "successful_attempts": self.stats.successful_attempts,
            "failed_attempts": self.stats.failed_attempts,
            "success_rate": (
                self.stats.successful_attempts / self.stats.total_attempts
                if self.stats.total_attempts > 0 else 0.0
            ),
            "total_delay": self.stats.total_delay,
            "average_delay": (
                self.stats.total_delay / max(1, self.stats.failed_attempts - 1)
                if self.stats.failed_attempts > 1 else 0.0
            ),
            "last_attempt_time": self.stats.last_attempt_time,
            "last_success_time": self.stats.last_success_time,
            "last_failure_time": self.stats.last_failure_time,
            "exception_count": len(self.stats.exceptions)
        }
    
    def reset_stats(self):
        """Reset retry statistics."""
        self.stats = RetryStats()


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    jitter: JitterType = JitterType.FULL,
    backoff_multiplier: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (),
    timeout: Optional[float] = None
):
    """
    Decorator for adding retry logic to functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        strategy: Retry strategy
        jitter: Jitter type for delays
        backoff_multiplier: Multiplier for exponential backoff
        retryable_exceptions: Exceptions that should trigger retry
        non_retryable_exceptions: Exceptions that should not trigger retry
        timeout: Timeout for individual function calls
        
    Example:
        @retry(max_attempts=3, base_delay=1.0)
        def unreliable_function():
            # Function that might fail
            pass
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy,
        jitter=jitter,
        backoff_multiplier=backoff_multiplier,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
        timeout=timeout
    )
    
    return Retry(config)


# Predefined retry configurations
DEFAULT_RETRY = RetryConfig()

NETWORK_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    jitter=JitterType.FULL,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
    non_retryable_exceptions=(ValueError, TypeError)
)

GRPC_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    jitter=JitterType.EQUAL,
    timeout=30.0
)

HTTP_RETRY = RetryConfig(
    max_attempts=4,
    base_delay=0.5,
    max_delay=20.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    jitter=JitterType.FULL,
    timeout=60.0
)