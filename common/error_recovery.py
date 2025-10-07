"""
Error Recovery and Resilience Mechanisms

Implements retry policies, circuit breakers, and recovery procedures
for distributed inference system.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Any, TypeVar, List
from dataclasses import dataclass, field
from enum import Enum
import random

from .distributed_errors import (
    DistributedInferenceError,
    ErrorSeverity,
    NetworkError,
    WorkerError,
    ResourceError
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    NO_RETRY = "no_retry"


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 0.1  # seconds
    max_delay: float = 10.0     # seconds
    backoff_multiplier: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Error types that should trigger retry
    retryable_errors: List[type] = field(default_factory=lambda: [
        NetworkError,
        WorkerError
    ])
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if error should trigger retry.
        
        Args:
            error: Exception that occurred
            attempt: Current attempt number (1-indexed)
            
        Returns:
            True if should retry
        """
        # Check attempt limit
        if attempt >= self.max_attempts:
            return False
        
        # Check if error is retryable
        if isinstance(error, DistributedInferenceError):
            # Don't retry critical errors
            if error.severity == ErrorSeverity.CRITICAL:
                return False
            
            # Check if error type is retryable
            for retryable_type in self.retryable_errors:
                if isinstance(error, retryable_type):
                    return True
        
        return False
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay before next retry.
        
        Args:
            attempt: Current attempt number (1-indexed)
            
        Returns:
            Delay in seconds
        """
        if self.strategy == RetryStrategy.NO_RETRY:
            return 0.0
        
        elif self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.initial_delay
        
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.initial_delay * attempt
        
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.initial_delay * (self.backoff_multiplier ** (attempt - 1))
        
        else:
            delay = self.initial_delay
        
        # Apply max delay cap
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by temporarily blocking requests
    to failing components.
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0  # seconds
    
    # State
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    
    def record_success(self):
        """Record successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._close()
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._open()
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Failed during test, reopen circuit
            self._open()
    
    def can_attempt(self) -> bool:
        """
        Check if operation can be attempted.
        
        Returns:
            True if operation should be attempted
        """
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        elif self.state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.timeout:
                    self._half_open()
                    return True
            return False
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def _open(self):
        """Open the circuit."""
        logger.warning("Circuit breaker opened due to failures")
        self.state = CircuitBreakerState.OPEN
        self.success_count = 0
    
    def _half_open(self):
        """Enter half-open state for testing."""
        logger.info("Circuit breaker entering half-open state")
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.failure_count = 0
    
    def _close(self):
        """Close the circuit."""
        logger.info("Circuit breaker closed, normal operation resumed")
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        self._close()


class ErrorRecovery:
    """
    Error recovery coordinator.
    
    Provides retry logic, circuit breaker integration, and recovery procedures.
    """
    
    def __init__(
        self,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        """
        Initialize error recovery.
        
        Args:
            retry_policy: Retry policy configuration
            circuit_breaker: Circuit breaker instance
        """
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = circuit_breaker
    
    async def execute_with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str = "operation"
    ) -> T:
        """
        Execute operation with retry logic.
        
        Args:
            operation: Async callable to execute
            operation_name: Name for logging
            
        Returns:
            Operation result
            
        Raises:
            Last exception if all retries exhausted
        """
        last_error = None
        
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            # Check circuit breaker
            if self.circuit_breaker and not self.circuit_breaker.can_attempt():
                raise DistributedInferenceError(
                    f"Circuit breaker open for {operation_name}",
                    severity=ErrorSeverity.HIGH
                )
            
            try:
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation()
                else:
                    result = operation()
                
                # Record success
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                if attempt > 1:
                    logger.info(
                        f"{operation_name} succeeded on attempt {attempt}"
                    )
                
                return result
            
            except Exception as e:
                last_error = e
                
                # Record failure
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                
                # Check if should retry
                if not self.retry_policy.should_retry(e, attempt):
                    logger.error(
                        f"{operation_name} failed on attempt {attempt}, "
                        f"not retrying: {e}"
                    )
                    raise
                
                # Calculate delay
                delay = self.retry_policy.get_delay(attempt)
                
                logger.warning(
                    f"{operation_name} failed on attempt {attempt}/{self.retry_policy.max_attempts}, "
                    f"retrying in {delay:.2f}s: {e}"
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries exhausted
        logger.error(
            f"{operation_name} failed after {self.retry_policy.max_attempts} attempts"
        )
        raise last_error
    
    def execute_with_retry_sync(
        self,
        operation: Callable[[], T],
        operation_name: str = "operation"
    ) -> T:
        """
        Execute synchronous operation with retry logic.
        
        Args:
            operation: Callable to execute
            operation_name: Name for logging
            
        Returns:
            Operation result
            
        Raises:
            Last exception if all retries exhausted
        """
        last_error = None
        
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            # Check circuit breaker
            if self.circuit_breaker and not self.circuit_breaker.can_attempt():
                raise DistributedInferenceError(
                    f"Circuit breaker open for {operation_name}",
                    severity=ErrorSeverity.HIGH
                )
            
            try:
                # Execute operation
                result = operation()
                
                # Record success
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                if attempt > 1:
                    logger.info(
                        f"{operation_name} succeeded on attempt {attempt}"
                    )
                
                return result
            
            except Exception as e:
                last_error = e
                
                # Record failure
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                
                # Check if should retry
                if not self.retry_policy.should_retry(e, attempt):
                    logger.error(
                        f"{operation_name} failed on attempt {attempt}, "
                        f"not retrying: {e}"
                    )
                    raise
                
                # Calculate delay
                delay = self.retry_policy.get_delay(attempt)
                
                logger.warning(
                    f"{operation_name} failed on attempt {attempt}/{self.retry_policy.max_attempts}, "
                    f"retrying in {delay:.2f}s: {e}"
                )
                
                # Wait before retry
                time.sleep(delay)
        
        # All retries exhausted
        logger.error(
            f"{operation_name} failed after {self.retry_policy.max_attempts} attempts"
        )
        raise last_error


class FallbackHandler:
    """
    Handles graceful degradation with fallback strategies.
    """
    
    def __init__(self):
        """Initialize fallback handler."""
        self.fallback_strategies = {}
    
    def register_fallback(
        self,
        operation_name: str,
        fallback: Callable[[], T]
    ):
        """
        Register fallback for operation.
        
        Args:
            operation_name: Name of operation
            fallback: Fallback callable
        """
        self.fallback_strategies[operation_name] = fallback
    
    async def execute_with_fallback(
        self,
        operation: Callable[[], T],
        operation_name: str,
        default_value: Optional[T] = None
    ) -> T:
        """
        Execute operation with fallback on failure.
        
        Args:
            operation: Primary operation
            operation_name: Operation name
            default_value: Default value if all fail
            
        Returns:
            Operation result or fallback result
        """
        try:
            # Try primary operation
            if asyncio.iscoroutinefunction(operation):
                return await operation()
            else:
                return operation()
        
        except Exception as e:
            logger.warning(
                f"Primary operation {operation_name} failed: {e}, "
                f"attempting fallback"
            )
            
            # Try registered fallback
            if operation_name in self.fallback_strategies:
                try:
                    fallback = self.fallback_strategies[operation_name]
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback()
                    else:
                        return fallback()
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback for {operation_name} also failed: {fallback_error}"
                    )
            
            # Return default value if available
            if default_value is not None:
                logger.info(f"Returning default value for {operation_name}")
                return default_value
            
            # No fallback available
            raise


# Convenience functions

async def retry_async(
    operation: Callable[[], T],
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    operation_name: str = "operation"
) -> T:
    """
    Convenience function for retrying async operations.
    
    Args:
        operation: Async operation to retry
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay between retries
        operation_name: Operation name for logging
        
    Returns:
        Operation result
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        initial_delay=initial_delay
    )
    recovery = ErrorRecovery(retry_policy=policy)
    return await recovery.execute_with_retry(operation, operation_name)


def retry_sync(
    operation: Callable[[], T],
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    operation_name: str = "operation"
) -> T:
    """
    Convenience function for retrying sync operations.
    
    Args:
        operation: Operation to retry
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay between retries
        operation_name: Operation name for logging
        
    Returns:
        Operation result
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        initial_delay=initial_delay
    )
    recovery = ErrorRecovery(retry_policy=policy)
    return recovery.execute_with_retry_sync(operation, operation_name)
