"""
Circuit Breaker Implementation

Provides circuit breaker pattern for resilient service communication
with configurable failure thresholds and recovery mechanisms.
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Any, Callable, Optional, Dict, Union
from dataclasses import dataclass, field
from threading import Lock
import random
import math

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2
    timeout: float = 30.0
    expected_exception: type = Exception


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0
    current_state: CircuitState = CircuitState.CLOSED


class CircuitBreakerError(Exception):
    """Circuit breaker specific exception."""
    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for resilient service calls.
    
    The circuit breaker monitors calls to a service and opens the circuit
    when failures exceed the threshold, preventing further calls until
    the service recovers.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.stats = CircuitBreakerStats()
        self._state = CircuitState.CLOSED
        self._lock = Lock()
        self._half_open_calls = 0
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function call through circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Exception: Original function exceptions
        """
        with self._lock:
            self._check_and_update_state()
            
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open. Last failure: {self.stats.last_failure_time}"
                )
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        "Circuit breaker is half-open and max calls reached"
                    )
                self._half_open_calls += 1
        
        # Execute the function call
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            self._record_success(time.time() - start_time)
            return result
        except self.config.expected_exception as e:
            self._record_failure(time.time() - start_time, e)
            raise
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function call through circuit breaker.
        
        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Exception: Original function exceptions
        """
        with self._lock:
            self._check_and_update_state()
            
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open. Last failure: {self.stats.last_failure_time}"
                )
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        "Circuit breaker is half-open and max calls reached"
                    )
                self._half_open_calls += 1
        
        # Execute the async function call
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            self._record_success(time.time() - start_time)
            return result
        except self.config.expected_exception as e:
            self._record_failure(time.time() - start_time, e)
            raise
    
    def _check_and_update_state(self):
        """Check and update circuit breaker state."""
        current_time = time.time()
        
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self.stats.last_failure_time and 
                current_time - self.stats.last_failure_time >= self.config.recovery_timeout):
                self._transition_to_half_open()
        
        elif self._state == CircuitState.HALF_OPEN:
            # Check if we should close the circuit
            if self.stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to_closed()
    
    def _record_success(self, duration: float):
        """Record successful call."""
        with self._lock:
            self.stats.total_calls += 1
            self.stats.successful_calls += 1
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0
            self.stats.last_success_time = time.time()
            
            logger.debug(
                f"Circuit breaker success: {self.stats.consecutive_successes} consecutive, "
                f"duration: {duration:.3f}s"
            )
            
            # Transition from half-open to closed if enough successes
            if (self._state == CircuitState.HALF_OPEN and 
                self.stats.consecutive_successes >= self.config.success_threshold):
                self._transition_to_closed()
    
    def _record_failure(self, duration: float, exception: Exception):
        """Record failed call."""
        with self._lock:
            self.stats.total_calls += 1
            self.stats.failed_calls += 1
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self.stats.last_failure_time = time.time()
            
            logger.warning(
                f"Circuit breaker failure: {self.stats.consecutive_failures} consecutive, "
                f"duration: {duration:.3f}s, error: {exception}"
            )
            
            # Transition to open if failure threshold reached
            if (self._state == CircuitState.CLOSED and 
                self.stats.consecutive_failures >= self.config.failure_threshold):
                self._transition_to_open()
            
            # Transition from half-open back to open on any failure
            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition circuit to open state."""
        old_state = self._state
        self._state = CircuitState.OPEN
        self.stats.state_changes += 1
        self._half_open_calls = 0
        
        logger.warning(
            f"Circuit breaker opened: {old_state} -> {self._state}, "
            f"failures: {self.stats.consecutive_failures}"
        )
    
    def _transition_to_half_open(self):
        """Transition circuit to half-open state."""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self.stats.state_changes += 1
        self._half_open_calls = 0
        
        logger.info(
            f"Circuit breaker half-opened: {old_state} -> {self._state}, "
            f"testing recovery after {self.config.recovery_timeout}s"
        )
    
    def _transition_to_closed(self):
        """Transition circuit to closed state."""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self.stats.state_changes += 1
        self._half_open_calls = 0
        
        logger.info(
            f"Circuit breaker closed: {old_state} -> {self._state}, "
            f"successes: {self.stats.consecutive_successes}"
        )
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self.stats = CircuitBreakerStats()
            self._half_open_calls = 0
            
        logger.info("Circuit breaker reset to initial state")
    
    def force_open(self):
        """Force circuit breaker to open state."""
        with self._lock:
            self._transition_to_open()
        
        logger.warning("Circuit breaker forced to open state")
    
    def force_close(self):
        """Force circuit breaker to closed state."""
        with self._lock:
            self._transition_to_closed()
        
        logger.info("Circuit breaker forced to closed state")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "success_rate": (
                    self.stats.successful_calls / self.stats.total_calls 
                    if self.stats.total_calls > 0 else 0.0
                ),
                "consecutive_failures": self.stats.consecutive_failures,
                "consecutive_successes": self.stats.consecutive_successes,
                "last_failure_time": self.stats.last_failure_time,
                "last_success_time": self.stats.last_success_time,
                "state_changes": self.stats.state_changes,
                "half_open_calls": self._half_open_calls if self._state == CircuitState.HALF_OPEN else 0
            }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = Lock()
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Get or create circuit breaker by name.
        
        Args:
            name: Circuit breaker name
            config: Configuration (uses default if not provided)
            
        Returns:
            Circuit breaker instance
        """
        with self._lock:
            if name not in self._breakers:
                if config is None:
                    config = CircuitBreakerConfig()
                self._breakers[name] = CircuitBreaker(config)
                logger.info(f"Created circuit breaker: {name}")
            
            return self._breakers[name]
    
    def remove_breaker(self, name: str) -> bool:
        """
        Remove circuit breaker by name.
        
        Args:
            name: Circuit breaker name
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                logger.info(f"Removed circuit breaker: {name}")
                return True
            return False
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_stats()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("Reset all circuit breakers")


# Global circuit breaker registry
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get circuit breaker from global registry."""
    return _global_registry.get_breaker(name, config)


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for applying circuit breaker to functions.
    
    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
        
    Example:
        @circuit_breaker("api_service")
        def call_api():
            # API call implementation
            pass
    """
    def decorator(func):
        breaker = get_circuit_breaker(name, config)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await breaker.async_call(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return breaker.call(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator