"""
Common utilities for resilient service communication.

This module provides circuit breakers, retry mechanisms, timeout handling,
and graceful degradation patterns for building resilient distributed systems.
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerOpenError,
    CircuitState,
    CircuitBreakerRegistry,
    get_circuit_breaker,
    circuit_breaker
)

from .retry import (
    Retry,
    RetryConfig,
    RetryStrategy,
    JitterType,
    RetryExhaustedError,
    retry,
    DEFAULT_RETRY,
    NETWORK_RETRY,
    GRPC_RETRY,
    HTTP_RETRY
)

from .timeout import (
    TimeoutManager,
    TimeoutConfig,
    TimeoutError,
    ConnectionTimeoutError,
    ReadTimeoutError,
    WriteTimeoutError,
    ConnectionPool,
    AsyncConnectionPool,
    with_timeout,
    async_with_timeout
)

from .graceful_degradation import (
    GracefulDegradationManager,
    DegradationConfig,
    ServiceHealth,
    DegradationLevel,
    ServiceStatus,
    FallbackStrategy,
    CacheFallback,
    DefaultValueFallback,
    AlternativeServiceFallback,
    ReducedFunctionalityFallback,
    get_degradation_manager,
    with_degradation
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig", 
    "CircuitBreakerError",
    "CircuitBreakerOpenError",
    "CircuitState",
    "CircuitBreakerRegistry",
    "get_circuit_breaker",
    "circuit_breaker",
    
    # Retry
    "Retry",
    "RetryConfig",
    "RetryStrategy",
    "JitterType", 
    "RetryExhaustedError",
    "retry",
    "DEFAULT_RETRY",
    "NETWORK_RETRY",
    "GRPC_RETRY",
    "HTTP_RETRY",
    
    # Timeout
    "TimeoutManager",
    "TimeoutConfig",
    "TimeoutError",
    "ConnectionTimeoutError",
    "ReadTimeoutError", 
    "WriteTimeoutError",
    "ConnectionPool",
    "AsyncConnectionPool",
    "with_timeout",
    "async_with_timeout",
    
    # Graceful Degradation
    "GracefulDegradationManager",
    "DegradationConfig",
    "ServiceHealth",
    "DegradationLevel",
    "ServiceStatus",
    "FallbackStrategy",
    "CacheFallback",
    "DefaultValueFallback",
    "AlternativeServiceFallback",
    "ReducedFunctionalityFallback",
    "get_degradation_manager",
    "with_degradation"
]