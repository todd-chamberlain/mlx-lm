"""
Tests for distributed inference error handling and recovery.

Tests error hierarchy, retry mechanisms, circuit breakers, and recovery procedures.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock

from common.distributed_errors import (
    DistributedInferenceError,
    ErrorCode,
    ErrorSeverity,
    ErrorContext,
    PipelineError,
    PipelineInitializationError,
    PipelineExecutionError,
    PipelineTimeoutError,
    WorkerError,
    WorkerUnavailableError,
    TransferError,
    ActivationTransferError,
    GenerationError,
    GenerationFailedError,
    StreamError,
    StreamBufferOverflowError,
    ResourceError,
    ResourceExhaustedError,
    NetworkError,
    NetworkConnectionError,
    ConfigurationError,
    InvalidConfigurationError
)

from common.error_recovery import (
    RetryPolicy,
    RetryStrategy,
    CircuitBreaker,
    CircuitBreakerState,
    ErrorRecovery,
    FallbackHandler,
    retry_async,
    retry_sync
)


class TestErrorHierarchy:
    """Test error class hierarchy and structure."""
    
    def test_base_error_creation(self):
        """Test creating base error."""
        error = DistributedInferenceError(
            "Test error",
            code=ErrorCode.UNKNOWN_ERROR,
            severity=ErrorSeverity.MEDIUM
        )
        
        assert error.message == "Test error"
        assert error.code == ErrorCode.UNKNOWN_ERROR
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context is not None
    
    def test_error_with_context(self):
        """Test error with context information."""
        context = ErrorContext(
            request_id="req-123",
            worker_id="worker-1",
            component="test_component"
        )
        
        error = DistributedInferenceError(
            "Test error",
            context=context
        )
        
        assert error.context.request_id == "req-123"
        assert error.context.worker_id == "worker-1"
        assert error.context.component == "test_component"
    
    def test_error_with_cause(self):
        """Test error with underlying cause."""
        cause = ValueError("Original error")
        error = DistributedInferenceError(
            "Wrapped error",
            cause=cause
        )
        
        assert error.cause == cause
        assert "ValueError" in str(error)
    
    def test_error_to_dict(self):
        """Test error serialization to dict."""
        context = ErrorContext(request_id="req-123")
        error = DistributedInferenceError(
            "Test error",
            code=ErrorCode.PIPELINE_EXECUTION_FAILED,
            severity=ErrorSeverity.HIGH,
            context=context
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["message"] == "Test error"
        assert error_dict["code"] == "PIPELINE_EXECUTION_FAILED"
        assert error_dict["severity"] == "high"
        assert error_dict["context"]["request_id"] == "req-123"
    
    def test_pipeline_errors(self):
        """Test pipeline error types."""
        error1 = PipelineInitializationError("Init failed")
        assert isinstance(error1, PipelineError)
        assert error1.code == ErrorCode.PIPELINE_INITIALIZATION_FAILED
        assert error1.severity == ErrorSeverity.HIGH
        
        error2 = PipelineExecutionError("Execution failed")
        assert error2.code == ErrorCode.PIPELINE_EXECUTION_FAILED
        
        error3 = PipelineTimeoutError("Timeout")
        assert error3.code == ErrorCode.PIPELINE_TIMEOUT
        assert error3.severity == ErrorSeverity.MEDIUM
    
    def test_worker_errors(self):
        """Test worker error types."""
        error = WorkerUnavailableError("Worker down")
        assert isinstance(error, WorkerError)
        assert error.code == ErrorCode.WORKER_UNAVAILABLE
        assert error.severity == ErrorSeverity.HIGH
    
    def test_transfer_errors(self):
        """Test transfer error types."""
        error = ActivationTransferError("Transfer failed")
        assert isinstance(error, TransferError)
        assert error.code == ErrorCode.ACTIVATION_TRANSFER_FAILED
    
    def test_generation_errors(self):
        """Test generation error types."""
        error = GenerationFailedError("Generation failed")
        assert isinstance(error, GenerationError)
        assert error.code == ErrorCode.GENERATION_FAILED
    
    def test_stream_errors(self):
        """Test stream error types."""
        error = StreamBufferOverflowError("Buffer overflow")
        assert isinstance(error, StreamError)
        assert error.code == ErrorCode.STREAM_BUFFER_OVERFLOW
    
    def test_resource_errors(self):
        """Test resource error types."""
        error = ResourceExhaustedError("Out of resources")
        assert isinstance(error, ResourceError)
        assert error.code == ErrorCode.RESOURCE_EXHAUSTED
        assert error.severity == ErrorSeverity.CRITICAL
    
    def test_network_errors(self):
        """Test network error types."""
        error = NetworkConnectionError("Connection failed")
        assert isinstance(error, NetworkError)
        assert error.code == ErrorCode.NETWORK_CONNECTION_FAILED
    
    def test_configuration_errors(self):
        """Test configuration error types."""
        error = InvalidConfigurationError("Invalid config")
        assert isinstance(error, ConfigurationError)
        assert error.code == ErrorCode.INVALID_CONFIGURATION


class TestRetryPolicy:
    """Test retry policy configuration and behavior."""
    
    def test_default_policy(self):
        """Test default retry policy."""
        policy = RetryPolicy()
        
        assert policy.max_attempts == 3
        assert policy.initial_delay == 0.1
        assert policy.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
    
    def test_should_retry_retryable_error(self):
        """Test retry decision for retryable errors."""
        policy = RetryPolicy(max_attempts=3)
        error = NetworkConnectionError("Connection failed")
        
        assert policy.should_retry(error, 1) == True
        assert policy.should_retry(error, 2) == True
        assert policy.should_retry(error, 3) == False  # Max attempts reached
    
    def test_should_not_retry_critical_error(self):
        """Test no retry for critical errors."""
        policy = RetryPolicy()
        error = ResourceExhaustedError("Out of memory")
        
        assert policy.should_retry(error, 1) == False
    
    def test_exponential_backoff_delay(self):
        """Test exponential backoff delay calculation."""
        policy = RetryPolicy(
            initial_delay=1.0,
            backoff_multiplier=2.0,
            jitter=False,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(3) == 4.0
        assert policy.get_delay(4) == 8.0
    
    def test_linear_backoff_delay(self):
        """Test linear backoff delay calculation."""
        policy = RetryPolicy(
            initial_delay=1.0,
            jitter=False,
            strategy=RetryStrategy.LINEAR_BACKOFF
        )
        
        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(3) == 3.0
    
    def test_fixed_delay(self):
        """Test fixed delay strategy."""
        policy = RetryPolicy(
            initial_delay=0.5,
            jitter=False,
            strategy=RetryStrategy.FIXED_DELAY
        )
        
        assert policy.get_delay(1) == 0.5
        assert policy.get_delay(2) == 0.5
        assert policy.get_delay(3) == 0.5
    
    def test_max_delay_cap(self):
        """Test maximum delay cap."""
        policy = RetryPolicy(
            initial_delay=1.0,
            max_delay=5.0,
            backoff_multiplier=2.0,
            jitter=False,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(3) == 4.0
        assert policy.get_delay(4) == 5.0  # Capped at max_delay
        assert policy.get_delay(5) == 5.0
    
    def test_jitter_adds_randomness(self):
        """Test jitter adds randomness to delay."""
        policy = RetryPolicy(
            initial_delay=1.0,
            jitter=True,
            strategy=RetryStrategy.FIXED_DELAY
        )
        
        delays = [policy.get_delay(1) for _ in range(10)]
        
        # All delays should be between 0.5 and 1.0
        assert all(0.5 <= d <= 1.0 for d in delays)
        # Should have some variation
        assert len(set(delays)) > 1


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.can_attempt() == True
    
    def test_opens_after_threshold_failures(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)
        
        assert cb.state == CircuitBreakerState.CLOSED
        
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED
        
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED
        
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.can_attempt() == False
    
    def test_half_open_after_timeout(self):
        """Test circuit enters half-open state after timeout."""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        # Trigger opening
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Should allow attempt and enter half-open
        assert cb.can_attempt() == True
        assert cb.state == CircuitBreakerState.HALF_OPEN
    
    def test_closes_after_success_threshold(self):
        """Test circuit closes after success threshold in half-open."""
        cb = CircuitBreaker(
            failure_threshold=2,
            success_threshold=2,
            timeout=0.1
        )
        
        # Open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # Wait and enter half-open
        time.sleep(0.15)
        cb.can_attempt()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Record successes
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open state."""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        # Open circuit
        cb.record_failure()
        cb.record_failure()
        
        # Enter half-open
        time.sleep(0.15)
        cb.can_attempt()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Fail again
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
    
    def test_reset(self):
        """Test circuit breaker reset."""
        cb = CircuitBreaker(failure_threshold=2)
        
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        cb.reset()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0


class TestErrorRecovery:
    """Test error recovery coordinator."""
    
    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self):
        """Test successful operation doesn't retry."""
        call_count = [0]
        
        async def operation():
            call_count[0] += 1
            return "success"
        
        recovery = ErrorRecovery()
        result = await recovery.execute_with_retry(operation, "test_op")
        
        assert result == "success"
        assert call_count[0] == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self):
        """Test retry on retryable error."""
        call_count = [0]
        
        async def operation():
            call_count[0] += 1
            if call_count[0] < 3:
                raise NetworkConnectionError("Connection failed")
            return "success"
        
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01)
        recovery = ErrorRecovery(retry_policy=policy)
        
        result = await recovery.execute_with_retry(operation, "test_op")
        
        assert result == "success"
        assert call_count[0] == 3
    
    @pytest.mark.asyncio
    async def test_no_retry_on_critical_error(self):
        """Test no retry on critical error."""
        call_count = [0]
        
        async def operation():
            call_count[0] += 1
            raise ResourceExhaustedError("Out of memory")
        
        recovery = ErrorRecovery()
        
        with pytest.raises(ResourceExhaustedError):
            await recovery.execute_with_retry(operation, "test_op")
        
        assert call_count[0] == 1
    
    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        """Test all retries exhausted."""
        call_count = [0]
        
        async def operation():
            call_count[0] += 1
            raise NetworkConnectionError("Connection failed")
        
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01)
        recovery = ErrorRecovery(retry_policy=policy)
        
        with pytest.raises(NetworkConnectionError):
            await recovery.execute_with_retry(operation, "test_op")
        
        assert call_count[0] == 3
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with retry."""
        call_count = [0]
        
        async def operation():
            call_count[0] += 1
            raise NetworkConnectionError("Connection failed")
        
        cb = CircuitBreaker(failure_threshold=2)
        policy = RetryPolicy(max_attempts=2, initial_delay=0.01)
        recovery = ErrorRecovery(retry_policy=policy, circuit_breaker=cb)
        
        # First attempt - should fail and open circuit after 2 failures
        with pytest.raises(NetworkConnectionError):
            await recovery.execute_with_retry(operation, "test_op")
        
        assert cb.state == CircuitBreakerState.OPEN
        assert call_count[0] == 2  # Should have tried twice before opening
        
        # Second attempt - should fail immediately due to open circuit
        with pytest.raises(DistributedInferenceError) as exc_info:
            await recovery.execute_with_retry(operation, "test_op")
        
        assert "Circuit breaker open" in str(exc_info.value)
        assert call_count[0] == 2  # Should not have incremented
    
    def test_sync_retry(self):
        """Test synchronous retry."""
        call_count = [0]
        
        def operation():
            call_count[0] += 1
            if call_count[0] < 2:
                raise NetworkConnectionError("Connection failed")
            return "success"
        
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01)
        recovery = ErrorRecovery(retry_policy=policy)
        
        result = recovery.execute_with_retry_sync(operation, "test_op")
        
        assert result == "success"
        assert call_count[0] == 2


class TestFallbackHandler:
    """Test fallback handler for graceful degradation."""
    
    @pytest.mark.asyncio
    async def test_successful_operation_no_fallback(self):
        """Test successful operation doesn't use fallback."""
        async def operation():
            return "primary"
        
        async def fallback():
            return "fallback"
        
        handler = FallbackHandler()
        handler.register_fallback("test_op", fallback)
        
        result = await handler.execute_with_fallback(operation, "test_op")
        assert result == "primary"
    
    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """Test fallback is used on failure."""
        async def operation():
            raise NetworkConnectionError("Connection failed")
        
        async def fallback():
            return "fallback"
        
        handler = FallbackHandler()
        handler.register_fallback("test_op", fallback)
        
        result = await handler.execute_with_fallback(operation, "test_op")
        assert result == "fallback"
    
    @pytest.mark.asyncio
    async def test_default_value_when_no_fallback(self):
        """Test default value when no fallback registered."""
        async def operation():
            raise NetworkConnectionError("Connection failed")
        
        handler = FallbackHandler()
        
        result = await handler.execute_with_fallback(
            operation,
            "test_op",
            default_value="default"
        )
        assert result == "default"
    
    @pytest.mark.asyncio
    async def test_raises_when_no_fallback_or_default(self):
        """Test raises exception when no fallback or default."""
        async def operation():
            raise NetworkConnectionError("Connection failed")
        
        handler = FallbackHandler()
        
        with pytest.raises(NetworkConnectionError):
            await handler.execute_with_fallback(operation, "test_op")


class TestConvenienceFunctions:
    """Test convenience retry functions."""
    
    @pytest.mark.asyncio
    async def test_retry_async_convenience(self):
        """Test async retry convenience function."""
        call_count = [0]
        
        async def operation():
            call_count[0] += 1
            if call_count[0] < 2:
                raise NetworkConnectionError("Connection failed")
            return "success"
        
        result = await retry_async(operation, max_attempts=3, initial_delay=0.01)
        
        assert result == "success"
        assert call_count[0] == 2
    
    def test_retry_sync_convenience(self):
        """Test sync retry convenience function."""
        call_count = [0]
        
        def operation():
            call_count[0] += 1
            if call_count[0] < 2:
                raise NetworkConnectionError("Connection failed")
            return "success"
        
        result = retry_sync(operation, max_attempts=3, initial_delay=0.01)
        
        assert result == "success"
        assert call_count[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
