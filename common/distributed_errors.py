"""
Distributed Inference Error Hierarchy

Defines custom exceptions and error codes for distributed layer-level inference.
Provides structured error handling with categorization and context preservation.
"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


class ErrorCode(Enum):
    """Error codes for distributed inference."""
    
    # Pipeline errors (1000-1999)
    PIPELINE_INITIALIZATION_FAILED = 1000
    PIPELINE_EXECUTION_FAILED = 1001
    PIPELINE_TIMEOUT = 1002
    PIPELINE_CANCELLED = 1003
    PIPELINE_INVALID_STATE = 1004
    
    # Worker errors (2000-2999)
    WORKER_UNAVAILABLE = 2000
    WORKER_REGISTRATION_FAILED = 2001
    WORKER_HEALTH_CHECK_FAILED = 2002
    WORKER_OVERLOADED = 2003
    WORKER_DISCONNECTED = 2004
    
    # Transfer errors (3000-3999)
    ACTIVATION_TRANSFER_FAILED = 3000
    ACTIVATION_SERIALIZATION_FAILED = 3001
    ACTIVATION_DESERIALIZATION_FAILED = 3002
    ACTIVATION_COMPRESSION_FAILED = 3003
    ACTIVATION_SIZE_EXCEEDED = 3004
    
    # Generation errors (4000-4999)
    GENERATION_FAILED = 4000
    GENERATION_TIMEOUT = 4001
    GENERATION_INVALID_CONFIG = 4002
    GENERATION_SAMPLING_FAILED = 4003
    GENERATION_KV_CACHE_ERROR = 4004
    
    # Stream errors (5000-5999)
    STREAM_INITIALIZATION_FAILED = 5000
    STREAM_BUFFER_OVERFLOW = 5001
    STREAM_CANCELLED = 5002
    STREAM_TIMEOUT = 5003
    STREAM_BACKPRESSURE_TIMEOUT = 5004
    
    # Resource errors (6000-6999)
    RESOURCE_EXHAUSTED = 6000
    MEMORY_ALLOCATION_FAILED = 6001
    GPU_OUT_OF_MEMORY = 6002
    DISK_SPACE_EXHAUSTED = 6003
    
    # Network errors (7000-7999)
    NETWORK_CONNECTION_FAILED = 7000
    NETWORK_TIMEOUT = 7001
    NETWORK_DISCONNECTED = 7002
    GRPC_ERROR = 7003
    
    # Configuration errors (8000-8999)
    INVALID_CONFIGURATION = 8000
    MISSING_CONFIGURATION = 8001
    CONFIGURATION_VALIDATION_FAILED = 8002
    
    # Unknown/Generic errors (9000-9999)
    UNKNOWN_ERROR = 9000
    INTERNAL_ERROR = 9001


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Recoverable, minimal impact
    MEDIUM = "medium"     # Recoverable, some impact
    HIGH = "high"         # May require intervention
    CRITICAL = "critical" # System-level failure


@dataclass
class ErrorContext:
    """Context information for errors."""
    request_id: Optional[str] = None
    worker_id: Optional[str] = None
    stage_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "worker_id": self.worker_id,
            "stage_id": self.stage_id,
            "component": self.component,
            "operation": self.operation,
            "metadata": self.metadata
        }


class DistributedInferenceError(Exception):
    """Base exception for distributed inference errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize error.
        
        Args:
            message: Error message
            code: Error code
            severity: Error severity
            context: Error context
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
    
    def __str__(self) -> str:
        """String representation."""
        parts = [
            f"[{self.code.name}]",
            f"Severity: {self.severity.value}",
            f"Message: {self.message}"
        ]
        
        if self.context.request_id:
            parts.append(f"Request: {self.context.request_id}")
        if self.context.worker_id:
            parts.append(f"Worker: {self.context.worker_id}")
        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {str(self.cause)}")
        
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "code": self.code.name,
            "code_value": self.code.value,
            "severity": self.severity.value,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None
        }


# Pipeline Errors

class PipelineError(DistributedInferenceError):
    """Base class for pipeline-related errors."""
    pass


class PipelineInitializationError(PipelineError):
    """Pipeline initialization failed."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.PIPELINE_INITIALIZATION_FAILED,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=cause
        )


class PipelineExecutionError(PipelineError):
    """Pipeline execution failed."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.PIPELINE_EXECUTION_FAILED,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=cause
        )


class PipelineTimeoutError(PipelineError):
    """Pipeline execution timed out."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.PIPELINE_TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            cause=cause
        )


class PipelineCancelledError(PipelineError):
    """Pipeline was cancelled."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.PIPELINE_CANCELLED,
            severity=ErrorSeverity.LOW,
            context=context,
            cause=cause
        )


# Worker Errors

class WorkerError(DistributedInferenceError):
    """Base class for worker-related errors."""
    pass


class WorkerUnavailableError(WorkerError):
    """Worker is unavailable."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.WORKER_UNAVAILABLE,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=cause
        )


class WorkerRegistrationError(WorkerError):
    """Worker registration failed."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.WORKER_REGISTRATION_FAILED,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=cause
        )


class WorkerHealthCheckError(WorkerError):
    """Worker health check failed."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.WORKER_HEALTH_CHECK_FAILED,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            cause=cause
        )


class WorkerOverloadedError(WorkerError):
    """Worker is overloaded."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.WORKER_OVERLOADED,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            cause=cause
        )


# Transfer Errors

class TransferError(DistributedInferenceError):
    """Base class for activation transfer errors."""
    pass


class ActivationTransferError(TransferError):
    """Activation transfer failed."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.ACTIVATION_TRANSFER_FAILED,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=cause
        )


class ActivationSerializationError(TransferError):
    """Activation serialization failed."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.ACTIVATION_SERIALIZATION_FAILED,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=cause
        )


class ActivationDeserializationError(TransferError):
    """Activation deserialization failed."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.ACTIVATION_DESERIALIZATION_FAILED,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=cause
        )


# Generation Errors

class GenerationError(DistributedInferenceError):
    """Base class for generation errors."""
    pass


class GenerationFailedError(GenerationError):
    """Token generation failed."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.GENERATION_FAILED,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=cause
        )


class GenerationTimeoutError(GenerationError):
    """Generation timed out."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.GENERATION_TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            cause=cause
        )


class GenerationConfigError(GenerationError):
    """Invalid generation configuration."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.GENERATION_INVALID_CONFIG,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            cause=cause
        )


# Stream Errors

class StreamError(DistributedInferenceError):
    """Base class for streaming errors."""
    pass


class StreamInitializationError(StreamError):
    """Stream initialization failed."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.STREAM_INITIALIZATION_FAILED,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=cause
        )


class StreamBufferOverflowError(StreamError):
    """Stream buffer overflow."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.STREAM_BUFFER_OVERFLOW,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            cause=cause
        )


class StreamCancelledError(StreamError):
    """Stream was cancelled."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.STREAM_CANCELLED,
            severity=ErrorSeverity.LOW,
            context=context,
            cause=cause
        )


class StreamTimeoutError(StreamError):
    """Stream timed out."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.STREAM_TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            cause=cause
        )


# Resource Errors

class ResourceError(DistributedInferenceError):
    """Base class for resource errors."""
    pass


class ResourceExhaustedError(ResourceError):
    """System resources exhausted."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.RESOURCE_EXHAUSTED,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            cause=cause
        )


class MemoryAllocationError(ResourceError):
    """Memory allocation failed."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.MEMORY_ALLOCATION_FAILED,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            cause=cause
        )


class GPUOutOfMemoryError(ResourceError):
    """GPU out of memory."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.GPU_OUT_OF_MEMORY,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            cause=cause
        )


# Network Errors

class NetworkError(DistributedInferenceError):
    """Base class for network errors."""
    pass


class NetworkConnectionError(NetworkError):
    """Network connection failed."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.NETWORK_CONNECTION_FAILED,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=cause
        )


class NetworkTimeoutError(NetworkError):
    """Network operation timed out."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.NETWORK_TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            cause=cause
        )


# Configuration Errors

class ConfigurationError(DistributedInferenceError):
    """Base class for configuration errors."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Invalid configuration."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.INVALID_CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=cause
        )


class MissingConfigurationError(ConfigurationError):
    """Required configuration missing."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(
            message,
            code=ErrorCode.MISSING_CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=cause
        )
