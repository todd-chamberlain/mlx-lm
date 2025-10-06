"""
MLX Distributed Inference Protocol Buffer Package.

This package contains gRPC protocol definitions, client/server base classes,
error handling, validation, and utilities for the MLX distributed inference system.
"""

# Import generated protobuf modules
try:
    from . import inference_pb2
    from . import inference_pb2_grpc
except ImportError:
    # Protobuf files not generated yet
    inference_pb2 = None
    inference_pb2_grpc = None

# Import core modules
from .errors import (
    MLXError,
    MLXErrorCode,
    ValidationError,
    WorkerError,
    WorkerNotFoundError,
    WorkerUnavailableError,
    WorkerOverloadedError,
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    InferenceError,
    InferenceTimeoutError,
    GenerationError,
    ResourceError,
    InsufficientMemoryError,
    ErrorHandler,
    handle_grpc_error,
    create_error_response,
    wrap_grpc_call,
)

from .grpc_base import (
    RetryConfig,
    GRPCServerBase,
    GRPCClientBase,
    MessageSerializer,
    HealthChecker,
)

from .validation import (
    MessageValidator,
    validate_message,
    ensure_valid_message,
)

from .utils import (
    generate_request_id,
    get_current_timestamp,
    format_grpc_address,
    parse_grpc_address,
    create_inference_request,
    create_worker_registration,
    create_heartbeat_request,
    create_stream_chunk,
    create_inference_response,
    create_health_check_request,
    create_empty_message,
    MessageBuilder,
    quick_inference_request,
    quick_worker_registration,
    log_grpc_call,
    grpc_call_context,
    is_retryable_error,
    get_error_details,
)

__all__ = [
    # Generated protobuf modules
    'inference_pb2',
    'inference_pb2_grpc',
    
    # Error handling
    'MLXError',
    'MLXErrorCode',
    'ValidationError',
    'WorkerError',
    'WorkerNotFoundError',
    'WorkerUnavailableError',
    'WorkerOverloadedError',
    'ModelError',
    'ModelNotFoundError',
    'ModelLoadError',
    'InferenceError',
    'InferenceTimeoutError',
    'GenerationError',
    'ResourceError',
    'InsufficientMemoryError',
    'ErrorHandler',
    'handle_grpc_error',
    'create_error_response',
    'wrap_grpc_call',
    
    # Base classes
    'RetryConfig',
    'GRPCServerBase',
    'GRPCClientBase',
    'MessageSerializer',
    'HealthChecker',
    
    # Validation
    'MessageValidator',
    'validate_message',
    'ensure_valid_message',
    
    # Utilities
    'generate_request_id',
    'get_current_timestamp',
    'format_grpc_address',
    'parse_grpc_address',
    'create_inference_request',
    'create_worker_registration',
    'create_heartbeat_request',
    'create_stream_chunk',
    'create_inference_response',
    'create_health_check_request',
    'create_empty_message',
    'MessageBuilder',
    'quick_inference_request',
    'quick_worker_registration',
    'log_grpc_call',
    'grpc_call_context',
    'is_retryable_error',
    'get_error_details',
]