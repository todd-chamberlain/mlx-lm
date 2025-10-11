"""
gRPC error handling and status codes for MLX distributed inference.
"""

import grpc
from typing import Optional, Dict, Any
from enum import Enum


class MLXErrorCode(Enum):
    """Custom error codes for MLX inference system."""
    
    # General errors
    UNKNOWN = "UNKNOWN"
    INVALID_REQUEST = "INVALID_REQUEST"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    
    # Worker errors
    WORKER_NOT_FOUND = "WORKER_NOT_FOUND"
    WORKER_UNAVAILABLE = "WORKER_UNAVAILABLE"
    WORKER_OVERLOADED = "WORKER_OVERLOADED"
    WORKER_REGISTRATION_FAILED = "WORKER_REGISTRATION_FAILED"
    
    # Model errors
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_NOT_SUPPORTED = "MODEL_NOT_SUPPORTED"
    
    # Inference errors
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    INFERENCE_CANCELLED = "INFERENCE_CANCELLED"
    GENERATION_FAILED = "GENERATION_FAILED"
    
    # Resource errors
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"
    DEVICE_ERROR = "DEVICE_ERROR"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    
    # Network errors
    CONNECTION_FAILED = "CONNECTION_FAILED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"


class MLXError(Exception):
    """Base exception for MLX inference system errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: MLXErrorCode = MLXErrorCode.UNKNOWN,
        grpc_status: grpc.StatusCode = grpc.StatusCode.INTERNAL,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.grpc_status = grpc_status
        self.details = details or {}
    
    def to_grpc_error(self) -> grpc.RpcError:
        """Convert to gRPC RpcError."""
        return grpc.RpcError(self.message)


class ValidationError(MLXError):
    """Raised when message validation fails."""
    
    def __init__(self, message: str, validation_errors: list = None):
        super().__init__(
            message=message,
            error_code=MLXErrorCode.INVALID_REQUEST,
            grpc_status=grpc.StatusCode.INVALID_ARGUMENT,
            details={"validation_errors": validation_errors or []}
        )


class WorkerError(MLXError):
    """Base class for worker-related errors."""
    pass


class WorkerNotFoundError(WorkerError):
    """Raised when a worker is not found."""
    
    def __init__(self, worker_id: str):
        super().__init__(
            message=f"Worker not found: {worker_id}",
            error_code=MLXErrorCode.WORKER_NOT_FOUND,
            grpc_status=grpc.StatusCode.NOT_FOUND,
            details={"worker_id": worker_id}
        )


class WorkerUnavailableError(WorkerError):
    """Raised when no workers are available."""
    
    def __init__(self, message: str = "No workers available"):
        super().__init__(
            message=message,
            error_code=MLXErrorCode.WORKER_UNAVAILABLE,
            grpc_status=grpc.StatusCode.UNAVAILABLE
        )


class WorkerOverloadedError(WorkerError):
    """Raised when a worker is overloaded."""
    
    def __init__(self, worker_id: str, current_load: int, max_load: int):
        super().__init__(
            message=f"Worker {worker_id} is overloaded: {current_load}/{max_load}",
            error_code=MLXErrorCode.WORKER_OVERLOADED,
            grpc_status=grpc.StatusCode.RESOURCE_EXHAUSTED,
            details={
                "worker_id": worker_id,
                "current_load": current_load,
                "max_load": max_load
            }
        )


class ModelError(MLXError):
    """Base class for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a model is not found."""
    
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model not found: {model_name}",
            error_code=MLXErrorCode.MODEL_NOT_FOUND,
            grpc_status=grpc.StatusCode.NOT_FOUND,
            details={"model_name": model_name}
        )


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"Failed to load model {model_name}: {reason}",
            error_code=MLXErrorCode.MODEL_LOAD_FAILED,
            grpc_status=grpc.StatusCode.INTERNAL,
            details={"model_name": model_name, "reason": reason}
        )


class InferenceError(MLXError):
    """Base class for inference-related errors."""
    pass


class InferenceTimeoutError(InferenceError):
    """Raised when inference times out."""
    
    def __init__(self, request_id: str, timeout_seconds: float):
        super().__init__(
            message=f"Inference timeout for request {request_id} after {timeout_seconds}s",
            error_code=MLXErrorCode.INFERENCE_TIMEOUT,
            grpc_status=grpc.StatusCode.DEADLINE_EXCEEDED,
            details={"request_id": request_id, "timeout_seconds": timeout_seconds}
        )


class GenerationError(InferenceError):
    """Raised when text generation fails."""
    
    def __init__(self, request_id: str, reason: str):
        super().__init__(
            message=f"Generation failed for request {request_id}: {reason}",
            error_code=MLXErrorCode.GENERATION_FAILED,
            grpc_status=grpc.StatusCode.INTERNAL,
            details={"request_id": request_id, "reason": reason}
        )


class ResourceError(MLXError):
    """Base class for resource-related errors."""
    pass


class InsufficientMemoryError(ResourceError):
    """Raised when there's insufficient memory."""
    
    def __init__(self, required_gb: float, available_gb: float):
        super().__init__(
            message=f"Insufficient memory: required {required_gb}GB, available {available_gb}GB",
            error_code=MLXErrorCode.INSUFFICIENT_MEMORY,
            grpc_status=grpc.StatusCode.RESOURCE_EXHAUSTED,
            details={"required_gb": required_gb, "available_gb": available_gb}
        )


def map_grpc_status_to_mlx_error(status_code: grpc.StatusCode) -> MLXErrorCode:
    """Map gRPC status codes to MLX error codes."""
    mapping = {
        grpc.StatusCode.OK: MLXErrorCode.UNKNOWN,
        grpc.StatusCode.CANCELLED: MLXErrorCode.INFERENCE_CANCELLED,
        grpc.StatusCode.UNKNOWN: MLXErrorCode.UNKNOWN,
        grpc.StatusCode.INVALID_ARGUMENT: MLXErrorCode.INVALID_REQUEST,
        grpc.StatusCode.DEADLINE_EXCEEDED: MLXErrorCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.NOT_FOUND: MLXErrorCode.WORKER_NOT_FOUND,
        grpc.StatusCode.ALREADY_EXISTS: MLXErrorCode.UNKNOWN,
        grpc.StatusCode.PERMISSION_DENIED: MLXErrorCode.AUTHENTICATION_FAILED,
        grpc.StatusCode.RESOURCE_EXHAUSTED: MLXErrorCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.FAILED_PRECONDITION: MLXErrorCode.INVALID_REQUEST,
        grpc.StatusCode.ABORTED: MLXErrorCode.INFERENCE_CANCELLED,
        grpc.StatusCode.OUT_OF_RANGE: MLXErrorCode.INVALID_REQUEST,
        grpc.StatusCode.UNIMPLEMENTED: MLXErrorCode.UNKNOWN,
        grpc.StatusCode.INTERNAL: MLXErrorCode.UNKNOWN,
        grpc.StatusCode.UNAVAILABLE: MLXErrorCode.SERVICE_UNAVAILABLE,
        grpc.StatusCode.DATA_LOSS: MLXErrorCode.UNKNOWN,
        grpc.StatusCode.UNAUTHENTICATED: MLXErrorCode.AUTHENTICATION_FAILED,
    }
    return mapping.get(status_code, MLXErrorCode.UNKNOWN)


def handle_grpc_error(context: grpc.ServicerContext, error: MLXError) -> None:
    """Handle MLX error by setting appropriate gRPC context."""
    context.set_code(error.grpc_status)
    context.set_details(error.message)
    
    # Add custom metadata for error details
    if error.details:
        metadata = []
        for key, value in error.details.items():
            metadata.append((f"mlx-error-{key}", str(value)))
        context.set_trailing_metadata(metadata)


def create_error_response(error: MLXError) -> Dict[str, Any]:
    """Create a structured error response dictionary."""
    return {
        "error": {
            "code": error.error_code.value,
            "message": error.message,
            "grpc_status": error.grpc_status.name,
            "details": error.details
        }
    }


def wrap_grpc_call(func):
    """Decorator to wrap gRPC calls with error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MLXError as e:
            # Re-raise MLX errors as-is
            raise e
        except grpc.RpcError as e:
            # Convert gRPC errors to MLX errors
            mlx_code = map_grpc_status_to_mlx_error(e.code())
            raise MLXError(
                message=e.details() or "gRPC call failed",
                error_code=mlx_code,
                grpc_status=e.code()
            )
        except Exception as e:
            # Convert unexpected errors to MLX errors
            raise MLXError(
                message=f"Unexpected error: {str(e)}",
                error_code=MLXErrorCode.UNKNOWN,
                grpc_status=grpc.StatusCode.INTERNAL
            )
    return wrapper


class ErrorHandler:
    """Centralized error handling for gRPC services."""
    
    @staticmethod
    def handle_validation_error(context: grpc.ServicerContext, errors: list) -> None:
        """Handle validation errors."""
        error = ValidationError(
            message="Request validation failed",
            validation_errors=errors
        )
        handle_grpc_error(context, error)
    
    @staticmethod
    def handle_worker_error(context: grpc.ServicerContext, error: Exception) -> None:
        """Handle worker-related errors."""
        if isinstance(error, WorkerError):
            handle_grpc_error(context, error)
        else:
            mlx_error = MLXError(
                message=f"Worker error: {str(error)}",
                error_code=MLXErrorCode.WORKER_UNAVAILABLE,
                grpc_status=grpc.StatusCode.UNAVAILABLE
            )
            handle_grpc_error(context, mlx_error)
    
    @staticmethod
    def handle_inference_error(context: grpc.ServicerContext, error: Exception) -> None:
        """Handle inference-related errors."""
        if isinstance(error, InferenceError):
            handle_grpc_error(context, error)
        else:
            mlx_error = MLXError(
                message=f"Inference error: {str(error)}",
                error_code=MLXErrorCode.INFERENCE_FAILED,
                grpc_status=grpc.StatusCode.INTERNAL
            )
            handle_grpc_error(context, mlx_error)
    
    @staticmethod
    def handle_model_error(context: grpc.ServicerContext, error: Exception) -> None:
        """Handle model-related errors."""
        if isinstance(error, ModelError):
            handle_grpc_error(context, error)
        else:
            mlx_error = MLXError(
                message=f"Model error: {str(error)}",
                error_code=MLXErrorCode.MODEL_LOAD_FAILED,
                grpc_status=grpc.StatusCode.INTERNAL
            )
            handle_grpc_error(context, mlx_error)
    
    @staticmethod
    def handle_resource_error(context: grpc.ServicerContext, error: Exception) -> None:
        """Handle resource-related errors."""
        if isinstance(error, ResourceError):
            handle_grpc_error(context, error)
        else:
            mlx_error = MLXError(
                message=f"Resource error: {str(error)}",
                error_code=MLXErrorCode.RESOURCE_EXHAUSTED,
                grpc_status=grpc.StatusCode.RESOURCE_EXHAUSTED
            )
            handle_grpc_error(context, mlx_error)