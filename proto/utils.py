"""
Utility functions for gRPC operations and message handling.
"""

import uuid
import time
import logging
from typing import Optional, Dict, Any, List, Union
from contextlib import asynccontextmanager

import grpc
from google.protobuf.message import Message

from .errors import MLXError, MLXErrorCode
from . import inference_pb2


logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def get_current_timestamp() -> float:
    """Get current timestamp in seconds."""
    return time.time()


def format_grpc_address(host: str, port: int) -> str:
    """Format host and port into gRPC address."""
    return f"{host}:{port}"


def parse_grpc_address(address: str) -> tuple[str, int]:
    """Parse gRPC address into host and port."""
    try:
        host, port_str = address.rsplit(':', 1)
        port = int(port_str)
        return host, port
    except (ValueError, AttributeError) as e:
        raise MLXError(
            message=f"Invalid gRPC address format: {address}",
            error_code=MLXErrorCode.INVALID_REQUEST,
            grpc_status=grpc.StatusCode.INVALID_ARGUMENT
        )


def create_inference_request(
    prompt: str,
    model: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    use_cache: bool = True,
    request_id: Optional[str] = None
) -> inference_pb2.InferenceRequest:
    """Create an InferenceRequest message with validation."""
    request = inference_pb2.InferenceRequest()
    request.request_id = request_id or generate_request_id()
    request.prompt = prompt
    request.model = model
    request.max_tokens = max_tokens
    request.temperature = temperature
    request.top_p = top_p
    request.use_cache = use_cache
    
    if seed is not None:
        request.seed = seed
    
    return request


def create_worker_registration(
    worker_id: str,
    address: str,
    device_info: Dict[str, Any],
    max_concurrent: int = 4,
    supported_models: Optional[List[str]] = None
) -> inference_pb2.WorkerRegistration:
    """Create a WorkerRegistration message."""
    registration = inference_pb2.WorkerRegistration()
    registration.worker_id = worker_id
    registration.address = address
    registration.max_concurrent = max_concurrent
    
    # Set device info
    registration.device_info.model = device_info.get('model', '')
    registration.device_info.memory_gb = device_info.get('memory_gb', 0)
    registration.device_info.gpu_cores = device_info.get('gpu_cores', 0)
    registration.device_info.os_version = device_info.get('os_version', '')
    registration.device_info.mlx_version = device_info.get('mlx_version', '')
    
    # Set supported models
    if supported_models:
        registration.supported_models.extend(supported_models)
    
    return registration


def create_heartbeat_request(
    worker_id: str,
    status: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None
) -> inference_pb2.HeartbeatRequest:
    """Create a HeartbeatRequest message."""
    heartbeat = inference_pb2.HeartbeatRequest()
    heartbeat.worker_id = worker_id
    
    # Set worker status
    if status:
        heartbeat.status.status = status.get('status', 'healthy')
        heartbeat.status.current_load = status.get('current_load', 0)
        heartbeat.status.memory_usage_percent = status.get('memory_usage_percent', 0.0)
        heartbeat.status.uptime_seconds = status.get('uptime_seconds', 0)
    
    # Set worker metrics
    if metrics:
        heartbeat.metrics.avg_latency_ms = metrics.get('avg_latency_ms', 0.0)
        heartbeat.metrics.tokens_per_second = metrics.get('tokens_per_second', 0.0)
        heartbeat.metrics.total_requests = metrics.get('total_requests', 0)
        heartbeat.metrics.failed_requests = metrics.get('failed_requests', 0)
        heartbeat.metrics.active_requests = metrics.get('active_requests', 0)
        heartbeat.metrics.memory_usage_gb = metrics.get('memory_usage_gb', 0.0)
    
    return heartbeat


def create_stream_chunk(
    request_id: str,
    text: str,
    token: int = 0,
    is_final: bool = False,
    worker_id: Optional[str] = None
) -> inference_pb2.StreamChunk:
    """Create a StreamChunk message."""
    chunk = inference_pb2.StreamChunk()
    chunk.request_id = request_id
    chunk.text = text
    chunk.token = token
    chunk.is_final = is_final
    
    if worker_id:
        chunk.worker_id = worker_id
    
    return chunk


def create_inference_response(
    request_id: str,
    text: str,
    tokens: Optional[List[int]] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    tokens_per_second: float = 0.0,
    worker_id: Optional[str] = None,
    latency_ms: float = 0.0
) -> inference_pb2.InferenceResponse:
    """Create an InferenceResponse message."""
    response = inference_pb2.InferenceResponse()
    response.request_id = request_id
    response.text = text
    response.prompt_tokens = prompt_tokens
    response.completion_tokens = completion_tokens
    response.tokens_per_second = tokens_per_second
    response.latency_ms = latency_ms
    
    if tokens:
        response.tokens.extend(tokens)
    
    if worker_id:
        response.worker_id = worker_id
    
    return response


def extract_metadata_dict(metadata) -> Dict[str, str]:
    """Extract metadata into a dictionary."""
    return {key: value for key, value in metadata}


def create_metadata_list(metadata_dict: Dict[str, str]) -> List[tuple]:
    """Create metadata list from dictionary."""
    return [(key, value) for key, value in metadata_dict.items()]


def log_grpc_call(
    method_name: str,
    request: Message,
    response: Optional[Message] = None,
    error: Optional[Exception] = None,
    duration_ms: Optional[float] = None
) -> None:
    """Log gRPC call details."""
    log_data = {
        'method': method_name,
        'request_type': type(request).__name__,
    }
    
    # Add request ID if available
    if hasattr(request, 'request_id') and request.request_id:
        log_data['request_id'] = request.request_id
    
    # Add worker ID if available
    if hasattr(request, 'worker_id') and request.worker_id:
        log_data['worker_id'] = request.worker_id
    
    if duration_ms is not None:
        log_data['duration_ms'] = duration_ms
    
    if error:
        log_data['error'] = str(error)
        log_data['error_type'] = type(error).__name__
        logger.error(f"gRPC call failed: {log_data}")
    else:
        if response:
            log_data['response_type'] = type(response).__name__
        logger.info(f"gRPC call completed: {log_data}")


@asynccontextmanager
async def grpc_call_context(method_name: str, request: Message):
    """Context manager for gRPC calls with logging and timing."""
    start_time = time.time()
    error = None
    response = None
    
    try:
        yield
    except Exception as e:
        error = e
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        log_grpc_call(method_name, request, response, error, duration_ms)


def is_retryable_error(error: grpc.RpcError) -> bool:
    """Check if a gRPC error is retryable."""
    retryable_codes = {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.ABORTED,
        grpc.StatusCode.INTERNAL,
    }
    return error.code() in retryable_codes


def get_error_details(error: grpc.RpcError) -> Dict[str, Any]:
    """Extract error details from gRPC error."""
    details = {
        'code': error.code().name,
        'message': error.details() or 'No details available',
    }
    
    # Extract custom metadata if available
    if hasattr(error, 'trailing_metadata') and error.trailing_metadata():
        metadata = {}
        for key, value in error.trailing_metadata():
            if key.startswith('mlx-error-'):
                metadata[key[10:]] = value  # Remove 'mlx-error-' prefix
        if metadata:
            details['metadata'] = metadata
    
    return details


def create_health_check_request(include_workers: bool = False) -> inference_pb2.HealthCheckRequest:
    """Create a HealthCheckRequest message."""
    request = inference_pb2.HealthCheckRequest()
    request.include_workers = include_workers
    return request


def create_empty_message() -> inference_pb2.Empty:
    """Create an Empty message."""
    return inference_pb2.Empty()


class MessageBuilder:
    """Builder class for creating complex protobuf messages."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the builder state."""
        self._request_id = None
        self._worker_id = None
        self._model = None
        self._prompt = None
        self._max_tokens = 100
        self._temperature = 0.7
        self._top_p = 0.9
        self._seed = None
        self._use_cache = True
        return self
    
    def request_id(self, request_id: str):
        """Set request ID."""
        self._request_id = request_id
        return self
    
    def worker_id(self, worker_id: str):
        """Set worker ID."""
        self._worker_id = worker_id
        return self
    
    def model(self, model: str):
        """Set model name."""
        self._model = model
        return self
    
    def prompt(self, prompt: str):
        """Set prompt text."""
        self._prompt = prompt
        return self
    
    def max_tokens(self, max_tokens: int):
        """Set max tokens."""
        self._max_tokens = max_tokens
        return self
    
    def temperature(self, temperature: float):
        """Set temperature."""
        self._temperature = temperature
        return self
    
    def top_p(self, top_p: float):
        """Set top_p."""
        self._top_p = top_p
        return self
    
    def seed(self, seed: int):
        """Set seed."""
        self._seed = seed
        return self
    
    def use_cache(self, use_cache: bool):
        """Set use_cache flag."""
        self._use_cache = use_cache
        return self
    
    def build_inference_request(self) -> inference_pb2.InferenceRequest:
        """Build InferenceRequest message."""
        if not self._prompt:
            raise MLXError(
                message="Prompt is required for inference request",
                error_code=MLXErrorCode.INVALID_REQUEST,
                grpc_status=grpc.StatusCode.INVALID_ARGUMENT
            )
        
        if not self._model:
            raise MLXError(
                message="Model is required for inference request",
                error_code=MLXErrorCode.INVALID_REQUEST,
                grpc_status=grpc.StatusCode.INVALID_ARGUMENT
            )
        
        return create_inference_request(
            prompt=self._prompt,
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            seed=self._seed,
            use_cache=self._use_cache,
            request_id=self._request_id
        )


# Convenience functions for common operations
def quick_inference_request(prompt: str, model: str, **kwargs) -> inference_pb2.InferenceRequest:
    """Quick way to create an inference request."""
    builder = MessageBuilder()
    return (builder
            .prompt(prompt)
            .model(model)
            .max_tokens(kwargs.get('max_tokens', 100))
            .temperature(kwargs.get('temperature', 0.7))
            .top_p(kwargs.get('top_p', 0.9))
            .use_cache(kwargs.get('use_cache', True))
            .build_inference_request())


def quick_worker_registration(worker_id: str, address: str, device_info: Dict[str, Any]) -> inference_pb2.WorkerRegistration:
    """Quick way to create a worker registration."""
    return create_worker_registration(
        worker_id=worker_id,
        address=address,
        device_info=device_info,
        max_concurrent=device_info.get('max_concurrent', 4),
        supported_models=device_info.get('supported_models', [])
    )