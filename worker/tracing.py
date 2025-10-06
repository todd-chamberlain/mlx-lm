"""
Distributed tracing setup for the MLX worker service using OpenTelemetry and Jaeger.
"""

import os
import logging
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer, GrpcInstrumentorClient
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.jaeger import JaegerPropagator

logger = logging.getLogger(__name__)


class WorkerTracingConfig:
    """Configuration for worker distributed tracing."""
    
    def __init__(self, worker_id: str):
        self.enabled = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.service_name = os.getenv("SERVICE_NAME", f"mlx-worker-{worker_id}")
        self.service_version = os.getenv("SERVICE_VERSION", "1.0.0")
        self.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://jaeger:14268/api/traces")
        self.sample_rate = float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))
        self.worker_id = worker_id
        
        # Environment and deployment info
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.deployment_version = os.getenv("DEPLOYMENT_VERSION", "unknown")


def setup_worker_tracing(worker_id: str, config: Optional[WorkerTracingConfig] = None) -> Optional[trace.Tracer]:
    """Set up distributed tracing for the worker service."""
    if config is None:
        config = WorkerTracingConfig(worker_id)
    
    if not config.enabled:
        logger.info(f"Distributed tracing is disabled for worker {worker_id}")
        return None
    
    try:
        # Create resource with service information
        resource = Resource.create({
            "service.name": config.service_name,
            "service.version": config.service_version,
            "service.environment": config.environment,
            "deployment.version": config.deployment_version,
            "worker.id": worker_id,
        })
        
        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            endpoint=config.jaeger_endpoint,
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Set up propagators for cross-service tracing
        set_global_textmap(JaegerPropagator())
        
        # Instrument gRPC server and client
        GrpcInstrumentorServer().instrument()
        GrpcInstrumentorClient().instrument()
        
        # Get tracer instance
        tracer = trace.get_tracer(__name__)
        
        logger.info(f"Distributed tracing initialized for worker {worker_id}")
        logger.info(f"Jaeger endpoint: {config.jaeger_endpoint}")
        
        return tracer
        
    except Exception as e:
        logger.error(f"Failed to initialize worker tracing: {e}")
        return None


def get_current_span() -> Optional[trace.Span]:
    """Get the current active span."""
    return trace.get_current_span()


def add_span_attributes(attributes: dict):
    """Add attributes to the current span."""
    span = get_current_span()
    if span and span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[dict] = None):
    """Add an event to the current span."""
    span = get_current_span()
    if span and span.is_recording():
        span.add_event(name, attributes or {})


def record_exception(exception: Exception):
    """Record an exception in the current span."""
    span = get_current_span()
    if span and span.is_recording():
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))


def trace_model_loading(model_path: str, model_size_mb: float, load_time_ms: float):
    """Add model loading tracing attributes."""
    add_span_attributes({
        "mlx.model_path": model_path,
        "mlx.model_size_mb": model_size_mb,
        "mlx.load_time_ms": load_time_ms,
        "mlx.operation": "model_loading"
    })


def trace_inference_request(request_id: str, model: str, prompt_tokens: int, 
                           max_tokens: int, temperature: float, top_p: float):
    """Add inference request tracing attributes."""
    add_span_attributes({
        "mlx.request_id": request_id,
        "mlx.model": model,
        "mlx.prompt_tokens": prompt_tokens,
        "mlx.max_tokens": max_tokens,
        "mlx.temperature": temperature,
        "mlx.top_p": top_p,
        "mlx.operation": "inference"
    })


def trace_inference_completion(completion_tokens: int, total_time_ms: float, 
                              tokens_per_second: float, memory_used_mb: float):
    """Add inference completion tracing attributes."""
    add_span_attributes({
        "mlx.completion_tokens": completion_tokens,
        "mlx.total_time_ms": total_time_ms,
        "mlx.tokens_per_second": tokens_per_second,
        "mlx.memory_used_mb": memory_used_mb,
        "mlx.operation": "inference_completion"
    })


def trace_streaming_inference(request_id: str, chunk_count: int, total_tokens: int, 
                             streaming_duration_ms: float):
    """Add streaming inference tracing attributes."""
    add_span_attributes({
        "mlx.request_id": request_id,
        "mlx.streaming_chunks": chunk_count,
        "mlx.total_tokens": total_tokens,
        "mlx.streaming_duration_ms": streaming_duration_ms,
        "mlx.operation": "streaming_inference"
    })


def trace_coordinator_communication(operation: str, response_time_ms: float, success: bool):
    """Add coordinator communication tracing attributes."""
    add_span_attributes({
        "mlx.coordinator_operation": operation,
        "mlx.response_time_ms": response_time_ms,
        "mlx.success": success,
        "mlx.operation": "coordinator_communication"
    })


def trace_health_check(cpu_usage: float, memory_usage_mb: float, gpu_memory_usage_mb: float,
                      active_requests: int):
    """Add health check tracing attributes."""
    add_span_attributes({
        "mlx.cpu_usage_percent": cpu_usage,
        "mlx.memory_usage_mb": memory_usage_mb,
        "mlx.gpu_memory_usage_mb": gpu_memory_usage_mb,
        "mlx.active_requests": active_requests,
        "mlx.operation": "health_check"
    })


# Context manager for custom spans
class custom_span:
    """Context manager for creating custom spans in worker."""
    
    def __init__(self, name: str, attributes: Optional[dict] = None):
        self.name = name
        self.attributes = attributes or {}
        self.span = None
        self.tracer = trace.get_tracer(__name__)
    
    def __enter__(self):
        self.span = self.tracer.start_span(self.name)
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.span:
            self.span.record_exception(exc_val)
            self.span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
        if self.span:
            self.span.end()


class WorkerTracingGRPCInterceptor:
    """gRPC interceptor for enhanced tracing functionality in worker."""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.tracer = trace.get_tracer(__name__)
    
    def intercept_service(self, continuation, handler_call_details):
        """Intercept gRPC service calls for enhanced tracing."""
        method = handler_call_details.method
        
        def tracing_wrapper(request, context):
            with self.tracer.start_span(f"grpc.{method}") as span:
                try:
                    # Add method-specific attributes
                    span.set_attribute("rpc.method", method)
                    span.set_attribute("rpc.service", "InferenceService")
                    span.set_attribute("mlx.worker_id", self.worker_id)
                    
                    # Add request-specific attributes based on method
                    if hasattr(request, 'request_id'):
                        span.set_attribute("mlx.request_id", request.request_id)
                    if hasattr(request, 'model'):
                        span.set_attribute("mlx.model", request.model)
                    if hasattr(request, 'prompt'):
                        span.set_attribute("mlx.prompt_length", len(request.prompt))
                    if hasattr(request, 'max_tokens'):
                        span.set_attribute("mlx.max_tokens", request.max_tokens)
                    
                    response = continuation(request, context)
                    
                    # Add response attributes
                    span.set_attribute("rpc.grpc.status_code", context.code().value[0])
                    if hasattr(response, 'tokens_per_second'):
                        span.set_attribute("mlx.tokens_per_second", response.tokens_per_second)
                    if hasattr(response, 'completion_tokens'):
                        span.set_attribute("mlx.completion_tokens", response.completion_tokens)
                    
                    return response
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        
        return tracing_wrapper


def trace_mlx_operation(operation_name: str, model: str, input_tokens: int, 
                       output_tokens: int, duration_ms: float):
    """Add MLX-specific operation tracing."""
    with custom_span(f"mlx.{operation_name}") as span:
        span.set_attribute("mlx.operation", operation_name)
        span.set_attribute("mlx.model", model)
        span.set_attribute("mlx.input_tokens", input_tokens)
        span.set_attribute("mlx.output_tokens", output_tokens)
        span.set_attribute("mlx.duration_ms", duration_ms)
        span.set_attribute("mlx.tokens_per_second", output_tokens / (duration_ms / 1000) if duration_ms > 0 else 0)