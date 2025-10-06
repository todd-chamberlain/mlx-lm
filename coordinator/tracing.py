"""
Distributed tracing setup for the coordinator service using OpenTelemetry and Jaeger.
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


class CoordinatorTracingConfig:
    """Configuration for coordinator distributed tracing."""
    
    def __init__(self):
        self.enabled = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.service_name = os.getenv("SERVICE_NAME", "mlx-coordinator")
        self.service_version = os.getenv("SERVICE_VERSION", "1.0.0")
        self.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://jaeger:14268/api/traces")
        self.sample_rate = float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))
        
        # Environment and deployment info
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.deployment_version = os.getenv("DEPLOYMENT_VERSION", "unknown")


def setup_coordinator_tracing(config: Optional[CoordinatorTracingConfig] = None) -> Optional[trace.Tracer]:
    """Set up distributed tracing for the coordinator service."""
    if config is None:
        config = CoordinatorTracingConfig()
    
    if not config.enabled:
        logger.info("Distributed tracing is disabled for coordinator")
        return None
    
    try:
        # Create resource with service information
        resource = Resource.create({
            "service.name": config.service_name,
            "service.version": config.service_version,
            "service.environment": config.environment,
            "deployment.version": config.deployment_version,
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
        
        logger.info(f"Distributed tracing initialized for {config.service_name}")
        logger.info(f"Jaeger endpoint: {config.jaeger_endpoint}")
        
        return tracer
        
    except Exception as e:
        logger.error(f"Failed to initialize coordinator tracing: {e}")
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


def trace_worker_registration(worker_id: str, address: str, device_info: dict):
    """Add worker registration tracing attributes."""
    add_span_attributes({
        "mlx.worker_id": worker_id,
        "mlx.worker_address": address,
        "mlx.device_model": device_info.get("model", "unknown"),
        "mlx.device_memory_gb": device_info.get("memory_gb", 0),
        "mlx.device_gpu_cores": device_info.get("gpu_cores", 0),
        "mlx.operation": "worker_registration"
    })


def trace_worker_selection(available_workers: int, selected_worker: str, 
                          selection_criteria: dict, selection_time_ms: float):
    """Add worker selection tracing attributes."""
    add_span_attributes({
        "mlx.available_workers": available_workers,
        "mlx.selected_worker": selected_worker,
        "mlx.selection_time_ms": selection_time_ms,
        "mlx.selection_load_factor": selection_criteria.get("load_factor", 0),
        "mlx.selection_latency_factor": selection_criteria.get("latency_factor", 0),
        "mlx.selection_reliability_factor": selection_criteria.get("reliability_factor", 0),
        "mlx.operation": "worker_selection"
    })


def trace_request_routing(request_id: str, worker_id: str, model: str, 
                         routing_time_ms: float, retry_count: int = 0):
    """Add request routing tracing attributes."""
    add_span_attributes({
        "mlx.request_id": request_id,
        "mlx.target_worker": worker_id,
        "mlx.model": model,
        "mlx.routing_time_ms": routing_time_ms,
        "mlx.retry_count": retry_count,
        "mlx.operation": "request_routing"
    })


def trace_heartbeat(worker_id: str, status: str, response_time_ms: float):
    """Add heartbeat tracing attributes."""
    add_span_attributes({
        "mlx.worker_id": worker_id,
        "mlx.heartbeat_status": status,
        "mlx.response_time_ms": response_time_ms,
        "mlx.operation": "heartbeat"
    })


def trace_health_check(total_workers: int, healthy_workers: int, active_requests: int):
    """Add health check tracing attributes."""
    add_span_attributes({
        "mlx.total_workers": total_workers,
        "mlx.healthy_workers": healthy_workers,
        "mlx.active_requests": active_requests,
        "mlx.operation": "health_check"
    })


# Context manager for custom spans
class custom_span:
    """Context manager for creating custom spans in coordinator."""
    
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


class TracingGRPCInterceptor:
    """gRPC interceptor for enhanced tracing functionality."""
    
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
    
    def intercept_service(self, continuation, handler_call_details):
        """Intercept gRPC service calls for enhanced tracing."""
        method = handler_call_details.method
        
        def tracing_wrapper(request, context):
            with self.tracer.start_span(f"grpc.{method}") as span:
                try:
                    # Add method-specific attributes
                    span.set_attribute("rpc.method", method)
                    span.set_attribute("rpc.service", "CoordinatorService")
                    
                    # Add request-specific attributes based on method
                    if hasattr(request, 'worker_id'):
                        span.set_attribute("mlx.worker_id", request.worker_id)
                    if hasattr(request, 'request_id'):
                        span.set_attribute("mlx.request_id", request.request_id)
                    if hasattr(request, 'model'):
                        span.set_attribute("mlx.model", request.model)
                    
                    response = continuation(request, context)
                    
                    # Add response attributes
                    span.set_attribute("rpc.grpc.status_code", context.code().value[0])
                    
                    return response
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        
        return tracing_wrapper