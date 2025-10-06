"""
Distributed tracing setup for the API server using OpenTelemetry and Jaeger.
"""

import os
import logging
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.jaeger import JaegerPropagator

logger = logging.getLogger(__name__)


class TracingConfig:
    """Configuration for distributed tracing."""
    
    def __init__(self):
        self.enabled = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.service_name = os.getenv("SERVICE_NAME", "mlx-api-server")
        self.service_version = os.getenv("SERVICE_VERSION", "1.0.0")
        self.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://jaeger:14268/api/traces")
        self.sample_rate = float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))
        
        # Environment and deployment info
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.deployment_version = os.getenv("DEPLOYMENT_VERSION", "unknown")


def setup_tracing(app, config: Optional[TracingConfig] = None) -> Optional[trace.Tracer]:
    """Set up distributed tracing for the FastAPI application."""
    if config is None:
        config = TracingConfig()
    
    if not config.enabled:
        logger.info("Distributed tracing is disabled")
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
        
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        
        # Instrument HTTP clients
        HTTPXClientInstrumentor().instrument()
        
        # Instrument gRPC clients
        GrpcInstrumentorClient().instrument()
        
        # Get tracer instance
        tracer = trace.get_tracer(__name__)
        
        logger.info(f"Distributed tracing initialized for {config.service_name}")
        logger.info(f"Jaeger endpoint: {config.jaeger_endpoint}")
        logger.info(f"Sample rate: {config.sample_rate}")
        
        return tracer
        
    except Exception as e:
        logger.error(f"Failed to initialize distributed tracing: {e}")
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


class TracingMiddleware:
    """Custom middleware for additional tracing functionality."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Add custom attributes to the span
        span = get_current_span()
        if span and span.is_recording():
            # Add request information
            span.set_attribute("http.user_agent", 
                             scope.get("headers", {}).get(b"user-agent", b"").decode())
            span.set_attribute("http.client_ip", 
                             scope.get("client", ["unknown", None])[0])
            
            # Add custom business logic attributes
            if scope["path"].startswith("/v1/chat/completions"):
                span.set_attribute("mlx.request_type", "inference")
            elif scope["path"].startswith("/health"):
                span.set_attribute("mlx.request_type", "health_check")
            elif scope["path"].startswith("/metrics"):
                span.set_attribute("mlx.request_type", "metrics")
        
        await self.app(scope, receive, send)


def trace_inference_request(model: str, prompt_length: int, max_tokens: int):
    """Add inference-specific tracing attributes."""
    add_span_attributes({
        "mlx.model": model,
        "mlx.prompt_length": prompt_length,
        "mlx.max_tokens": max_tokens,
        "mlx.operation": "inference"
    })


def trace_worker_selection(worker_count: int, selected_worker: str, selection_time_ms: float):
    """Add worker selection tracing attributes."""
    add_span_attributes({
        "mlx.available_workers": worker_count,
        "mlx.selected_worker": selected_worker,
        "mlx.selection_time_ms": selection_time_ms,
        "mlx.operation": "worker_selection"
    })


def trace_streaming_response(chunk_count: int, total_tokens: int, duration_ms: float):
    """Add streaming response tracing attributes."""
    add_span_attributes({
        "mlx.streaming_chunks": chunk_count,
        "mlx.total_tokens": total_tokens,
        "mlx.streaming_duration_ms": duration_ms,
        "mlx.operation": "streaming"
    })


# Context manager for custom spans
class custom_span:
    """Context manager for creating custom spans."""
    
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