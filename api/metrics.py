"""
Prometheus metrics collection for the API server.
"""

import time
from typing import Dict, Optional
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import logging

logger = logging.getLogger(__name__)

# Request metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Inference metrics
INFERENCE_REQUESTS = Counter(
    'inference_requests_total',
    'Total number of inference requests',
    ['model', 'status']
)

INFERENCE_DURATION = Histogram(
    'inference_duration_seconds',
    'Inference request duration in seconds',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

TOKENS_GENERATED = Counter(
    'tokens_generated_total',
    'Total number of tokens generated',
    ['model']
)

TOKENS_PER_SECOND = Histogram(
    'tokens_per_second',
    'Tokens generated per second',
    ['model'],
    buckets=[1, 5, 10, 20, 30, 50, 75, 100]
)

# System metrics
ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

COORDINATOR_HEALTH = Gauge(
    'coordinator_health',
    'Coordinator service health status (1=healthy, 0=unhealthy)'
)

# Worker metrics
AVAILABLE_WORKERS = Gauge(
    'available_workers_total',
    'Total number of available workers'
)

WORKER_REQUESTS = Counter(
    'worker_requests_total',
    'Total requests sent to workers',
    ['worker_id', 'status']
)

# Application info
APP_INFO = Info(
    'api_server_info',
    'API server information'
)

class MetricsCollector:
    """Centralized metrics collection for the API server."""
    
    def __init__(self):
        self.start_time = time.time()
        APP_INFO.info({
            'version': '1.0.0',
            'component': 'api-server',
            'start_time': str(int(self.start_time))
        })
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics."""
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_inference(self, model: str, status: str, duration: float, 
                        tokens_generated: int = 0, tokens_per_sec: float = 0):
        """Record inference request metrics."""
        INFERENCE_REQUESTS.labels(model=model, status=status).inc()
        INFERENCE_DURATION.labels(model=model).observe(duration)
        
        if tokens_generated > 0:
            TOKENS_GENERATED.labels(model=model).inc(tokens_generated)
        
        if tokens_per_sec > 0:
            TOKENS_PER_SECOND.labels(model=model).observe(tokens_per_sec)
    
    def record_worker_request(self, worker_id: str, status: str):
        """Record worker request metrics."""
        WORKER_REQUESTS.labels(worker_id=worker_id, status=status).inc()
    
    def update_system_metrics(self, active_connections: int, coordinator_healthy: bool, 
                            available_workers: int):
        """Update system-level metrics."""
        ACTIVE_CONNECTIONS.set(active_connections)
        COORDINATOR_HEALTH.set(1 if coordinator_healthy else 0)
        AVAILABLE_WORKERS.set(available_workers)

# Global metrics collector instance
metrics_collector = MetricsCollector()

def get_metrics_response() -> Response:
    """Generate Prometheus metrics response."""
    try:
        metrics_data = generate_latest()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return Response(
            content="# Error generating metrics\n",
            media_type=CONTENT_TYPE_LATEST,
            status_code=500
        )

class MetricsMiddleware:
    """FastAPI middleware for automatic metrics collection."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        method = scope["method"]
        path = scope["path"]
        
        # Normalize endpoint paths for metrics
        endpoint = self._normalize_endpoint(path)
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                duration = time.time() - start_time
                
                # Record metrics
                metrics_collector.record_request(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code,
                    duration=duration
                )
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint paths to reduce cardinality."""
        # Remove query parameters
        if "?" in path:
            path = path.split("?")[0]
        
        # Normalize common patterns
        if path.startswith("/v1/chat/completions"):
            return "/v1/chat/completions"
        elif path.startswith("/health"):
            return "/health"
        elif path.startswith("/metrics"):
            return "/metrics"
        elif path.startswith("/models"):
            return "/models"
        else:
            return path

def setup_metrics(app):
    """Set up metrics collection for the FastAPI application."""
    from fastapi import FastAPI
    
    # Add metrics middleware
    app.add_middleware(MetricsMiddleware)
    
    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        return get_metrics_response()