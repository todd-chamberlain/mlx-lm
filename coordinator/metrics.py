"""
Prometheus metrics collection for the coordinator service.
"""

import time
from typing import Dict, Optional
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
import logging

logger = logging.getLogger(__name__)

# gRPC request metrics
GRPC_REQUESTS = Counter(
    'coordinator_grpc_requests_total',
    'Total number of gRPC requests',
    ['method', 'status']
)

GRPC_REQUEST_DURATION = Histogram(
    'coordinator_grpc_request_duration_seconds',
    'gRPC request duration in seconds',
    ['method'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Worker management metrics
WORKER_REGISTRATIONS = Counter(
    'worker_registrations_total',
    'Total number of worker registrations',
    ['status']
)

WORKER_HEARTBEATS = Counter(
    'worker_heartbeats_total',
    'Total number of worker heartbeats',
    ['worker_id', 'status']
)

ACTIVE_WORKERS = Gauge(
    'coordinator_active_workers',
    'Number of active workers'
)

WORKER_HEALTH_STATUS = Gauge(
    'worker_health_status',
    'Worker health status (1=healthy, 0=unhealthy)',
    ['worker_id', 'address']
)

# Request routing metrics
INFERENCE_ROUTING = Counter(
    'inference_routing_total',
    'Total number of inference requests routed',
    ['worker_id', 'status']
)

ROUTING_DURATION = Histogram(
    'routing_duration_seconds',
    'Time spent routing requests to workers',
    ['worker_id'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

WORKER_SELECTION_TIME = Histogram(
    'worker_selection_duration_seconds',
    'Time spent selecting optimal worker',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1]
)

# Worker performance metrics
WORKER_LOAD = Gauge(
    'worker_current_load',
    'Current load on each worker',
    ['worker_id']
)

WORKER_LATENCY = Gauge(
    'worker_average_latency_seconds',
    'Average latency for each worker',
    ['worker_id']
)

WORKER_SUCCESS_RATE = Gauge(
    'worker_success_rate',
    'Success rate for each worker (0-1)',
    ['worker_id']
)

# Queue and backlog metrics
PENDING_REQUESTS = Gauge(
    'pending_requests',
    'Number of requests waiting to be processed'
)

QUEUE_WAIT_TIME = Histogram(
    'queue_wait_time_seconds',
    'Time requests spend waiting in queue',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Circuit breaker metrics
CIRCUIT_BREAKER_STATE = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['worker_id']
)

CIRCUIT_BREAKER_FAILURES = Counter(
    'circuit_breaker_failures_total',
    'Total circuit breaker failures',
    ['worker_id']
)

# Application info
APP_INFO = Info(
    'coordinator_info',
    'Coordinator service information'
)

class CoordinatorMetrics:
    """Centralized metrics collection for the coordinator service."""
    
    def __init__(self):
        self.start_time = time.time()
        APP_INFO.info({
            'version': '1.0.0',
            'component': 'coordinator',
            'start_time': str(int(self.start_time))
        })
    
    def record_grpc_request(self, method: str, status: str, duration: float):
        """Record gRPC request metrics."""
        GRPC_REQUESTS.labels(method=method, status=status).inc()
        GRPC_REQUEST_DURATION.labels(method=method).observe(duration)
    
    def record_worker_registration(self, status: str):
        """Record worker registration attempt."""
        WORKER_REGISTRATIONS.labels(status=status).inc()
    
    def record_worker_heartbeat(self, worker_id: str, status: str):
        """Record worker heartbeat."""
        WORKER_HEARTBEATS.labels(worker_id=worker_id, status=status).inc()
    
    def update_worker_metrics(self, worker_id: str, address: str, is_healthy: bool, 
                            current_load: int, avg_latency: float, success_rate: float):
        """Update worker-specific metrics."""
        WORKER_HEALTH_STATUS.labels(worker_id=worker_id, address=address).set(
            1 if is_healthy else 0
        )
        WORKER_LOAD.labels(worker_id=worker_id).set(current_load)
        WORKER_LATENCY.labels(worker_id=worker_id).set(avg_latency)
        WORKER_SUCCESS_RATE.labels(worker_id=worker_id).set(success_rate)
    
    def record_inference_routing(self, worker_id: str, status: str, duration: float):
        """Record inference request routing."""
        INFERENCE_ROUTING.labels(worker_id=worker_id, status=status).inc()
        ROUTING_DURATION.labels(worker_id=worker_id).observe(duration)
    
    def record_worker_selection_time(self, duration: float):
        """Record time spent selecting worker."""
        WORKER_SELECTION_TIME.observe(duration)
    
    def update_queue_metrics(self, pending_count: int, wait_time: Optional[float] = None):
        """Update queue-related metrics."""
        PENDING_REQUESTS.set(pending_count)
        if wait_time is not None:
            QUEUE_WAIT_TIME.observe(wait_time)
    
    def update_circuit_breaker_state(self, worker_id: str, state: int):
        """Update circuit breaker state (0=closed, 1=open, 2=half-open)."""
        CIRCUIT_BREAKER_STATE.labels(worker_id=worker_id).set(state)
    
    def record_circuit_breaker_failure(self, worker_id: str):
        """Record circuit breaker failure."""
        CIRCUIT_BREAKER_FAILURES.labels(worker_id=worker_id).inc()
    
    def update_active_workers_count(self, count: int):
        """Update the count of active workers."""
        ACTIVE_WORKERS.set(count)

# Global metrics collector instance
coordinator_metrics = CoordinatorMetrics()

def get_metrics_data() -> bytes:
    """Generate Prometheus metrics data."""
    try:
        return generate_latest()
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return b"# Error generating metrics\n"

class GRPCMetricsInterceptor:
    """gRPC interceptor for automatic metrics collection."""
    
    def __init__(self, metrics: CoordinatorMetrics):
        self.metrics = metrics
    
    def intercept_service(self, continuation, handler_call_details):
        """Intercept gRPC service calls for metrics collection."""
        start_time = time.time()
        method = handler_call_details.method
        
        def metrics_wrapper(request, context):
            try:
                response = continuation(request, context)
                duration = time.time() - start_time
                self.metrics.record_grpc_request(method, "success", duration)
                return response
            except Exception as e:
                duration = time.time() - start_time
                self.metrics.record_grpc_request(method, "error", duration)
                raise
        
        return metrics_wrapper