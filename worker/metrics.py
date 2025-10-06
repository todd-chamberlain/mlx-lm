"""
Prometheus metrics collection for the MLX worker service.
"""

import time
import psutil
from typing import Dict, Optional
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
import logging

logger = logging.getLogger(__name__)

# Inference metrics
INFERENCE_REQUESTS = Counter(
    'worker_inference_requests_total',
    'Total number of inference requests processed',
    ['model', 'status']
)

INFERENCE_DURATION = Histogram(
    'worker_inference_duration_seconds',
    'Inference processing duration in seconds',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

TOKENS_GENERATED = Counter(
    'worker_tokens_generated_total',
    'Total number of tokens generated',
    ['model']
)

TOKENS_PER_SECOND = Histogram(
    'worker_tokens_per_second',
    'Tokens generated per second',
    ['model'],
    buckets=[1, 5, 10, 20, 30, 50, 75, 100, 150]
)

PROMPT_TOKENS = Histogram(
    'worker_prompt_tokens',
    'Number of tokens in prompts',
    ['model'],
    buckets=[10, 50, 100, 250, 500, 1000, 2000, 4000]
)

COMPLETION_TOKENS = Histogram(
    'worker_completion_tokens',
    'Number of tokens in completions',
    ['model'],
    buckets=[10, 50, 100, 250, 500, 1000, 2000]
)

# Model management metrics
MODEL_LOAD_TIME = Histogram(
    'model_load_duration_seconds',
    'Time taken to load models',
    ['model'],
    buckets=[1, 5, 10, 30, 60, 120, 300]
)

MODEL_MEMORY_USAGE = Gauge(
    'model_memory_usage_bytes',
    'Memory usage by loaded models',
    ['model']
)

ACTIVE_MODELS = Gauge(
    'active_models_count',
    'Number of currently loaded models'
)

# System resource metrics
CPU_USAGE = Gauge(
    'worker_cpu_usage_percent',
    'CPU usage percentage'
)

MEMORY_USAGE = Gauge(
    'worker_memory_usage_bytes',
    'Memory usage in bytes'
)

MEMORY_AVAILABLE = Gauge(
    'worker_memory_available_bytes',
    'Available memory in bytes'
)

GPU_MEMORY_USAGE = Gauge(
    'worker_gpu_memory_usage_bytes',
    'GPU memory usage in bytes (Metal)'
)

GPU_MEMORY_AVAILABLE = Gauge(
    'worker_gpu_memory_available_bytes',
    'Available GPU memory in bytes (Metal)'
)

# Request handling metrics
CONCURRENT_REQUESTS = Gauge(
    'worker_concurrent_requests',
    'Number of concurrent requests being processed'
)

REQUEST_QUEUE_SIZE = Gauge(
    'worker_request_queue_size',
    'Number of requests waiting in queue'
)

REQUEST_QUEUE_WAIT_TIME = Histogram(
    'worker_request_queue_wait_seconds',
    'Time requests spend waiting in queue',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# gRPC metrics
GRPC_REQUESTS = Counter(
    'worker_grpc_requests_total',
    'Total number of gRPC requests',
    ['method', 'status']
)

GRPC_REQUEST_DURATION = Histogram(
    'worker_grpc_request_duration_seconds',
    'gRPC request duration in seconds',
    ['method'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# Streaming metrics
STREAMING_REQUESTS = Counter(
    'worker_streaming_requests_total',
    'Total number of streaming requests',
    ['model', 'status']
)

STREAMING_CHUNKS_SENT = Counter(
    'worker_streaming_chunks_sent_total',
    'Total number of streaming chunks sent',
    ['model']
)

STREAMING_DURATION = Histogram(
    'worker_streaming_duration_seconds',
    'Total duration of streaming requests',
    ['model'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# Health and heartbeat metrics
HEARTBEAT_SENT = Counter(
    'worker_heartbeats_sent_total',
    'Total number of heartbeats sent to coordinator',
    ['status']
)

COORDINATOR_CONNECTION_STATUS = Gauge(
    'worker_coordinator_connection_status',
    'Connection status to coordinator (1=connected, 0=disconnected)'
)

HEALTH_CHECK_DURATION = Histogram(
    'worker_health_check_duration_seconds',
    'Duration of health checks',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1]
)

# Error metrics
INFERENCE_ERRORS = Counter(
    'worker_inference_errors_total',
    'Total number of inference errors',
    ['model', 'error_type']
)

MODEL_LOAD_ERRORS = Counter(
    'worker_model_load_errors_total',
    'Total number of model loading errors',
    ['model', 'error_type']
)

# Application info
APP_INFO = Info(
    'worker_info',
    'Worker service information'
)

class WorkerMetrics:
    """Centralized metrics collection for the MLX worker service."""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.start_time = time.time()
        self._setup_app_info()
        
        # Initialize system metrics collection
        self._last_cpu_times = psutil.cpu_times()
        
    def _setup_app_info(self):
        """Set up application information metrics."""
        try:
            import platform
            import mlx
            
            APP_INFO.info({
                'version': '1.0.0',
                'component': 'mlx-worker',
                'worker_id': self.worker_id,
                'start_time': str(int(self.start_time)),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'mlx_version': mlx.__version__ if hasattr(mlx, '__version__') else 'unknown'
            })
        except Exception as e:
            logger.warning(f"Could not set up app info: {e}")
    
    def record_inference_request(self, model: str, status: str, duration: float,
                               prompt_tokens: int = 0, completion_tokens: int = 0,
                               tokens_per_sec: float = 0):
        """Record inference request metrics."""
        INFERENCE_REQUESTS.labels(model=model, status=status).inc()
        INFERENCE_DURATION.labels(model=model).observe(duration)
        
        if prompt_tokens > 0:
            PROMPT_TOKENS.labels(model=model).observe(prompt_tokens)
        
        if completion_tokens > 0:
            COMPLETION_TOKENS.labels(model=model).observe(completion_tokens)
            TOKENS_GENERATED.labels(model=model).inc(completion_tokens)
        
        if tokens_per_sec > 0:
            TOKENS_PER_SECOND.labels(model=model).observe(tokens_per_sec)
    
    def record_streaming_request(self, model: str, status: str, duration: float, chunks_sent: int):
        """Record streaming request metrics."""
        STREAMING_REQUESTS.labels(model=model, status=status).inc()
        STREAMING_DURATION.labels(model=model).observe(duration)
        STREAMING_CHUNKS_SENT.labels(model=model).inc(chunks_sent)
    
    def record_model_load(self, model: str, duration: float, memory_usage: int, success: bool):
        """Record model loading metrics."""
        if success:
            MODEL_LOAD_TIME.labels(model=model).observe(duration)
            MODEL_MEMORY_USAGE.labels(model=model).set(memory_usage)
        else:
            MODEL_LOAD_ERRORS.labels(model=model, error_type="load_failed").inc()
    
    def record_grpc_request(self, method: str, status: str, duration: float):
        """Record gRPC request metrics."""
        GRPC_REQUESTS.labels(method=method, status=status).inc()
        GRPC_REQUEST_DURATION.labels(method=method).observe(duration)
    
    def record_heartbeat(self, status: str):
        """Record heartbeat attempt."""
        HEARTBEAT_SENT.labels(status=status).inc()
    
    def record_inference_error(self, model: str, error_type: str):
        """Record inference error."""
        INFERENCE_ERRORS.labels(model=model, error_type=error_type).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            CPU_USAGE.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)
            MEMORY_AVAILABLE.set(memory.available)
            
            # GPU memory (placeholder - would need Metal-specific implementation)
            # This would require integration with Metal performance shaders
            # For now, we'll set placeholder values
            
        except Exception as e:
            logger.warning(f"Error updating system metrics: {e}")
    
    def update_request_metrics(self, concurrent_count: int, queue_size: int, 
                             queue_wait_time: Optional[float] = None):
        """Update request handling metrics."""
        CONCURRENT_REQUESTS.set(concurrent_count)
        REQUEST_QUEUE_SIZE.set(queue_size)
        
        if queue_wait_time is not None:
            REQUEST_QUEUE_WAIT_TIME.observe(queue_wait_time)
    
    def update_coordinator_connection(self, connected: bool):
        """Update coordinator connection status."""
        COORDINATOR_CONNECTION_STATUS.set(1 if connected else 0)
    
    def record_health_check(self, duration: float):
        """Record health check duration."""
        HEALTH_CHECK_DURATION.observe(duration)
    
    def update_active_models_count(self, count: int):
        """Update the count of active models."""
        ACTIVE_MODELS.set(count)

# Global metrics collector instance (will be initialized with worker_id)
worker_metrics: Optional[WorkerMetrics] = None

def initialize_worker_metrics(worker_id: str):
    """Initialize the global worker metrics instance."""
    global worker_metrics
    worker_metrics = WorkerMetrics(worker_id)
    return worker_metrics

def get_metrics_data() -> bytes:
    """Generate Prometheus metrics data."""
    try:
        # Update system metrics before generating output
        if worker_metrics:
            worker_metrics.update_system_metrics()
        
        return generate_latest()
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return b"# Error generating metrics\n"