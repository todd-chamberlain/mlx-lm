"""
Data models for the coordinator service.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import threading


@dataclass
class InferenceResult:
    """Result from inference processing."""
    
    request_id: str
    generated_text: str
    tokens_generated: int
    processing_time_ms: float
    success: bool
    error: Optional[str] = None


class WorkerStatus(Enum):
    """Worker status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass
class DeviceCapabilities:
    """Device capabilities information."""
    model: str
    memory_gb: int
    gpu_cores: int
    os_version: str
    mlx_version: str
    
    @classmethod
    def from_proto(cls, device_info) -> 'DeviceCapabilities':
        """Create from protobuf DeviceInfo message."""
        return cls(
            model=device_info.model,
            memory_gb=device_info.memory_gb,
            gpu_cores=device_info.gpu_cores,
            os_version=device_info.os_version,
            mlx_version=device_info.mlx_version
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model': self.model,
            'memory_gb': self.memory_gb,
            'gpu_cores': self.gpu_cores,
            'os_version': self.os_version,
            'mlx_version': self.mlx_version
        }


@dataclass
class WorkerMetrics:
    """Worker performance metrics."""
    avg_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    active_requests: int = 0
    memory_usage_gb: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    @classmethod
    def from_proto(cls, metrics) -> 'WorkerMetrics':
        """Create from protobuf WorkerMetrics message."""
        return cls(
            avg_latency_ms=metrics.avg_latency_ms,
            tokens_per_second=metrics.tokens_per_second,
            total_requests=metrics.total_requests,
            failed_requests=metrics.failed_requests,
            active_requests=metrics.active_requests,
            memory_usage_gb=metrics.memory_usage_gb,
            last_updated=time.time()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'avg_latency_ms': self.avg_latency_ms,
            'tokens_per_second': self.tokens_per_second,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'active_requests': self.active_requests,
            'memory_usage_gb': self.memory_usage_gb,
            'last_updated': self.last_updated
        }
    
    def update_from_proto(self, metrics) -> None:
        """Update metrics from protobuf message."""
        self.avg_latency_ms = metrics.avg_latency_ms
        self.tokens_per_second = metrics.tokens_per_second
        self.total_requests = metrics.total_requests
        self.failed_requests = metrics.failed_requests
        self.active_requests = metrics.active_requests
        self.memory_usage_gb = metrics.memory_usage_gb
        self.last_updated = time.time()


@dataclass
class WorkerInfo:
    """Complete worker information and state."""
    worker_id: str
    address: str
    device_capabilities: DeviceCapabilities
    max_concurrent: int
    supported_models: List[str]
    
    # Runtime state
    status: WorkerStatus = WorkerStatus.HEALTHY
    current_load: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    registration_time: float = field(default_factory=time.time)
    
    # Performance metrics
    metrics: WorkerMetrics = field(default_factory=WorkerMetrics)
    
    # Circuit breaker state
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    circuit_breaker_open: bool = False
    
    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    
    @classmethod
    def from_registration(cls, registration) -> 'WorkerInfo':
        """Create WorkerInfo from registration request."""
        device_capabilities = DeviceCapabilities.from_proto(registration.device_info)
        
        return cls(
            worker_id=registration.worker_id,
            address=registration.address,
            device_capabilities=device_capabilities,
            max_concurrent=registration.max_concurrent,
            supported_models=list(registration.supported_models)
        )
    
    def update_heartbeat(self, heartbeat_request) -> None:
        """Update worker state from heartbeat."""
        with self._lock:
            self.last_heartbeat = time.time()
            
            # Update status from heartbeat
            if heartbeat_request.status:
                status_str = heartbeat_request.status.status.lower()
                if status_str in [s.value for s in WorkerStatus]:
                    self.status = WorkerStatus(status_str)
                
                self.current_load = heartbeat_request.status.current_load
            
            # Update metrics from heartbeat
            if heartbeat_request.metrics:
                self.metrics.update_from_proto(heartbeat_request.metrics)
    
    def is_healthy(self, heartbeat_timeout: float = 30.0) -> bool:
        """Check if worker is considered healthy."""
        with self._lock:
            # Check heartbeat timeout
            if time.time() - self.last_heartbeat > heartbeat_timeout:
                return False
            
            # Check status
            if self.status not in [WorkerStatus.HEALTHY]:
                return False
            
            # Check circuit breaker
            if self.circuit_breaker_open:
                return False
            
            return True
    
    def is_available_for_request(self) -> bool:
        """Check if worker can accept new requests."""
        with self._lock:
            if not self.is_healthy():
                return False
            
            # Check load capacity
            if self.current_load >= self.max_concurrent:
                return False
            
            return True
    
    def calculate_score(self, load_weight: float = 0.5, latency_weight: float = 0.3, reliability_weight: float = 0.2) -> float:
        """Calculate worker selection score (lower is better)."""
        with self._lock:
            if not self.is_available_for_request():
                return float('inf')
            
            # Load factor (0.0 to 1.0)
            load_factor = self.current_load / max(self.max_concurrent, 1)
            
            # Latency factor (normalized to 0-1 range, assuming max 1000ms)
            latency_factor = min(self.metrics.avg_latency_ms / 1000.0, 1.0)
            
            # Reliability factor (failure rate)
            total_requests = max(self.metrics.total_requests, 1)
            reliability_factor = self.metrics.failed_requests / total_requests
            
            # Weighted score
            score = (
                load_factor * load_weight +
                latency_factor * latency_weight +
                reliability_factor * reliability_weight
            )
            
            return score
    
    def record_request_start(self) -> None:
        """Record that a request has started on this worker."""
        with self._lock:
            self.current_load += 1
    
    def record_request_complete(self, success: bool = True, latency_ms: float = 0.0) -> None:
        """Record that a request has completed on this worker."""
        with self._lock:
            self.current_load = max(0, self.current_load - 1)
            
            # Update metrics
            self.metrics.total_requests += 1
            if not success:
                self.metrics.failed_requests += 1
                self.consecutive_failures += 1
                self.last_failure_time = time.time()
            else:
                self.consecutive_failures = 0
                self.last_failure_time = None
            
            # Update latency (simple moving average)
            if latency_ms > 0:
                alpha = 0.1  # Smoothing factor
                self.metrics.avg_latency_ms = (
                    alpha * latency_ms + 
                    (1 - alpha) * self.metrics.avg_latency_ms
                )
    
    def update_circuit_breaker(self, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        """Update circuit breaker state."""
        with self._lock:
            if self.consecutive_failures >= failure_threshold:
                self.circuit_breaker_open = True
            elif (self.circuit_breaker_open and 
                  self.last_failure_time and 
                  time.time() - self.last_failure_time > recovery_timeout):
                self.circuit_breaker_open = False
                self.consecutive_failures = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        with self._lock:
            return {
                'worker_id': self.worker_id,
                'address': self.address,
                'device_capabilities': self.device_capabilities.to_dict(),
                'max_concurrent': self.max_concurrent,
                'supported_models': self.supported_models,
                'status': self.status.value,
                'current_load': self.current_load,
                'last_heartbeat': self.last_heartbeat,
                'registration_time': self.registration_time,
                'metrics': self.metrics.to_dict(),
                'consecutive_failures': self.consecutive_failures,
                'circuit_breaker_open': self.circuit_breaker_open,
                'is_healthy': self.is_healthy(),
                'is_available': self.is_available_for_request(),
                'score': self.calculate_score()
            }