"""
Health monitoring and heartbeat system for the coordinator service.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import threading

from .models import WorkerInfo, WorkerStatus
from proto.errors import MLXError, MLXErrorCode


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthThresholds:
    """Health monitoring thresholds."""
    heartbeat_timeout: float = 30.0  # seconds
    heartbeat_warning: float = 15.0  # seconds
    unhealthy_worker_ratio: float = 0.5  # 50% unhealthy = degraded
    critical_worker_ratio: float = 0.8  # 80% unhealthy = critical
    min_healthy_workers: int = 1
    max_consecutive_failures: int = 3
    failure_rate_threshold: float = 0.2  # 20% failure rate
    latency_warning_ms: float = 1000.0
    latency_critical_ms: float = 2000.0


@dataclass
class ClusterHealth:
    """Overall cluster health information."""
    status: HealthStatus
    total_workers: int
    healthy_workers: int
    unhealthy_workers: int
    degraded_workers: int
    total_capacity: int
    available_capacity: int
    active_requests: int
    average_latency_ms: float
    failure_rate: float
    uptime_seconds: float
    last_updated: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status.value,
            'total_workers': self.total_workers,
            'healthy_workers': self.healthy_workers,
            'unhealthy_workers': self.unhealthy_workers,
            'degraded_workers': self.degraded_workers,
            'total_capacity': self.total_capacity,
            'available_capacity': self.available_capacity,
            'active_requests': self.active_requests,
            'utilization': self.active_requests / max(self.total_capacity, 1),
            'average_latency_ms': self.average_latency_ms,
            'failure_rate': self.failure_rate,
            'uptime_seconds': self.uptime_seconds,
            'last_updated': self.last_updated
        }


@dataclass
class WorkerHealthInfo:
    """Individual worker health information."""
    worker_id: str
    status: WorkerStatus
    is_healthy: bool
    last_heartbeat: float
    time_since_heartbeat: float
    current_load: int
    max_concurrent: int
    utilization: float
    consecutive_failures: int
    failure_rate: float
    average_latency_ms: float
    tokens_per_second: float
    circuit_breaker_open: bool
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'worker_id': self.worker_id,
            'status': self.status.value,
            'is_healthy': self.is_healthy,
            'last_heartbeat': self.last_heartbeat,
            'time_since_heartbeat': self.time_since_heartbeat,
            'current_load': self.current_load,
            'max_concurrent': self.max_concurrent,
            'utilization': self.utilization,
            'consecutive_failures': self.consecutive_failures,
            'failure_rate': self.failure_rate,
            'average_latency_ms': self.average_latency_ms,
            'tokens_per_second': self.tokens_per_second,
            'circuit_breaker_open': self.circuit_breaker_open,
            'warnings': self.warnings
        }


class HealthMonitor:
    """Health monitoring system for workers and cluster."""
    
    def __init__(
        self,
        thresholds: Optional[HealthThresholds] = None,
        check_interval: float = 10.0,
        enable_auto_recovery: bool = True
    ):
        self.thresholds = thresholds or HealthThresholds()
        self.check_interval = check_interval
        self.enable_auto_recovery = enable_auto_recovery
        
        # State tracking
        self._start_time = time.time()
        self._last_health_check = 0.0
        self._cluster_health: Optional[ClusterHealth] = None
        self._worker_health: Dict[str, WorkerHealthInfo] = {}
        self._health_history: List[ClusterHealth] = []
        self._max_history_size = 100
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Event callbacks
        self._health_changed_callbacks: List[Callable[[ClusterHealth], None]] = []
        self._worker_unhealthy_callbacks: List[Callable[[str, WorkerHealthInfo], None]] = []
        self._worker_recovered_callbacks: List[Callable[[str, WorkerHealthInfo], None]] = []
        self._critical_alert_callbacks: List[Callable[[ClusterHealth], None]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("HealthMonitor initialized")
    
    async def start(self) -> None:
        """Start health monitoring."""
        logger.info("Starting health monitor")
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self) -> None:
        """Stop health monitoring."""
        logger.info("Stopping health monitor")
        self._shutdown_event.set()
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitor stopped")
    
    def check_cluster_health(self, workers: List[WorkerInfo]) -> ClusterHealth:
        """Check overall cluster health."""
        current_time = time.time()
        
        # Calculate worker health statistics
        total_workers = len(workers)
        healthy_workers = 0
        unhealthy_workers = 0
        degraded_workers = 0
        total_capacity = 0
        available_capacity = 0
        active_requests = 0
        total_latency = 0.0
        total_requests = 0
        total_failures = 0
        
        worker_health_info = {}
        
        for worker in workers:
            # Update individual worker health
            worker_health = self._check_worker_health(worker)
            worker_health_info[worker.worker_id] = worker_health
            
            # Aggregate statistics
            total_capacity += worker.max_concurrent
            available_capacity += max(0, worker.max_concurrent - worker.current_load)
            active_requests += worker.current_load
            
            if worker_health.is_healthy:
                healthy_workers += 1
            else:
                unhealthy_workers += 1
                if worker_health.time_since_heartbeat < self.thresholds.heartbeat_timeout:
                    degraded_workers += 1
            
            # Aggregate performance metrics
            if worker.metrics.total_requests > 0:
                total_latency += worker.metrics.avg_latency_ms * worker.metrics.total_requests
                total_requests += worker.metrics.total_requests
                total_failures += worker.metrics.failed_requests
        
        # Calculate averages
        average_latency_ms = total_latency / max(total_requests, 1)
        failure_rate = total_failures / max(total_requests, 1)
        uptime_seconds = current_time - self._start_time
        
        # Determine overall status
        status = self._determine_cluster_status(
            total_workers, healthy_workers, unhealthy_workers, failure_rate
        )
        
        # Create cluster health object
        cluster_health = ClusterHealth(
            status=status,
            total_workers=total_workers,
            healthy_workers=healthy_workers,
            unhealthy_workers=unhealthy_workers,
            degraded_workers=degraded_workers,
            total_capacity=total_capacity,
            available_capacity=available_capacity,
            active_requests=active_requests,
            average_latency_ms=average_latency_ms,
            failure_rate=failure_rate,
            uptime_seconds=uptime_seconds
        )
        
        # Update state
        with self._lock:
            old_health = self._cluster_health
            self._cluster_health = cluster_health
            self._worker_health = worker_health_info
            self._last_health_check = current_time
            
            # Add to history
            self._health_history.append(cluster_health)
            if len(self._health_history) > self._max_history_size:
                self._health_history.pop(0)
            
            # Trigger callbacks for status changes
            if old_health and old_health.status != cluster_health.status:
                logger.info(f"Cluster health status changed: {old_health.status.value} -> {cluster_health.status.value}")
                
                for callback in self._health_changed_callbacks:
                    try:
                        callback(cluster_health)
                    except Exception as e:
                        logger.error(f"Error in health changed callback: {e}")
                
                # Trigger critical alerts
                if cluster_health.status == HealthStatus.CRITICAL:
                    for callback in self._critical_alert_callbacks:
                        try:
                            callback(cluster_health)
                        except Exception as e:
                            logger.error(f"Error in critical alert callback: {e}")
        
        return cluster_health
    
    def _check_worker_health(self, worker: WorkerInfo) -> WorkerHealthInfo:
        """Check individual worker health."""
        current_time = time.time()
        time_since_heartbeat = current_time - worker.last_heartbeat
        
        # Determine if worker is healthy
        is_healthy = (
            time_since_heartbeat <= self.thresholds.heartbeat_timeout and
            worker.status == WorkerStatus.HEALTHY and
            not worker.circuit_breaker_open and
            worker.consecutive_failures <= self.thresholds.max_consecutive_failures
        )
        
        # Calculate metrics
        utilization = worker.current_load / max(worker.max_concurrent, 1)
        failure_rate = (
            worker.metrics.failed_requests / max(worker.metrics.total_requests, 1)
            if worker.metrics.total_requests > 0 else 0.0
        )
        
        # Generate warnings
        warnings = []
        if time_since_heartbeat > self.thresholds.heartbeat_warning:
            warnings.append(f"Heartbeat delayed by {time_since_heartbeat:.1f}s")
        
        if worker.metrics.avg_latency_ms > self.thresholds.latency_warning_ms:
            warnings.append(f"High latency: {worker.metrics.avg_latency_ms:.1f}ms")
        
        if failure_rate > self.thresholds.failure_rate_threshold:
            warnings.append(f"High failure rate: {failure_rate:.1%}")
        
        if utilization > 0.9:
            warnings.append(f"High utilization: {utilization:.1%}")
        
        if worker.circuit_breaker_open:
            warnings.append("Circuit breaker is open")
        
        # Create worker health info
        worker_health = WorkerHealthInfo(
            worker_id=worker.worker_id,
            status=worker.status,
            is_healthy=is_healthy,
            last_heartbeat=worker.last_heartbeat,
            time_since_heartbeat=time_since_heartbeat,
            current_load=worker.current_load,
            max_concurrent=worker.max_concurrent,
            utilization=utilization,
            consecutive_failures=worker.consecutive_failures,
            failure_rate=failure_rate,
            average_latency_ms=worker.metrics.avg_latency_ms,
            tokens_per_second=worker.metrics.tokens_per_second,
            circuit_breaker_open=worker.circuit_breaker_open,
            warnings=warnings
        )
        
        # Check for worker status changes
        with self._lock:
            old_worker_health = self._worker_health.get(worker.worker_id)
            
            if old_worker_health:
                # Worker became unhealthy
                if old_worker_health.is_healthy and not is_healthy:
                    logger.warning(f"Worker {worker.worker_id} became unhealthy")
                    for callback in self._worker_unhealthy_callbacks:
                        try:
                            callback(worker.worker_id, worker_health)
                        except Exception as e:
                            logger.error(f"Error in worker unhealthy callback: {e}")
                
                # Worker recovered
                elif not old_worker_health.is_healthy and is_healthy:
                    logger.info(f"Worker {worker.worker_id} recovered")
                    for callback in self._worker_recovered_callbacks:
                        try:
                            callback(worker.worker_id, worker_health)
                        except Exception as e:
                            logger.error(f"Error in worker recovered callback: {e}")
        
        return worker_health
    
    def _determine_cluster_status(
        self,
        total_workers: int,
        healthy_workers: int,
        unhealthy_workers: int,
        failure_rate: float
    ) -> HealthStatus:
        """Determine overall cluster health status."""
        if total_workers == 0:
            return HealthStatus.CRITICAL
        
        if healthy_workers < self.thresholds.min_healthy_workers:
            return HealthStatus.CRITICAL
        
        unhealthy_ratio = unhealthy_workers / total_workers
        
        if unhealthy_ratio >= self.thresholds.critical_worker_ratio:
            return HealthStatus.CRITICAL
        elif unhealthy_ratio >= self.thresholds.unhealthy_worker_ratio:
            return HealthStatus.DEGRADED
        elif failure_rate > self.thresholds.failure_rate_threshold:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        logger.info("Starting health monitoring loop")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.check_interval)
                
                # Health checking is done externally by calling check_cluster_health
                # This loop can be used for additional monitoring tasks
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
        
        logger.info("Health monitoring loop stopped")
    
    def get_cluster_health(self) -> Optional[ClusterHealth]:
        """Get current cluster health."""
        with self._lock:
            return self._cluster_health
    
    def get_worker_health(self, worker_id: str) -> Optional[WorkerHealthInfo]:
        """Get health info for specific worker."""
        with self._lock:
            return self._worker_health.get(worker_id)
    
    def get_all_worker_health(self) -> Dict[str, WorkerHealthInfo]:
        """Get health info for all workers."""
        with self._lock:
            return self._worker_health.copy()
    
    def get_health_history(self, limit: int = 50) -> List[ClusterHealth]:
        """Get recent health history."""
        with self._lock:
            return self._health_history[-limit:] if self._health_history else []
    
    def get_unhealthy_workers(self) -> List[str]:
        """Get list of unhealthy worker IDs."""
        with self._lock:
            return [
                worker_id for worker_id, health in self._worker_health.items()
                if not health.is_healthy
            ]
    
    def get_degraded_workers(self) -> List[str]:
        """Get list of degraded worker IDs."""
        with self._lock:
            return [
                worker_id for worker_id, health in self._worker_health.items()
                if health.warnings and health.is_healthy
            ]
    
    def is_worker_healthy(self, worker_id: str) -> bool:
        """Check if specific worker is healthy."""
        with self._lock:
            health = self._worker_health.get(worker_id)
            return health.is_healthy if health else False
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self._lock:
            cluster_health = self._cluster_health
            
            if not cluster_health:
                return {'status': 'no_data'}
            
            return {
                'cluster_health': cluster_health.to_dict(),
                'monitoring': {
                    'uptime_seconds': time.time() - self._start_time,
                    'last_check': self._last_health_check,
                    'check_interval': self.check_interval,
                    'history_size': len(self._health_history)
                },
                'thresholds': {
                    'heartbeat_timeout': self.thresholds.heartbeat_timeout,
                    'heartbeat_warning': self.thresholds.heartbeat_warning,
                    'failure_rate_threshold': self.thresholds.failure_rate_threshold,
                    'latency_warning_ms': self.thresholds.latency_warning_ms,
                    'latency_critical_ms': self.thresholds.latency_critical_ms
                }
            }
    
    # Event callback management
    def add_health_changed_callback(self, callback: Callable[[ClusterHealth], None]) -> None:
        """Add callback for cluster health changes."""
        self._health_changed_callbacks.append(callback)
    
    def add_worker_unhealthy_callback(self, callback: Callable[[str, WorkerHealthInfo], None]) -> None:
        """Add callback for worker becoming unhealthy."""
        self._worker_unhealthy_callbacks.append(callback)
    
    def add_worker_recovered_callback(self, callback: Callable[[str, WorkerHealthInfo], None]) -> None:
        """Add callback for worker recovery."""
        self._worker_recovered_callbacks.append(callback)
    
    def add_critical_alert_callback(self, callback: Callable[[ClusterHealth], None]) -> None:
        """Add callback for critical health alerts."""
        self._critical_alert_callbacks.append(callback)
    
    def remove_health_changed_callback(self, callback: Callable[[ClusterHealth], None]) -> None:
        """Remove health changed callback."""
        if callback in self._health_changed_callbacks:
            self._health_changed_callbacks.remove(callback)
    
    def remove_worker_unhealthy_callback(self, callback: Callable[[str, WorkerHealthInfo], None]) -> None:
        """Remove worker unhealthy callback."""
        if callback in self._worker_unhealthy_callbacks:
            self._worker_unhealthy_callbacks.remove(callback)
    
    def remove_worker_recovered_callback(self, callback: Callable[[str, WorkerHealthInfo], None]) -> None:
        """Remove worker recovered callback."""
        if callback in self._worker_recovered_callbacks:
            self._worker_recovered_callbacks.remove(callback)
    
    def remove_critical_alert_callback(self, callback: Callable[[ClusterHealth], None]) -> None:
        """Remove critical alert callback."""
        if callback in self._critical_alert_callbacks:
            self._critical_alert_callbacks.remove(callback)