"""
Health Monitor for MLX Worker

This module provides health monitoring, metrics collection, and Metal memory
usage tracking for the MLX worker.
"""

import time
import logging
import asyncio
import psutil
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Health and performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    active_requests: int
    avg_latency_ms: float
    tokens_per_second: float
    total_requests: int
    failed_requests: int
    uptime_seconds: float


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: str  # healthy, degraded, unhealthy
    message: str
    score: float  # 0.0 to 1.0
    issues: list
    metrics: HealthMetrics


class MetalMemoryTracker:
    """Track Metal GPU memory usage"""
    
    def __init__(self):
        self.peak_usage = 0.0
        self.current_usage = 0.0
        self.allocation_count = 0
        self.lock = threading.Lock()
    
    def update_usage(self):
        """Update Metal memory usage statistics"""
        try:
            with self.lock:
                # Get Metal memory info (this is a simplified approach)
                # In practice, you might need to use Metal-specific APIs
                self.current_usage = self._estimate_metal_memory()
                self.peak_usage = max(self.peak_usage, self.current_usage)
                self.allocation_count += 1
                
        except Exception as e:
            logger.warning(f"Failed to update Metal memory usage: {e}")
    
    def _estimate_metal_memory(self) -> float:
        """Estimate Metal memory usage in GB"""
        try:
            # This is a rough estimation - in practice you'd use Metal APIs
            # For now, we'll estimate based on MLX array memory usage
            
            # Get process memory as a proxy
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            
            # Estimate that 60-80% of process memory is Metal memory for ML workloads
            estimated_metal_gb = (memory_mb * 0.7) / 1024
            
            return estimated_metal_gb
            
        except Exception as e:
            logger.warning(f"Error estimating Metal memory: {e}")
            return 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """Get Metal memory statistics"""
        with self.lock:
            return {
                'current_gb': self.current_usage,
                'peak_gb': self.peak_usage,
                'allocation_count': self.allocation_count
            }
    
    def reset_peak(self):
        """Reset peak usage counter"""
        with self.lock:
            self.peak_usage = self.current_usage


class HealthMonitor:
    """Comprehensive health monitoring for MLX worker"""
    
    def __init__(self, model_manager, inference_engine, 
                 history_size: int = 100, check_interval: float = 30.0):
        self.model_manager = model_manager
        self.inference_engine = inference_engine
        self.history_size = history_size
        self.check_interval = check_interval
        
        self.start_time = time.time()
        self.metrics_history = deque(maxlen=history_size)
        self.metal_tracker = MetalMemoryTracker()
        
        self.monitor_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Health thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'disk_warning': 90.0,
            'disk_critical': 98.0,
            'latency_warning': 5000.0,  # 5 seconds
            'latency_critical': 10000.0,  # 10 seconds
            'error_rate_warning': 0.05,  # 5%
            'error_rate_critical': 0.15,  # 15%
        }
        
        logger.info("Initialized HealthMonitor")
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_gb': memory_used_gb,
                'memory_available_gb': memory_available_gb,
                'disk_usage_percent': disk_usage_percent
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_used_gb': 0.0,
                'memory_available_gb': 0.0,
                'disk_usage_percent': 0.0
            }
    
    def _collect_inference_metrics(self) -> Dict[str, Any]:
        """Collect inference-specific metrics"""
        try:
            metrics = self.inference_engine.get_metrics()
            return {
                'active_requests': metrics['active_requests'],
                'avg_latency_ms': metrics['avg_latency_ms'],
                'tokens_per_second': metrics['tokens_per_second'],
                'total_requests': metrics['total_requests'],
                'failed_requests': metrics['failed_requests']
            }
        except Exception as e:
            logger.error(f"Error collecting inference metrics: {e}")
            return {
                'active_requests': 0,
                'avg_latency_ms': 0.0,
                'tokens_per_second': 0.0,
                'total_requests': 0,
                'failed_requests': 0
            }
    
    def collect_metrics(self) -> HealthMetrics:
        """Collect comprehensive health metrics"""
        timestamp = time.time()
        uptime_seconds = timestamp - self.start_time
        
        # Update Metal memory tracking
        self.metal_tracker.update_usage()
        
        # Collect system metrics
        system_metrics = self._collect_system_metrics()
        
        # Collect inference metrics
        inference_metrics = self._collect_inference_metrics()
        
        # Create health metrics object
        metrics = HealthMetrics(
            timestamp=timestamp,
            uptime_seconds=uptime_seconds,
            **system_metrics,
            **inference_metrics
        )
        
        # Add to history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_health_score(self, metrics: HealthMetrics) -> float:
        """Calculate overall health score (0.0 to 1.0)"""
        score = 1.0
        
        # CPU penalty
        if metrics.cpu_percent > self.thresholds['cpu_critical']:
            score -= 0.3
        elif metrics.cpu_percent > self.thresholds['cpu_warning']:
            score -= 0.1
        
        # Memory penalty
        if metrics.memory_percent > self.thresholds['memory_critical']:
            score -= 0.3
        elif metrics.memory_percent > self.thresholds['memory_warning']:
            score -= 0.1
        
        # Disk penalty
        if metrics.disk_usage_percent > self.thresholds['disk_critical']:
            score -= 0.2
        elif metrics.disk_usage_percent > self.thresholds['disk_warning']:
            score -= 0.05
        
        # Latency penalty
        if metrics.avg_latency_ms > self.thresholds['latency_critical']:
            score -= 0.2
        elif metrics.avg_latency_ms > self.thresholds['latency_warning']:
            score -= 0.1
        
        # Error rate penalty
        if metrics.total_requests > 0:
            error_rate = metrics.failed_requests / metrics.total_requests
            if error_rate > self.thresholds['error_rate_critical']:
                score -= 0.3
            elif error_rate > self.thresholds['error_rate_warning']:
                score -= 0.1
        
        return max(0.0, score)
    
    def _identify_issues(self, metrics: HealthMetrics) -> list:
        """Identify specific health issues"""
        issues = []
        
        # CPU issues
        if metrics.cpu_percent > self.thresholds['cpu_critical']:
            issues.append(f"Critical CPU usage: {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent > self.thresholds['cpu_warning']:
            issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Memory issues
        if metrics.memory_percent > self.thresholds['memory_critical']:
            issues.append(f"Critical memory usage: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent > self.thresholds['memory_warning']:
            issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # Disk issues
        if metrics.disk_usage_percent > self.thresholds['disk_critical']:
            issues.append(f"Critical disk usage: {metrics.disk_usage_percent:.1f}%")
        elif metrics.disk_usage_percent > self.thresholds['disk_warning']:
            issues.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        # Latency issues
        if metrics.avg_latency_ms > self.thresholds['latency_critical']:
            issues.append(f"Critical latency: {metrics.avg_latency_ms:.0f}ms")
        elif metrics.avg_latency_ms > self.thresholds['latency_warning']:
            issues.append(f"High latency: {metrics.avg_latency_ms:.0f}ms")
        
        # Error rate issues
        if metrics.total_requests > 0:
            error_rate = metrics.failed_requests / metrics.total_requests
            if error_rate > self.thresholds['error_rate_critical']:
                issues.append(f"Critical error rate: {error_rate:.1%}")
            elif error_rate > self.thresholds['error_rate_warning']:
                issues.append(f"High error rate: {error_rate:.1%}")
        
        # Active request overload
        if metrics.active_requests > 20:
            issues.append(f"High request load: {metrics.active_requests} active")
        
        return issues
    
    def get_health_status(self) -> SystemHealth:
        """Get current health status"""
        # Collect current metrics
        metrics = self.collect_metrics()
        
        # Calculate health score
        score = self._calculate_health_score(metrics)
        
        # Identify issues
        issues = self._identify_issues(metrics)
        
        # Determine overall status
        if score >= 0.8 and not issues:
            status = "healthy"
            message = "All systems operating normally"
        elif score >= 0.6:
            status = "degraded"
            message = f"System degraded: {len(issues)} issues detected"
        else:
            status = "unhealthy"
            message = f"System unhealthy: {len(issues)} critical issues"
        
        return SystemHealth(
            status=status,
            message=message,
            score=score,
            issues=issues,
            metrics=metrics
        )
    
    def get_metrics_summary(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Get metrics summary over a time window"""
        if not self.metrics_history:
            return {}
        
        # Filter metrics within time window
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            recent_metrics = [self.metrics_history[-1]]  # Use latest if no recent data
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_latency = sum(m.avg_latency_ms for m in recent_metrics) / len(recent_metrics)
        avg_tokens_per_sec = sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics)
        
        # Get totals
        total_requests = recent_metrics[-1].total_requests if recent_metrics else 0
        failed_requests = recent_metrics[-1].failed_requests if recent_metrics else 0
        
        # Metal memory stats
        metal_stats = self.metal_tracker.get_stats()
        
        return {
            'window_minutes': window_minutes,
            'sample_count': len(recent_metrics),
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'avg_latency_ms': avg_latency,
            'avg_tokens_per_second': avg_tokens_per_sec,
            'total_requests': total_requests,
            'failed_requests': failed_requests,
            'success_rate': (total_requests - failed_requests) / total_requests if total_requests > 0 else 1.0,
            'metal_memory_gb': metal_stats['current_gb'],
            'metal_memory_peak_gb': metal_stats['peak_gb'],
            'uptime_seconds': recent_metrics[-1].uptime_seconds if recent_metrics else 0
        }
    
    async def start_monitoring(self):
        """Start the health monitoring task"""
        if self.monitor_task and not self.monitor_task.done():
            logger.warning("Health monitoring already running")
            return
        
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started health monitoring with {self.check_interval}s interval")
    
    async def stop_monitoring(self):
        """Stop the health monitoring task"""
        self.shutdown_event.set()
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
        
        logger.info("Stopped health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = self.collect_metrics()
                
                # Log health status periodically
                health = self.get_health_status()
                if health.status != "healthy":
                    logger.warning(f"Health status: {health.status} - {health.message}")
                    for issue in health.issues:
                        logger.warning(f"  Issue: {issue}")
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                logger.info("Health monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    def export_metrics_for_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        if not self.metrics_history:
            return ""
        
        latest = self.metrics_history[-1]
        metal_stats = self.metal_tracker.get_stats()
        
        metrics = [
            f"mlx_worker_cpu_percent {latest.cpu_percent}",
            f"mlx_worker_memory_percent {latest.memory_percent}",
            f"mlx_worker_memory_used_gb {latest.memory_used_gb}",
            f"mlx_worker_disk_usage_percent {latest.disk_usage_percent}",
            f"mlx_worker_active_requests {latest.active_requests}",
            f"mlx_worker_avg_latency_ms {latest.avg_latency_ms}",
            f"mlx_worker_tokens_per_second {latest.tokens_per_second}",
            f"mlx_worker_total_requests {latest.total_requests}",
            f"mlx_worker_failed_requests {latest.failed_requests}",
            f"mlx_worker_uptime_seconds {latest.uptime_seconds}",
            f"mlx_worker_metal_memory_gb {metal_stats['current_gb']}",
            f"mlx_worker_metal_memory_peak_gb {metal_stats['peak_gb']}",
        ]
        
        return "\n".join(metrics) + "\n"