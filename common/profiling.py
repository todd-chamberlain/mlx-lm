"""
Performance Profiling Utilities

Comprehensive profiling infrastructure for measuring and analyzing
performance characteristics of the MLX distributed inference system.
"""

import time
import functools
import logging
import threading
import psutil
import os
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class TimingMetric:
    """Timing metric data."""
    name: str
    duration_ms: float
    timestamp: datetime
    thread_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryMetric:
    """Memory usage metric data."""
    name: str
    memory_mb: float
    timestamp: datetime
    process_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """
    Performance profiler for collecting timing and resource metrics.
    
    Features:
    - Function timing with decorators
    - Context manager for code block timing
    - Memory usage tracking
    - CPU usage monitoring
    - Metrics aggregation and reporting
    """
    
    def __init__(self):
        self._timing_metrics: List[TimingMetric] = []
        self._memory_metrics: List[MemoryMetric] = []
        self._lock = threading.Lock()
        self._enabled = True
        self._process = psutil.Process(os.getpid())
    
    def enable(self):
        """Enable profiling."""
        self._enabled = True
    
    def disable(self):
        """Disable profiling."""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self._enabled
    
    def record_timing(self, name: str, duration_ms: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a timing metric.
        
        Args:
            name: Metric name
            duration_ms: Duration in milliseconds
            metadata: Optional metadata
        """
        if not self._enabled:
            return
        
        metric = TimingMetric(
            name=name,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            thread_id=threading.get_ident(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._timing_metrics.append(metric)
    
    def record_memory(self, name: str, memory_mb: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a memory metric.
        
        Args:
            name: Metric name
            memory_mb: Memory usage in MB
            metadata: Optional metadata
        """
        if not self._enabled:
            return
        
        metric = MemoryMetric(
            name=name,
            memory_mb=memory_mb,
            timestamp=datetime.now(),
            process_id=os.getpid(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._memory_metrics.append(metric)
    
    def get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            return self._process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def get_current_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self._process.cpu_percent(interval=0.1)
        except Exception as e:
            logger.warning(f"Failed to get CPU usage: {e}")
            return 0.0
    
    @contextmanager
    def profile_block(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Block name
            metadata: Optional metadata
            
        Example:
            with profiler.profile_block("inference"):
                result = model.generate(prompt)
        """
        if not self._enabled:
            yield
            return
        
        start_time = time.perf_counter()
        start_memory = self.get_current_memory_mb()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self.get_current_memory_mb()
            
            duration_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory
            
            self.record_timing(name, duration_ms, metadata)
            
            if abs(memory_delta) > 0.1:  # Only record significant memory changes
                self.record_memory(name, memory_delta, metadata)
    
    def profile_function(self, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Decorator for profiling function execution.
        
        Args:
            name: Optional custom name (defaults to function name)
            metadata: Optional metadata
            
        Example:
            @profiler.profile_function()
            def my_function():
                pass
        """
        def decorator(func: Callable) -> Callable:
            func_name = name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self._enabled:
                    return func(*args, **kwargs)
                
                with self.profile_block(func_name, metadata):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_timing_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get timing statistics.
        
        Args:
            name: Optional metric name filter
            
        Returns:
            Dictionary with timing statistics
        """
        with self._lock:
            metrics = self._timing_metrics
            if name:
                metrics = [m for m in metrics if m.name == name]
            
            if not metrics:
                return {}
            
            durations = [m.duration_ms for m in metrics]
            durations.sort()
            
            count = len(durations)
            total = sum(durations)
            mean = total / count
            
            return {
                "count": count,
                "total_ms": total,
                "mean_ms": mean,
                "min_ms": durations[0],
                "max_ms": durations[-1],
                "p50_ms": durations[count // 2],
                "p95_ms": durations[int(count * 0.95)],
                "p99_ms": durations[int(count * 0.99)]
            }
    
    def get_memory_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Args:
            name: Optional metric name filter
            
        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            metrics = self._memory_metrics
            if name:
                metrics = [m for m in metrics if m.name == name]
            
            if not metrics:
                return {}
            
            memory_values = [m.memory_mb for m in metrics]
            memory_values.sort()
            
            count = len(memory_values)
            total = sum(memory_values)
            mean = total / count
            
            return {
                "count": count,
                "total_mb": total,
                "mean_mb": mean,
                "min_mb": memory_values[0],
                "max_mb": memory_values[-1],
                "p50_mb": memory_values[count // 2],
                "p95_mb": memory_values[int(count * 0.95)],
                "p99_mb": memory_values[int(count * 0.99)]
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all profiling statistics."""
        with self._lock:
            # Group timing metrics by name
            timing_by_name = defaultdict(list)
            for metric in self._timing_metrics:
                timing_by_name[metric.name].append(metric.duration_ms)
            
            # Group memory metrics by name
            memory_by_name = defaultdict(list)
            for metric in self._memory_metrics:
                memory_by_name[metric.name].append(metric.memory_mb)
            
            stats = {
                "timing": {},
                "memory": {},
                "summary": {
                    "total_timing_metrics": len(self._timing_metrics),
                    "total_memory_metrics": len(self._memory_metrics),
                    "unique_timing_names": len(timing_by_name),
                    "unique_memory_names": len(memory_by_name)
                }
            }
            
            # Calculate stats for each timing metric name
            for name, durations in timing_by_name.items():
                durations.sort()
                count = len(durations)
                total = sum(durations)
                
                stats["timing"][name] = {
                    "count": count,
                    "total_ms": total,
                    "mean_ms": total / count,
                    "min_ms": durations[0],
                    "max_ms": durations[-1],
                    "p50_ms": durations[count // 2],
                    "p95_ms": durations[int(count * 0.95)] if count > 20 else durations[-1],
                    "p99_ms": durations[int(count * 0.99)] if count > 100 else durations[-1]
                }
            
            # Calculate stats for each memory metric name
            for name, memory_values in memory_by_name.items():
                memory_values.sort()
                count = len(memory_values)
                total = sum(memory_values)
                
                stats["memory"][name] = {
                    "count": count,
                    "total_mb": total,
                    "mean_mb": total / count,
                    "min_mb": memory_values[0],
                    "max_mb": memory_values[-1],
                    "p50_mb": memory_values[count // 2],
                    "p95_mb": memory_values[int(count * 0.95)] if count > 20 else memory_values[-1],
                    "p99_mb": memory_values[int(count * 0.99)] if count > 100 else memory_values[-1]
                }
            
            return stats
    
    def print_stats(self, name: Optional[str] = None):
        """
        Print profiling statistics.
        
        Args:
            name: Optional metric name filter
        """
        if name:
            print(f"\n=== Profiling Stats for '{name}' ===")
            timing_stats = self.get_timing_stats(name)
            if timing_stats:
                print("\nTiming:")
                for key, value in timing_stats.items():
                    print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
            
            memory_stats = self.get_memory_stats(name)
            if memory_stats:
                print("\nMemory:")
                for key, value in memory_stats.items():
                    print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        else:
            stats = self.get_all_stats()
            print("\n=== Profiling Stats (All Metrics) ===")
            print(f"\nSummary:")
            for key, value in stats["summary"].items():
                print(f"  {key}: {value}")
            
            if stats["timing"]:
                print("\nTiming Metrics:")
                for name, timing_stats in stats["timing"].items():
                    print(f"\n  {name}:")
                    print(f"    count: {timing_stats['count']}")
                    print(f"    mean: {timing_stats['mean_ms']:.2f}ms")
                    print(f"    p50: {timing_stats['p50_ms']:.2f}ms")
                    print(f"    p95: {timing_stats['p95_ms']:.2f}ms")
                    print(f"    p99: {timing_stats['p99_ms']:.2f}ms")
            
            if stats["memory"]:
                print("\nMemory Metrics:")
                for name, memory_stats in stats["memory"].items():
                    print(f"\n  {name}:")
                    print(f"    count: {memory_stats['count']}")
                    print(f"    mean: {memory_stats['mean_mb']:.2f}MB")
                    print(f"    max: {memory_stats['max_mb']:.2f}MB")
    
    def export_stats(self, filepath: str):
        """
        Export profiling statistics to JSON file.
        
        Args:
            filepath: Output file path
        """
        stats = self.get_all_stats()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Profiling stats exported to {filepath}")
    
    def reset(self):
        """Reset all profiling data."""
        with self._lock:
            self._timing_metrics.clear()
            self._memory_metrics.clear()
        
        logger.info("Profiling data reset")


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def profile_function(name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for profiling function execution using global profiler.
    
    Args:
        name: Optional custom name
        metadata: Optional metadata
        
    Example:
        @profile_function()
        def my_function():
            pass
    """
    return _global_profiler.profile_function(name, metadata)


@contextmanager
def profile_block(name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager for profiling a code block using global profiler.
    
    Args:
        name: Block name
        metadata: Optional metadata
        
    Example:
        with profile_block("inference"):
            result = model.generate(prompt)
    """
    with _global_profiler.profile_block(name, metadata):
        yield


def print_profiling_stats(name: Optional[str] = None):
    """Print profiling statistics from global profiler."""
    _global_profiler.print_stats(name)


def export_profiling_stats(filepath: str):
    """Export profiling statistics to file."""
    _global_profiler.export_stats(filepath)


def reset_profiling():
    """Reset global profiling data."""
    _global_profiler.reset()
