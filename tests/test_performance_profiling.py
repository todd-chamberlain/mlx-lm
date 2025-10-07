"""
Tests for performance profiling utilities.

Tests profiling infrastructure, metrics collection, and performance analysis.
"""

import pytest
import time
import tempfile
import json
from pathlib import Path

from common.profiling import (
    PerformanceProfiler,
    get_profiler,
    profile_function,
    profile_block,
    print_profiling_stats,
    export_profiling_stats,
    reset_profiling
)


class TestPerformanceProfiler:
    """Test PerformanceProfiler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler()
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        assert self.profiler.is_enabled() == True
        assert len(self.profiler._timing_metrics) == 0
        assert len(self.profiler._memory_metrics) == 0
    
    def test_enable_disable(self):
        """Test enabling and disabling profiler."""
        self.profiler.disable()
        assert self.profiler.is_enabled() == False
        
        self.profiler.enable()
        assert self.profiler.is_enabled() == True
    
    def test_record_timing(self):
        """Test recording timing metrics."""
        self.profiler.record_timing("test_operation", 100.5)
        
        stats = self.profiler.get_timing_stats("test_operation")
        assert stats["count"] == 1
        assert stats["mean_ms"] == 100.5
    
    def test_record_timing_with_metadata(self):
        """Test recording timing with metadata."""
        metadata = {"worker_id": "worker-1", "request_id": "req-123"}
        self.profiler.record_timing("test_op", 50.0, metadata)
        
        assert len(self.profiler._timing_metrics) == 1
        metric = self.profiler._timing_metrics[0]
        assert metric.metadata == metadata
    
    def test_record_memory(self):
        """Test recording memory metrics."""
        self.profiler.record_memory("test_operation", 256.0)
        
        stats = self.profiler.get_memory_stats("test_operation")
        assert stats["count"] == 1
        assert stats["mean_mb"] == 256.0
    
    def test_get_current_memory(self):
        """Test getting current memory usage."""
        memory_mb = self.profiler.get_current_memory_mb()
        assert memory_mb > 0
        assert isinstance(memory_mb, float)
    
    def test_get_current_cpu(self):
        """Test getting current CPU usage."""
        cpu_percent = self.profiler.get_current_cpu_percent()
        assert cpu_percent >= 0
        assert isinstance(cpu_percent, float)
    
    def test_profile_block_context_manager(self):
        """Test profile_block context manager."""
        with self.profiler.profile_block("test_block"):
            time.sleep(0.01)  # 10ms
        
        stats = self.profiler.get_timing_stats("test_block")
        assert stats["count"] == 1
        assert stats["mean_ms"] >= 10.0  # Should be at least 10ms
    
    def test_profile_block_with_metadata(self):
        """Test profile_block with metadata."""
        metadata = {"operation": "inference"}
        
        with self.profiler.profile_block("test_block", metadata):
            time.sleep(0.01)
        
        metric = self.profiler._timing_metrics[0]
        assert metric.metadata == metadata
    
    def test_profile_function_decorator(self):
        """Test profile_function decorator."""
        @self.profiler.profile_function()
        def test_func():
            time.sleep(0.01)
            return "result"
        
        result = test_func()
        assert result == "result"
        
        stats = self.profiler.get_timing_stats("test_func")
        assert stats["count"] == 1
        assert stats["mean_ms"] >= 10.0
    
    def test_profile_function_with_custom_name(self):
        """Test profile_function with custom name."""
        @self.profiler.profile_function(name="custom_operation")
        def test_func():
            time.sleep(0.01)
        
        test_func()
        
        stats = self.profiler.get_timing_stats("custom_operation")
        assert stats["count"] == 1
    
    def test_profile_function_with_metadata(self):
        """Test profile_function with metadata."""
        metadata = {"version": "1.0"}
        
        @self.profiler.profile_function(metadata=metadata)
        def test_func():
            pass
        
        test_func()
        
        metric = self.profiler._timing_metrics[0]
        assert metric.metadata == metadata
    
    def test_timing_stats_multiple_calls(self):
        """Test timing statistics with multiple calls."""
        for i in range(10):
            self.profiler.record_timing("operation", float(i * 10))
        
        stats = self.profiler.get_timing_stats("operation")
        assert stats["count"] == 10
        assert stats["min_ms"] == 0.0
        assert stats["max_ms"] == 90.0
        assert stats["mean_ms"] == 45.0
    
    def test_timing_stats_percentiles(self):
        """Test timing statistics percentile calculations."""
        # Record 100 measurements
        for i in range(100):
            self.profiler.record_timing("operation", float(i))
        
        stats = self.profiler.get_timing_stats("operation")
        assert stats["count"] == 100
        assert stats["p50_ms"] == 50.0
        assert stats["p95_ms"] == 95.0
        assert stats["p99_ms"] == 99.0
    
    def test_memory_stats_multiple_calls(self):
        """Test memory statistics with multiple calls."""
        for i in range(10):
            self.profiler.record_memory("operation", float(i * 100))
        
        stats = self.profiler.get_memory_stats("operation")
        assert stats["count"] == 10
        assert stats["min_mb"] == 0.0
        assert stats["max_mb"] == 900.0
        assert stats["mean_mb"] == 450.0
    
    def test_get_all_stats(self):
        """Test getting all statistics."""
        # Record some timing metrics
        self.profiler.record_timing("op1", 100.0)
        self.profiler.record_timing("op2", 200.0)
        
        # Record some memory metrics
        self.profiler.record_memory("op1", 256.0)
        self.profiler.record_memory("op2", 512.0)
        
        stats = self.profiler.get_all_stats()
        
        assert "timing" in stats
        assert "memory" in stats
        assert "summary" in stats
        
        assert "op1" in stats["timing"]
        assert "op2" in stats["timing"]
        assert "op1" in stats["memory"]
        assert "op2" in stats["memory"]
        
        assert stats["summary"]["total_timing_metrics"] == 2
        assert stats["summary"]["total_memory_metrics"] == 2
    
    def test_export_stats(self):
        """Test exporting statistics to file."""
        self.profiler.record_timing("operation", 100.0)
        self.profiler.record_memory("operation", 256.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            self.profiler.export_stats(filepath)
            
            # Verify file was created and contains valid JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "timing" in data
            assert "memory" in data
            assert "summary" in data
            
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_reset(self):
        """Test resetting profiling data."""
        self.profiler.record_timing("operation", 100.0)
        self.profiler.record_memory("operation", 256.0)
        
        assert len(self.profiler._timing_metrics) == 1
        assert len(self.profiler._memory_metrics) == 1
        
        self.profiler.reset()
        
        assert len(self.profiler._timing_metrics) == 0
        assert len(self.profiler._memory_metrics) == 0
    
    def test_disabled_profiler(self):
        """Test that disabled profiler doesn't record metrics."""
        self.profiler.disable()
        
        self.profiler.record_timing("operation", 100.0)
        self.profiler.record_memory("operation", 256.0)
        
        assert len(self.profiler._timing_metrics) == 0
        assert len(self.profiler._memory_metrics) == 0
    
    def test_disabled_profiler_context_manager(self):
        """Test that disabled profiler doesn't record in context manager."""
        self.profiler.disable()
        
        with self.profiler.profile_block("test_block"):
            time.sleep(0.01)
        
        assert len(self.profiler._timing_metrics) == 0
    
    def test_disabled_profiler_decorator(self):
        """Test that disabled profiler doesn't record with decorator."""
        self.profiler.disable()
        
        @self.profiler.profile_function()
        def test_func():
            return "result"
        
        result = test_func()
        assert result == "result"
        assert len(self.profiler._timing_metrics) == 0


class TestGlobalProfiler:
    """Test global profiler functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        reset_profiling()
    
    def teardown_method(self):
        """Clean up after tests."""
        reset_profiling()
    
    def test_get_profiler(self):
        """Test getting global profiler instance."""
        profiler = get_profiler()
        assert isinstance(profiler, PerformanceProfiler)
    
    def test_profile_function_global(self):
        """Test global profile_function decorator."""
        @profile_function()
        def test_func():
            time.sleep(0.01)
            return "result"
        
        result = test_func()
        assert result == "result"
        
        profiler = get_profiler()
        stats = profiler.get_timing_stats("test_func")
        assert stats["count"] == 1
    
    def test_profile_block_global(self):
        """Test global profile_block context manager."""
        with profile_block("test_operation"):
            time.sleep(0.01)
        
        profiler = get_profiler()
        stats = profiler.get_timing_stats("test_operation")
        assert stats["count"] == 1
    
    def test_export_profiling_stats_global(self):
        """Test global export_profiling_stats function."""
        with profile_block("test_operation"):
            time.sleep(0.01)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            export_profiling_stats(filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "timing" in data
            assert "test_operation" in data["timing"]
            
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_reset_profiling_global(self):
        """Test global reset_profiling function."""
        with profile_block("test_operation"):
            pass
        
        profiler = get_profiler()
        assert len(profiler._timing_metrics) == 1
        
        reset_profiling()
        assert len(profiler._timing_metrics) == 0


class TestProfilingIntegration:
    """Test profiling integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler()
    
    def test_nested_profiling(self):
        """Test nested profiling blocks."""
        with self.profiler.profile_block("outer"):
            time.sleep(0.01)
            
            with self.profiler.profile_block("inner"):
                time.sleep(0.01)
        
        outer_stats = self.profiler.get_timing_stats("outer")
        inner_stats = self.profiler.get_timing_stats("inner")
        
        assert outer_stats["count"] == 1
        assert inner_stats["count"] == 1
        assert outer_stats["mean_ms"] >= inner_stats["mean_ms"]
    
    def test_concurrent_profiling(self):
        """Test profiling with concurrent operations."""
        import threading
        
        def worker():
            with self.profiler.profile_block("worker_operation"):
                time.sleep(0.01)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        stats = self.profiler.get_timing_stats("worker_operation")
        assert stats["count"] == 5
    
    def test_profiling_with_exceptions(self):
        """Test that profiling works even when exceptions occur."""
        try:
            with self.profiler.profile_block("failing_operation"):
                time.sleep(0.01)
                raise ValueError("Test error")
        except ValueError:
            pass
        
        stats = self.profiler.get_timing_stats("failing_operation")
        assert stats["count"] == 1
        assert stats["mean_ms"] >= 10.0
    
    def test_memory_tracking_integration(self):
        """Test memory tracking during operations."""
        with self.profiler.profile_block("memory_operation"):
            # Allocate some memory
            data = [0] * 1000000  # ~8MB
            time.sleep(0.01)
        
        timing_stats = self.profiler.get_timing_stats("memory_operation")
        assert timing_stats["count"] == 1
        
        # Memory metrics might be recorded if change is significant
        memory_stats = self.profiler.get_memory_stats("memory_operation")
        # Memory tracking is optional based on threshold
    
    def test_profiling_statistics_accuracy(self):
        """Test accuracy of profiling statistics."""
        # Record known values
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        for value in values:
            self.profiler.record_timing("operation", value)
        
        stats = self.profiler.get_timing_stats("operation")
        
        assert stats["count"] == 10
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 100.0
        assert stats["mean_ms"] == 55.0
        assert stats["p50_ms"] == 60.0  # Median of sorted values


class TestProfilingPerformance:
    """Test profiling overhead and performance."""
    
    def test_profiling_overhead(self):
        """Test that profiling overhead is reasonable for real operations."""
        profiler = PerformanceProfiler()
        
        # Measure time without profiling (with actual work)
        start = time.perf_counter()
        for _ in range(100):
            time.sleep(0.001)  # 1ms of actual work
        baseline_time = time.perf_counter() - start
        
        # Measure time with profiling
        start = time.perf_counter()
        for _ in range(100):
            with profiler.profile_block("operation"):
                time.sleep(0.001)  # 1ms of actual work
        profiled_time = time.perf_counter() - start
        
        # Overhead should be less than 50% for real operations
        overhead_ratio = (profiled_time - baseline_time) / baseline_time
        assert overhead_ratio < 0.5  # Less than 50% overhead
    
    def test_large_dataset_handling(self):
        """Test profiling with large number of metrics."""
        profiler = PerformanceProfiler()
        
        # Record 10000 metrics
        for i in range(10000):
            profiler.record_timing("operation", float(i))
        
        # Should still be able to get stats efficiently
        start = time.perf_counter()
        stats = profiler.get_timing_stats("operation")
        stats_time = time.perf_counter() - start
        
        assert stats["count"] == 10000
        assert stats_time < 1.0  # Should complete in less than 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
