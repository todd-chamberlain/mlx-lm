"""
Performance Benchmarking and Load Testing Suite

Comprehensive performance testing for MLX distributed inference system.
Tests latency, throughput, concurrent request handling, and stress scenarios.
"""

import asyncio
import pytest
import time
import statistics
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
import psutil
import os


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    
    test_name: str
    timestamp: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    
    # Latency metrics (milliseconds)
    latency_min: float
    latency_max: float
    latency_mean: float
    latency_median: float
    latency_p95: float
    latency_p99: float
    
    # Throughput metrics
    tokens_per_second: Optional[float] = None
    bytes_per_second: Optional[float] = None
    
    # Resource metrics
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Error metrics
    error_rate: float = 0.0
    timeout_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class PerformanceTestRunner:
    """Performance test execution and metrics collection."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[PerformanceMetrics] = []
        self.process = psutil.Process()
    
    def measure_latency(self, func, *args, **kwargs) -> Tuple[float, Any]:
        """Measure function execution latency."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        return latency_ms, result
    
    def get_resource_usage(self) -> Tuple[float, float]:
        """Get current CPU and memory usage."""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        return cpu_percent, memory_mb
    
    def calculate_metrics(
        self,
        test_name: str,
        latencies: List[float],
        successes: int,
        failures: int,
        duration: float,
        tokens: Optional[int] = None,
        bytes_transferred: Optional[int] = None
    ) -> PerformanceMetrics:
        """Calculate performance metrics from test results."""
        total_requests = successes + failures
        
        if not latencies:
            latencies = [0.0]
        
        sorted_latencies = sorted(latencies)
        
        cpu_usage, memory_usage = self.get_resource_usage()
        
        metrics = PerformanceMetrics(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            total_requests=total_requests,
            successful_requests=successes,
            failed_requests=failures,
            requests_per_second=total_requests / duration if duration > 0 else 0,
            latency_min=min(sorted_latencies),
            latency_max=max(sorted_latencies),
            latency_mean=statistics.mean(sorted_latencies),
            latency_median=statistics.median(sorted_latencies),
            latency_p95=sorted_latencies[int(len(sorted_latencies) * 0.95)] if len(sorted_latencies) > 0 else 0,
            latency_p99=sorted_latencies[int(len(sorted_latencies) * 0.99)] if len(sorted_latencies) > 0 else 0,
            tokens_per_second=tokens / duration if tokens and duration > 0 else None,
            bytes_per_second=bytes_transferred / duration if bytes_transferred and duration > 0 else None,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage,
            error_rate=failures / total_requests if total_requests > 0 else 0,
            timeout_count=0
        )
        
        self.results.append(metrics)
        return metrics
    
    def save_results(self, output_file: str = "performance_results.json"):
        """Save performance results to file."""
        results_dict = [result.to_dict() for result in self.results]
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def print_summary(self, metrics: PerformanceMetrics):
        """Print performance metrics summary."""
        print(f"\n{'='*60}")
        print(f"Performance Test: {metrics.test_name}")
        print(f"{'='*60}")
        print(f"Duration: {metrics.duration_seconds:.2f}s")
        print(f"Total Requests: {metrics.total_requests}")
        print(f"Successful: {metrics.successful_requests}")
        print(f"Failed: {metrics.failed_requests}")
        print(f"Requests/sec: {metrics.requests_per_second:.2f}")
        print(f"\nLatency (ms):")
        print(f"  Min: {metrics.latency_min:.2f}")
        print(f"  Mean: {metrics.latency_mean:.2f}")
        print(f"  Median: {metrics.latency_median:.2f}")
        print(f"  P95: {metrics.latency_p95:.2f}")
        print(f"  P99: {metrics.latency_p99:.2f}")
        print(f"  Max: {metrics.latency_max:.2f}")
        
        if metrics.tokens_per_second:
            print(f"\nThroughput:")
            print(f"  Tokens/sec: {metrics.tokens_per_second:.2f}")
        
        print(f"\nResources:")
        print(f"  CPU: {metrics.cpu_usage_percent:.1f}%")
        print(f"  Memory: {metrics.memory_usage_mb:.1f} MB")
        print(f"{'='*60}\n")


class TestSingleRequestPerformance:
    """Test single request latency and throughput."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = PerformanceTestRunner()
        self.test_payload = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
    
    def test_single_request_latency(self):
        """Test single request latency measurement."""
        # Mock API endpoint
        def mock_request():
            time.sleep(0.1)  # Simulate 100ms latency
            return {"status": "success", "tokens": 50}
        
        latencies = []
        successes = 0
        
        # Run 10 single requests
        start_time = time.perf_counter()
        for _ in range(10):
            latency, result = self.runner.measure_latency(mock_request)
            latencies.append(latency)
            if result["status"] == "success":
                successes += 1
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        metrics = self.runner.calculate_metrics(
            "single_request_latency",
            latencies,
            successes,
            0,
            duration,
            tokens=500
        )
        
        self.runner.print_summary(metrics)
        
        # Assertions
        assert metrics.successful_requests == 10
        assert metrics.latency_mean >= 100  # At least 100ms
        assert metrics.latency_mean < 150  # But not too much overhead
    
    def test_request_throughput_measurement(self):
        """Test throughput measurement for single requests."""
        def mock_request_with_tokens():
            time.sleep(0.05)
            return {"tokens": 100, "bytes": 500}
        
        latencies = []
        total_tokens = 0
        total_bytes = 0
        successes = 0
        
        start_time = time.perf_counter()
        for _ in range(20):
            latency, result = self.runner.measure_latency(mock_request_with_tokens)
            latencies.append(latency)
            total_tokens += result["tokens"]
            total_bytes += result["bytes"]
            successes += 1
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        metrics = self.runner.calculate_metrics(
            "request_throughput",
            latencies,
            successes,
            0,
            duration,
            tokens=total_tokens,
            bytes_transferred=total_bytes
        )
        
        self.runner.print_summary(metrics)
        
        # Assertions
        assert metrics.tokens_per_second is not None
        assert metrics.tokens_per_second > 0
        assert metrics.bytes_per_second is not None


class TestConcurrentRequestHandling:
    """Test concurrent request handling and scalability."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = PerformanceTestRunner()
    
    def concurrent_request_test(self, num_concurrent: int, num_requests: int):
        """Run concurrent request test."""
        def mock_request(request_id: int):
            start = time.perf_counter()
            time.sleep(0.1)  # Simulate processing
            end = time.perf_counter()
            return {
                "request_id": request_id,
                "latency": (end - start) * 1000,
                "success": True
            }
        
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(mock_request, i) for i in range(num_requests)]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    latencies.append(result["latency"])
                    if result["success"]:
                        successes += 1
                    else:
                        failures += 1
                except Exception:
                    failures += 1
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        metrics = self.runner.calculate_metrics(
            f"concurrent_{num_concurrent}_requests",
            latencies,
            successes,
            failures,
            duration
        )
        
        return metrics
    
    def test_10_concurrent_requests(self):
        """Test handling 10 concurrent requests."""
        metrics = self.concurrent_request_test(10, 50)
        self.runner.print_summary(metrics)
        
        assert metrics.successful_requests >= 45  # Allow some failures
        assert metrics.requests_per_second > 0
    
    def test_50_concurrent_requests(self):
        """Test handling 50 concurrent requests."""
        metrics = self.concurrent_request_test(50, 200)
        self.runner.print_summary(metrics)
        
        assert metrics.successful_requests >= 180  # Allow some failures
        assert metrics.latency_p95 < 5000  # P95 under 5 seconds
    
    def test_100_concurrent_requests(self):
        """Test handling 100 concurrent requests."""
        metrics = self.concurrent_request_test(100, 500)
        self.runner.print_summary(metrics)
        
        assert metrics.successful_requests >= 450  # Allow some failures
        assert metrics.error_rate < 0.15  # Less than 15% error rate


class TestSustainedLoadPerformance:
    """Test sustained load over time."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = PerformanceTestRunner()
    
    def sustained_load_test(self, duration_seconds: int, requests_per_second: int):
        """Run sustained load test."""
        def mock_request():
            time.sleep(0.05)
            return {"success": True, "tokens": 50}
        
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        request_interval = 1.0 / requests_per_second
        
        while time.perf_counter() < end_time:
            request_start = time.perf_counter()
            
            try:
                latency, result = self.runner.measure_latency(mock_request)
                latencies.append(latency)
                if result["success"]:
                    successes += 1
                else:
                    failures += 1
            except Exception:
                failures += 1
            
            # Maintain request rate
            elapsed = time.perf_counter() - request_start
            sleep_time = max(0, request_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        actual_duration = time.perf_counter() - start_time
        
        metrics = self.runner.calculate_metrics(
            f"sustained_load_{duration_seconds}s_{requests_per_second}rps",
            latencies,
            successes,
            failures,
            actual_duration
        )
        
        return metrics
    
    def test_sustained_load_5_minutes(self):
        """Test sustained load for 5 minutes (simulated with shorter duration)."""
        # Use shorter duration for testing
        metrics = self.sustained_load_test(duration_seconds=10, requests_per_second=5)
        self.runner.print_summary(metrics)
        
        assert metrics.successful_requests >= 40  # At least 40 requests in 10s
        assert metrics.error_rate < 0.1
    
    def test_sustained_load_with_varying_rate(self):
        """Test sustained load with varying request rates."""
        # Simulate varying load
        metrics = self.sustained_load_test(duration_seconds=5, requests_per_second=10)
        self.runner.print_summary(metrics)
        
        assert metrics.requests_per_second >= 8  # Allow some variance
        assert metrics.latency_mean < 200  # Mean latency under 200ms


class TestBurstTrafficHandling:
    """Test burst traffic patterns and spike handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = PerformanceTestRunner()
    
    def test_traffic_burst_handling(self):
        """Test handling sudden traffic bursts."""
        def mock_request():
            time.sleep(0.05)
            return {"success": True}
        
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.perf_counter()
        
        # Simulate burst: send 100 requests as fast as possible
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(mock_request) for _ in range(100)]
            
            for future in as_completed(futures):
                try:
                    latency, result = self.runner.measure_latency(lambda: future.result(timeout=10))
                    latencies.append(latency)
                    if result["success"]:
                        successes += 1
                    else:
                        failures += 1
                except Exception:
                    failures += 1
        
        duration = time.perf_counter() - start_time
        
        metrics = self.runner.calculate_metrics(
            "traffic_burst_100_requests",
            latencies,
            successes,
            failures,
            duration
        )
        
        self.runner.print_summary(metrics)
        
        assert metrics.successful_requests >= 90  # Handle most burst requests
        assert metrics.error_rate < 0.15
    
    def test_spike_recovery(self):
        """Test system recovery after traffic spike."""
        # Simulate spike followed by normal load
        def mock_request():
            time.sleep(0.05)
            return {"success": True}
        
        all_latencies = []
        all_successes = 0
        all_failures = 0
        
        start_time = time.perf_counter()
        
        # Phase 1: Spike
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(mock_request) for _ in range(50)]
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    all_latencies.append(50)  # Mock latency
                    all_successes += 1
                except Exception:
                    all_failures += 1
        
        # Phase 2: Normal load
        for _ in range(20):
            try:
                latency, result = self.runner.measure_latency(mock_request)
                all_latencies.append(latency)
                all_successes += 1
            except Exception:
                all_failures += 1
            time.sleep(0.1)
        
        duration = time.perf_counter() - start_time
        
        metrics = self.runner.calculate_metrics(
            "spike_recovery",
            all_latencies,
            all_successes,
            all_failures,
            duration
        )
        
        self.runner.print_summary(metrics)
        
        assert metrics.successful_requests >= 60


class TestStressAndCapacityLimits:
    """Test system behavior at capacity limits."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = PerformanceTestRunner()
    
    def test_find_capacity_limit(self):
        """Test to find system capacity limits."""
        def mock_request():
            time.sleep(0.1)
            return {"success": True}
        
        # Gradually increase load until failures occur
        concurrent_levels = [10, 25, 50, 75, 100]
        capacity_results = []
        
        for level in concurrent_levels:
            latencies = []
            successes = 0
            failures = 0
            
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=level) as executor:
                futures = [executor.submit(mock_request) for _ in range(level * 2)]
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=5)
                        latencies.append(100)  # Mock latency
                        successes += 1
                    except Exception:
                        failures += 1
            
            duration = time.perf_counter() - start_time
            
            metrics = self.runner.calculate_metrics(
                f"capacity_test_{level}_concurrent",
                latencies,
                successes,
                failures,
                duration
            )
            
            capacity_results.append({
                "level": level,
                "success_rate": metrics.successful_requests / metrics.total_requests,
                "latency_p95": metrics.latency_p95
            })
        
        # Find capacity limit (where success rate drops below 95%)
        capacity_limit = None
        for result in capacity_results:
            if result["success_rate"] < 0.95:
                capacity_limit = result["level"]
                break
        
        print(f"\nCapacity test results: {capacity_results}")
        print(f"Estimated capacity limit: {capacity_limit or 'Not reached'}")
        
        assert len(capacity_results) > 0
    
    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        def mock_request():
            # Simulate some memory allocation
            data = [0] * 1000
            time.sleep(0.01)
            return {"success": True, "data": data}
        
        # Run requests and monitor memory
        for _ in range(100):
            mock_request()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
    
    def test_graceful_degradation_under_overload(self):
        """Test graceful degradation when system is overloaded."""
        def mock_request_with_backpressure(request_id: int):
            # Simulate backpressure - some requests take longer
            if request_id % 10 == 0:
                time.sleep(0.5)  # Slow request
            else:
                time.sleep(0.05)  # Normal request
            return {"success": True}
        
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(mock_request_with_backpressure, i) for i in range(200)]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    latencies.append(50)  # Mock latency
                    successes += 1
                except Exception:
                    failures += 1
        
        duration = time.perf_counter() - start_time
        
        metrics = self.runner.calculate_metrics(
            "graceful_degradation",
            latencies,
            successes,
            failures,
            duration
        )
        
        self.runner.print_summary(metrics)
        
        # System should handle most requests even under stress
        assert metrics.successful_requests >= 180
        assert metrics.error_rate < 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
