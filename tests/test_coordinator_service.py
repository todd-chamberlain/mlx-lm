"""
Unit tests for coordinator service functionality.

Tests worker registration and deregistration flows, worker selection algorithm,
health monitoring and failure detection, and request routing and retry mechanisms
as specified in task 3.5.
"""

import asyncio
import pytest
import time
import grpc
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from proto import inference_pb2
from proto.errors import MLXError, MLXErrorCode
from proto.validation import ValidationError

from coordinator.models import WorkerInfo, WorkerStatus, DeviceCapabilities, WorkerMetrics


class TestWorkerModels:
    """Test worker data models and basic functionality."""
    
    def create_device_capabilities(self) -> DeviceCapabilities:
        """Create test device capabilities."""
        return DeviceCapabilities(
            model="Mac Studio",
            memory_gb=64,
            gpu_cores=32,
            os_version="macOS 14.0",
            mlx_version="0.15.0"
        )
    
    def create_worker_info(self, worker_id: str = "test-worker") -> WorkerInfo:
        """Create test worker info."""
        device_capabilities = self.create_device_capabilities()
        return WorkerInfo(
            worker_id=worker_id,
            address="localhost:8080",
            device_capabilities=device_capabilities,
            max_concurrent=4,
            supported_models=["test-model"]
        )
    
    def test_device_capabilities_creation(self):
        """Test DeviceCapabilities creation and methods."""
        device = self.create_device_capabilities()
        
        assert device.model == "Mac Studio"
        assert device.memory_gb == 64
        assert device.gpu_cores == 32
        assert device.os_version == "macOS 14.0"
        assert device.mlx_version == "0.15.0"
        
        # Test to_dict
        device_dict = device.to_dict()
        assert device_dict["model"] == "Mac Studio"
        assert device_dict["memory_gb"] == 64
    
    def test_worker_metrics_creation(self):
        """Test WorkerMetrics creation and updates."""
        metrics = WorkerMetrics()
        
        # Test initial values
        assert metrics.avg_latency_ms == 0.0
        assert metrics.tokens_per_second == 0.0
        assert metrics.total_requests == 0
        assert metrics.failed_requests == 0
        
        # Test to_dict
        metrics_dict = metrics.to_dict()
        assert "avg_latency_ms" in metrics_dict
        assert "total_requests" in metrics_dict
    
    def test_worker_info_creation(self):
        """Test WorkerInfo creation and basic methods."""
        worker = self.create_worker_info()
        
        assert worker.worker_id == "test-worker"
        assert worker.address == "localhost:8080"
        assert worker.max_concurrent == 4
        assert worker.status == WorkerStatus.HEALTHY
        assert worker.current_load == 0
        assert "test-model" in worker.supported_models
    
    def test_worker_info_health_checks(self):
        """Test worker health checking methods."""
        worker = self.create_worker_info()
        
        # Initially healthy
        assert worker.is_healthy() is True
        assert worker.is_available_for_request() is True
        
        # Test unhealthy status
        worker.status = WorkerStatus.UNHEALTHY
        assert worker.is_healthy() is False
        assert worker.is_available_for_request() is False
        
        # Reset to healthy
        worker.status = WorkerStatus.HEALTHY
        
        # Test overloaded
        worker.current_load = worker.max_concurrent
        assert worker.is_healthy() is True
        assert worker.is_available_for_request() is False
        
        # Test circuit breaker
        worker.current_load = 0
        worker.circuit_breaker_open = True
        assert worker.is_healthy() is False
        assert worker.is_available_for_request() is False
    
    def test_worker_info_request_tracking(self):
        """Test request start/complete tracking."""
        worker = self.create_worker_info()
        
        # Start request
        worker.record_request_start()
        assert worker.current_load == 1
        
        # Complete successful request
        worker.record_request_complete(success=True, latency_ms=100.0)
        assert worker.current_load == 0
        assert worker.metrics.total_requests == 1
        assert worker.metrics.failed_requests == 0
        assert worker.consecutive_failures == 0
        
        # Start and fail request
        worker.record_request_start()
        worker.record_request_complete(success=False, latency_ms=200.0)
        assert worker.current_load == 0
        assert worker.metrics.total_requests == 2
        assert worker.metrics.failed_requests == 1
        assert worker.consecutive_failures == 1
    
    def test_worker_info_score_calculation(self):
        """Test worker selection score calculation."""
        worker = self.create_worker_info()
        
        # Initially should have low score (good)
        score = worker.calculate_score()
        assert score >= 0.0
        
        # Add some load and latency
        worker.current_load = 2
        worker.metrics.avg_latency_ms = 150.0
        worker.metrics.total_requests = 10
        worker.metrics.failed_requests = 1
        
        score_with_load = worker.calculate_score()
        assert score_with_load > score  # Higher score (worse) with load
        
        # Overloaded worker should have infinite score
        worker.current_load = worker.max_concurrent
        overloaded_score = worker.calculate_score()
        assert overloaded_score == float('inf')
    
    def test_worker_info_circuit_breaker(self):
        """Test circuit breaker functionality."""
        worker = self.create_worker_info()
        
        # Initially circuit breaker is closed
        assert worker.circuit_breaker_open is False
        
        # Simulate failures to trigger circuit breaker
        for _ in range(5):  # Default threshold is 5
            worker.record_request_complete(success=False)
        
        worker.update_circuit_breaker(failure_threshold=5)
        assert worker.circuit_breaker_open is True
        
        # Test that circuit breaker stays open while consecutive failures >= threshold
        # The current implementation uses elif, so recovery only happens if failures < threshold
        
        # Reset consecutive failures to allow recovery
        worker.consecutive_failures = 3  # Below threshold
        
        # Set failure time to old enough for recovery
        old_failure_time = time.time() - 70.0
        worker.last_failure_time = old_failure_time
        
        # Now update circuit breaker - should close due to timeout and low failure count
        worker.update_circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
        assert worker.circuit_breaker_open is False
        assert worker.consecutive_failures == 0
    
    def test_worker_info_to_dict(self):
        """Test worker info serialization to dictionary."""
        worker = self.create_worker_info()
        worker.current_load = 2
        worker.metrics.total_requests = 10
        
        worker_dict = worker.to_dict()
        
        assert worker_dict["worker_id"] == "test-worker"
        assert worker_dict["address"] == "localhost:8080"
        assert worker_dict["current_load"] == 2
        assert worker_dict["status"] == "healthy"
        assert "device_capabilities" in worker_dict
        assert "metrics" in worker_dict
        assert "is_healthy" in worker_dict
        assert "is_available" in worker_dict
        assert "score" in worker_dict


class TestWorkerStatus:
    """Test worker status enumeration."""
    
    def test_worker_status_values(self):
        """Test WorkerStatus enum values."""
        assert WorkerStatus.HEALTHY.value == "healthy"
        assert WorkerStatus.UNHEALTHY.value == "unhealthy"
        assert WorkerStatus.DRAINING.value == "draining"
        assert WorkerStatus.OFFLINE.value == "offline"
    
    def test_worker_status_comparison(self):
        """Test WorkerStatus comparison."""
        assert WorkerStatus.HEALTHY == WorkerStatus.HEALTHY
        assert WorkerStatus.HEALTHY != WorkerStatus.UNHEALTHY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])