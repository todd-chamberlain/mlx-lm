"""
Smoke Tests for Deployment Validation

Quick validation tests to verify basic functionality after deployment.
These tests should run fast and catch critical issues immediately.
"""

import pytest
import requests
import time
from typing import Dict, Any


class TestAPIServerSmoke:
    """Smoke tests for API server deployment."""
    
    def setup_method(self, request):
        """Set up test fixtures."""
        self.base_url = request.config.getoption("--base-url", "http://localhost:8000")
        self.timeout = 10
    
    def test_health_endpoint(self):
        """Test that health endpoint is accessible."""
        response = requests.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
    
    def test_metrics_endpoint(self):
        """Test that metrics endpoint is accessible."""
        response = requests.get(
            f"{self.base_url}/metrics",
            timeout=self.timeout
        )
        
        assert response.status_code == 200
    
    def test_models_endpoint(self):
        """Test that models endpoint returns available models."""
        response = requests.get(
            f"{self.base_url}/v1/models",
            timeout=self.timeout
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)
    
    def test_basic_completion_request(self):
        """Test basic completion request works."""
        payload = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 10
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout
        )
        
        assert response.status_code in [200, 503]  # 503 if no workers available
        
        if response.status_code == 200:
            data = response.json()
            assert "choices" in data
            assert len(data["choices"]) > 0


class TestCoordinatorSmoke:
    """Smoke tests for coordinator service."""
    
    def setup_method(self, request):
        """Set up test fixtures."""
        self.coordinator_url = request.config.getoption(
            "--coordinator-url",
            "http://localhost:50051"
        )
    
    def test_coordinator_health(self):
        """Test coordinator health check."""
        # This would use gRPC health check
        # For now, just verify the service is reachable
        import socket
        
        host, port = self.coordinator_url.replace("http://", "").split(":")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, int(port)))
        sock.close()
        
        assert result == 0, "Coordinator service not reachable"


class TestSystemSmoke:
    """System-level smoke tests."""
    
    def setup_method(self, request):
        """Set up test fixtures."""
        self.base_url = request.config.getoption("--base-url", "http://localhost:8000")
    
    def test_end_to_end_latency(self):
        """Test end-to-end request latency is acceptable."""
        payload = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Test"}
            ],
            "max_tokens": 5
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            # Latency should be under 5 seconds for small requests
            assert latency < 5000, f"Latency too high: {latency}ms"
    
    def test_concurrent_requests(self):
        """Test system handles concurrent requests."""
        import concurrent.futures
        
        def make_request():
            payload = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5
            }
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            return response.status_code
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # At least some requests should succeed
        success_count = sum(1 for status in results if status == 200)
        assert success_count >= 3, "Too many failed requests"


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--base-url",
        action="store",
        default="http://localhost:8000",
        help="Base URL for API server"
    )
    parser.addoption(
        "--coordinator-url",
        action="store",
        default="http://localhost:50051",
        help="URL for coordinator service"
    )
    parser.addoption(
        "--env",
        action="store",
        default="local",
        help="Environment (local, staging, production)"
    )
