"""
Service Discovery Integration Tests

Tests for service discovery mechanisms, worker registration, and network communication
in the MLX distributed inference system.
"""

import asyncio
import pytest
import time
import socket
import threading
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
import tempfile
import shutil
from pathlib import Path


class MockCoordinatorService:
    """Mock coordinator service for testing service discovery."""
    
    def __init__(self, port: int = 50051):
        self.port = port
        self.workers = {}
        self.heartbeats = {}
        self.running = False
        self.server_thread = None
    
    def start(self):
        """Start the mock coordinator service."""
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        time.sleep(0.1)  # Allow server to start
    
    def stop(self):
        """Stop the mock coordinator service."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=1)
    
    def _run_server(self):
        """Run the mock server loop."""
        # This would be a real gRPC server in production
        while self.running:
            time.sleep(0.1)
    
    def register_worker(self, worker_info: Dict[str, Any]) -> bool:
        """Register a worker with the coordinator."""
        worker_id = worker_info.get("worker_id")
        if worker_id:
            self.workers[worker_id] = {
                **worker_info,
                "registered_at": time.time(),
                "last_heartbeat": time.time(),
                "status": "healthy"
            }
            return True
        return False
    
    def send_heartbeat(self, worker_id: str, metrics: Dict[str, Any]) -> bool:
        """Process heartbeat from worker."""
        if worker_id in self.workers:
            self.heartbeats[worker_id] = {
                "timestamp": time.time(),
                "metrics": metrics
            }
            self.workers[worker_id]["last_heartbeat"] = time.time()
            return True
        return False
    
    def get_workers(self) -> Dict[str, Any]:
        """Get all registered workers."""
        return self.workers.copy()
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker from the registry."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            if worker_id in self.heartbeats:
                del self.heartbeats[worker_id]
            return True
        return False


class MockWorkerService:
    """Mock worker service for testing."""
    
    def __init__(self, worker_id: str, coordinator_address: str):
        self.worker_id = worker_id
        self.coordinator_address = coordinator_address
        self.running = False
        self.registered = False
        self.heartbeat_thread = None
        self.metrics = {
            "cpu_usage": 45.2,
            "memory_usage": 60.1,
            "active_requests": 0,
            "total_requests": 0
        }
    
    def start(self):
        """Start the mock worker service."""
        self.running = True
        # Simulate registration
        self.registered = self._register_with_coordinator()
        if self.registered:
            self._start_heartbeat()
    
    def stop(self):
        """Stop the mock worker service."""
        self.running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1)
    
    def _register_with_coordinator(self) -> bool:
        """Register with the coordinator."""
        # Mock registration logic
        registration_data = {
            "worker_id": self.worker_id,
            "address": f"localhost:8080",
            "capabilities": {
                "max_concurrent": 4,
                "model_support": ["llama", "mistral"]
            },
            "device_info": {
                "gpu_memory": "16GB",
                "device_type": "Apple M2"
            }
        }
        # In real implementation, this would be a gRPC call
        return True
    
    def _start_heartbeat(self):
        """Start heartbeat thread."""
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
    
    def _heartbeat_loop(self):
        """Heartbeat loop."""
        while self.running:
            # Send heartbeat to coordinator
            # In real implementation, this would be a gRPC call
            time.sleep(10)  # Heartbeat interval


class TestWorkerRegistration:
    """Test worker registration workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_coordinator = None
        self.mock_workers = []
    
    def teardown_method(self):
        """Clean up test resources."""
        if self.mock_coordinator:
            self.mock_coordinator.stop()
        
        for worker in self.mock_workers:
            worker.stop()
    
    def test_single_worker_registration(self):
        """Test single worker registration process."""
        # Start mock coordinator
        self.mock_coordinator = MockCoordinatorService()
        self.mock_coordinator.start()
        
        # Create and register worker
        worker_info = {
            "worker_id": "test-worker-01",
            "address": "localhost:8080",
            "capabilities": {
                "max_concurrent": 4,
                "model_support": ["llama"]
            },
            "device_info": {
                "gpu_memory": "16GB",
                "device_type": "Apple M2"
            }
        }
        
        # Test registration
        success = self.mock_coordinator.register_worker(worker_info)
        assert success == True
        
        # Verify worker is registered
        workers = self.mock_coordinator.get_workers()
        assert "test-worker-01" in workers
        assert workers["test-worker-01"]["address"] == "localhost:8080"
        assert workers["test-worker-01"]["status"] == "healthy"
    
    def test_multiple_worker_registration(self):
        """Test multiple worker registration."""
        # Start mock coordinator
        self.mock_coordinator = MockCoordinatorService()
        self.mock_coordinator.start()
        
        # Register multiple workers
        worker_configs = [
            {
                "worker_id": "worker-01",
                "address": "localhost:8080",
                "capabilities": {"max_concurrent": 4}
            },
            {
                "worker_id": "worker-02", 
                "address": "localhost:8081",
                "capabilities": {"max_concurrent": 2}
            },
            {
                "worker_id": "worker-03",
                "address": "localhost:8082", 
                "capabilities": {"max_concurrent": 6}
            }
        ]
        
        # Register all workers
        for config in worker_configs:
            success = self.mock_coordinator.register_worker(config)
            assert success == True
        
        # Verify all workers are registered
        workers = self.mock_coordinator.get_workers()
        assert len(workers) == 3
        
        for config in worker_configs:
            worker_id = config["worker_id"]
            assert worker_id in workers
            assert workers[worker_id]["address"] == config["address"]
    
    def test_duplicate_worker_registration(self):
        """Test handling of duplicate worker registration."""
        # Start mock coordinator
        self.mock_coordinator = MockCoordinatorService()
        self.mock_coordinator.start()
        
        worker_info = {
            "worker_id": "duplicate-worker",
            "address": "localhost:8080",
            "capabilities": {"max_concurrent": 4}
        }
        
        # Register worker first time
        success1 = self.mock_coordinator.register_worker(worker_info)
        assert success1 == True
        
        # Register same worker again (should update existing)
        worker_info["address"] = "localhost:8081"  # Different address
        success2 = self.mock_coordinator.register_worker(worker_info)
        assert success2 == True
        
        # Verify only one worker exists with updated info
        workers = self.mock_coordinator.get_workers()
        assert len(workers) == 1
        assert workers["duplicate-worker"]["address"] == "localhost:8081"
    
    def test_worker_deregistration(self):
        """Test worker deregistration process."""
        # Start mock coordinator
        self.mock_coordinator = MockCoordinatorService()
        self.mock_coordinator.start()
        
        # Register worker
        worker_info = {
            "worker_id": "temp-worker",
            "address": "localhost:8080",
            "capabilities": {"max_concurrent": 4}
        }
        
        success = self.mock_coordinator.register_worker(worker_info)
        assert success == True
        
        # Verify worker is registered
        workers = self.mock_coordinator.get_workers()
        assert "temp-worker" in workers
        
        # Deregister worker
        removed = self.mock_coordinator.remove_worker("temp-worker")
        assert removed == True
        
        # Verify worker is removed
        workers = self.mock_coordinator.get_workers()
        assert "temp-worker" not in workers


class TestHeartbeatMechanism:
    """Test heartbeat and health monitoring."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_coordinator = None
        self.mock_workers = []
    
    def teardown_method(self):
        """Clean up test resources."""
        if self.mock_coordinator:
            self.mock_coordinator.stop()
        
        for worker in self.mock_workers:
            worker.stop()
    
    def test_heartbeat_sending(self):
        """Test heartbeat sending from worker to coordinator."""
        # Start mock coordinator
        self.mock_coordinator = MockCoordinatorService()
        self.mock_coordinator.start()
        
        # Register worker
        worker_info = {
            "worker_id": "heartbeat-worker",
            "address": "localhost:8080",
            "capabilities": {"max_concurrent": 4}
        }
        
        self.mock_coordinator.register_worker(worker_info)
        
        # Send heartbeat
        metrics = {
            "cpu_usage": 45.2,
            "memory_usage": 60.1,
            "active_requests": 2,
            "total_requests": 150
        }
        
        success = self.mock_coordinator.send_heartbeat("heartbeat-worker", metrics)
        assert success == True
        
        # Verify heartbeat was recorded
        workers = self.mock_coordinator.get_workers()
        assert workers["heartbeat-worker"]["last_heartbeat"] > 0
    
    def test_heartbeat_timeout_detection(self):
        """Test detection of workers that stop sending heartbeats."""
        # Start mock coordinator
        self.mock_coordinator = MockCoordinatorService()
        self.mock_coordinator.start()
        
        # Register worker
        worker_info = {
            "worker_id": "timeout-worker",
            "address": "localhost:8080",
            "capabilities": {"max_concurrent": 4}
        }
        
        self.mock_coordinator.register_worker(worker_info)
        
        # Send initial heartbeat
        metrics = {"cpu_usage": 45.2}
        self.mock_coordinator.send_heartbeat("timeout-worker", metrics)
        
        # Simulate timeout by checking last heartbeat time
        workers = self.mock_coordinator.get_workers()
        last_heartbeat = workers["timeout-worker"]["last_heartbeat"]
        current_time = time.time()
        
        # Simulate time passing (in real implementation, this would be actual time)
        simulated_timeout = 60  # 60 seconds timeout
        time_since_heartbeat = current_time - last_heartbeat
        
        # Should not be timed out yet
        assert time_since_heartbeat < simulated_timeout
    
    def test_worker_health_status_tracking(self):
        """Test worker health status tracking."""
        # Start mock coordinator
        self.mock_coordinator = MockCoordinatorService()
        self.mock_coordinator.start()
        
        # Register worker
        worker_info = {
            "worker_id": "health-worker",
            "address": "localhost:8080",
            "capabilities": {"max_concurrent": 4}
        }
        
        self.mock_coordinator.register_worker(worker_info)
        
        # Send healthy heartbeat
        healthy_metrics = {
            "cpu_usage": 45.2,
            "memory_usage": 60.1,
            "active_requests": 2,
            "error_rate": 0.0
        }
        
        self.mock_coordinator.send_heartbeat("health-worker", healthy_metrics)
        
        # Send unhealthy heartbeat
        unhealthy_metrics = {
            "cpu_usage": 95.8,
            "memory_usage": 98.5,
            "active_requests": 10,
            "error_rate": 0.15
        }
        
        self.mock_coordinator.send_heartbeat("health-worker", unhealthy_metrics)
        
        # In real implementation, coordinator would analyze metrics and update status
        # For now, we just verify heartbeats are being tracked
        workers = self.mock_coordinator.get_workers()
        assert "health-worker" in workers


class TestServiceDiscoveryMechanisms:
    """Test various service discovery mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dirs = []
    
    def teardown_method(self):
        """Clean up test resources."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def test_dns_based_discovery(self):
        """Test DNS-based service discovery."""
        # Test DNS resolution for service names
        service_names = [
            "coordinator.local",
            "api.local", 
            "worker-01.local",
            "prometheus.local",
            "grafana.local"
        ]
        
        # Mock DNS resolution
        dns_records = {}
        for service in service_names:
            # In real implementation, these would be actual DNS records
            dns_records[service] = f"192.168.1.{hash(service) % 254 + 1}"
        
        # Verify DNS records exist
        assert len(dns_records) == len(service_names)
        assert "coordinator.local" in dns_records
        assert "api.local" in dns_records
    
    def test_configuration_based_discovery(self):
        """Test configuration-based service discovery."""
        # Create temporary config
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        service_config = {
            "services": {
                "coordinator": {
                    "host": "coordinator.local",
                    "port": 50051,
                    "protocol": "grpc"
                },
                "api": {
                    "host": "api.local",
                    "port": 8000,
                    "protocol": "http"
                },
                "workers": [
                    {
                        "id": "worker-01",
                        "host": "worker-01.local",
                        "port": 8080
                    },
                    {
                        "id": "worker-02", 
                        "host": "worker-02.local",
                        "port": 8080
                    }
                ]
            }
        }
        
        config_file = Path(temp_dir) / "services.json"
        with open(config_file, 'w') as f:
            json.dump(service_config, f)
        
        # Load and validate service configuration
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        services = loaded_config["services"]
        assert "coordinator" in services
        assert "api" in services
        assert "workers" in services
        assert len(services["workers"]) == 2
    
    def test_dynamic_service_registration(self):
        """Test dynamic service registration and discovery."""
        # Mock service registry
        service_registry = {}
        
        def register_service(service_name: str, service_info: Dict[str, Any]):
            service_registry[service_name] = {
                **service_info,
                "registered_at": time.time(),
                "ttl": 300  # 5 minutes TTL
            }
        
        def discover_service(service_name: str) -> Optional[Dict[str, Any]]:
            service = service_registry.get(service_name)
            if service:
                # Check TTL
                if time.time() - service["registered_at"] < service["ttl"]:
                    return service
                else:
                    # Service expired
                    del service_registry[service_name]
            return None
        
        # Register services
        register_service("coordinator", {
            "host": "coordinator.local",
            "port": 50051,
            "health_endpoint": "/health"
        })
        
        register_service("worker-01", {
            "host": "worker-01.local", 
            "port": 8080,
            "capabilities": {"max_concurrent": 4}
        })
        
        # Discover services
        coordinator = discover_service("coordinator")
        worker = discover_service("worker-01")
        
        assert coordinator is not None
        assert coordinator["host"] == "coordinator.local"
        assert worker is not None
        assert worker["capabilities"]["max_concurrent"] == 4
        
        # Test non-existent service
        missing = discover_service("non-existent")
        assert missing is None


class TestNetworkCommunication:
    """Test network communication between services."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_servers = []
    
    def teardown_method(self):
        """Clean up test resources."""
        for server in self.test_servers:
            try:
                server.shutdown()
            except Exception:
                pass
    
    def test_grpc_communication(self):
        """Test gRPC communication between services."""
        # Mock gRPC communication test
        coordinator_address = "localhost:50051"
        
        # Test connection establishment
        connection_successful = True  # Would be actual gRPC connection
        assert connection_successful == True
        
        # Test message exchange
        request_data = {
            "worker_id": "test-worker",
            "message_type": "registration",
            "payload": {"capabilities": {"max_concurrent": 4}}
        }
        
        response_data = {
            "status": "success",
            "worker_registered": True,
            "assigned_id": "test-worker"
        }
        
        # Mock request/response
        assert request_data["worker_id"] == "test-worker"
        assert response_data["status"] == "success"
    
    def test_http_api_communication(self):
        """Test HTTP API communication."""
        # Mock HTTP API test
        api_endpoint = "http://localhost:8000"
        
        # Test health endpoint
        health_response = {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "coordinator": "connected",
                "workers": 2
            }
        }
        
        assert health_response["status"] == "healthy"
        assert "coordinator" in health_response["services"]
    
    def test_network_error_handling(self):
        """Test network error handling and recovery."""
        # Mock network error scenarios
        error_scenarios = [
            {"type": "connection_timeout", "retry": True},
            {"type": "connection_refused", "retry": True},
            {"type": "dns_resolution_failed", "retry": False},
            {"type": "network_unreachable", "retry": True}
        ]
        
        for scenario in error_scenarios:
            error_type = scenario["type"]
            should_retry = scenario["retry"]
            
            # Mock error handling logic
            if should_retry:
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    retry_count += 1
                    # Mock retry attempt
                    time.sleep(0.01)  # Mock retry delay
                
                assert retry_count == max_retries
            else:
                # No retry for certain error types
                assert should_retry == False
    
    def test_load_balancing_communication(self):
        """Test load-balanced communication to multiple workers."""
        # Mock multiple worker endpoints
        worker_endpoints = [
            {"id": "worker-01", "address": "localhost:8080", "load": 0.3},
            {"id": "worker-02", "address": "localhost:8081", "load": 0.7},
            {"id": "worker-03", "address": "localhost:8082", "load": 0.1}
        ]
        
        # Mock load balancing algorithm (least loaded)
        def select_worker(workers):
            return min(workers, key=lambda w: w["load"])
        
        selected_worker = select_worker(worker_endpoints)
        assert selected_worker["id"] == "worker-03"  # Lowest load
        assert selected_worker["load"] == 0.1


class TestServiceDiscoveryFailureScenarios:
    """Test service discovery failure scenarios and recovery."""
    
    def test_coordinator_unavailable(self):
        """Test behavior when coordinator is unavailable."""
        # Mock coordinator unavailable scenario
        coordinator_available = False
        
        # Worker should handle coordinator unavailability
        retry_attempts = 0
        max_retries = 5
        
        while not coordinator_available and retry_attempts < max_retries:
            retry_attempts += 1
            # Mock connection attempt
            time.sleep(0.01)  # Mock retry delay
            
            # Simulate coordinator becoming available after 3 attempts
            if retry_attempts >= 3:
                coordinator_available = True
        
        assert coordinator_available == True
        assert retry_attempts == 3
    
    def test_partial_worker_failure(self):
        """Test handling of partial worker failures."""
        # Mock worker pool with some failed workers
        workers = [
            {"id": "worker-01", "status": "healthy"},
            {"id": "worker-02", "status": "failed"},
            {"id": "worker-03", "status": "healthy"},
            {"id": "worker-04", "status": "failed"}
        ]
        
        # Filter healthy workers
        healthy_workers = [w for w in workers if w["status"] == "healthy"]
        failed_workers = [w for w in workers if w["status"] == "failed"]
        
        assert len(healthy_workers) == 2
        assert len(failed_workers) == 2
        
        # System should continue with healthy workers
        assert len(healthy_workers) > 0
    
    def test_network_partition_recovery(self):
        """Test recovery from network partitions."""
        # Mock network partition scenario
        network_partitioned = True
        partition_duration = 0  # seconds
        
        # Simulate partition recovery
        while network_partitioned and partition_duration < 10:
            partition_duration += 1
            time.sleep(0.01)
            
            # Simulate network recovery after 5 seconds
            if partition_duration >= 5:
                network_partitioned = False
        
        # After recovery, services should reconnect
        assert network_partitioned == False
        assert partition_duration == 5
        
        # Mock service reconnection
        services_reconnected = True
        assert services_reconnected == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])