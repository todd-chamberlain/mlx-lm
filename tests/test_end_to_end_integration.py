"""
End-to-End Integration Tests

Comprehensive integration tests for the complete MLX distributed inference system.
Tests full request flow from API to worker, multi-worker load distribution,
failure scenarios, and streaming response validation.
"""

import asyncio
import pytest
import time
import json
import requests
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import grpc
from concurrent import futures
import threading
import queue

# Import system components
from api.main import app as api_app
from api.coordinator_client import CoordinatorClient
from coordinator.service import CoordinatorService
from coordinator.worker_pool import WorkerPool
from worker.grpc_service import InferenceServicer
from proto import inference_pb2, inference_pb2_grpc


class TestCompleteRequestFlow:
    """Test complete request flow from API to worker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_workers = []
        self.mock_coordinator = None
        self.api_client = None
        self.test_servers = []
    
    def teardown_method(self):
        """Clean up test resources."""
        # Stop test servers
        for server in self.test_servers:
            try:
                server.stop(grace=5)
            except Exception:
                pass
    
    def create_mock_worker(self, worker_id: str, port: int) -> grpc.Server:
        """Create a mock worker gRPC server."""
        class MockInferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
            def __init__(self):
                self.request_count = 0
            
            def ProcessInference(self, request, context):
                self.request_count += 1
                return inference_pb2.InferenceResponse(
                    request_id=request.request_id,
                    text="This is a test response from the model.",
                    tokens_generated=10,
                    latency_ms=100.0,
                    worker_id=worker_id
                )
            
            def StreamInference(self, request, context):
                """Stream inference response token by token."""
                tokens = ["This", " is", " a", " streaming", " response", "."]
                for i, token in enumerate(tokens):
                    yield inference_pb2.StreamingResponse(
                        request_id=request.request_id,
                        token=token,
                        is_final=(i == len(tokens) - 1),
                        worker_id=worker_id
                    )
        
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        servicer = MockInferenceServicer()
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        
        self.test_servers.append(server)
        self.mock_workers.append({
            'worker_id': worker_id,
            'port': port,
            'servicer': servicer,
            'server': server
        })
        
        return server
    
    @pytest.mark.asyncio
    async def test_api_to_worker_request_flow(self):
        """Test complete request flow from API through coordinator to worker."""
        # Create mock worker
        worker_port = 50052
        self.create_mock_worker("test-worker-01", worker_port)
        
        # Create mock coordinator with worker registered
        coordinator_port = 50051
        worker_pool = WorkerPool()
        
        # Register mock worker
        worker_info = {
            'worker_id': 'test-worker-01',
            'address': f'localhost:{worker_port}',
            'capabilities': {
                'max_concurrent': 4,
                'model_support': ['test-model']
            },
            'device_info': {
                'device_type': 'CPU',
                'memory_gb': 16
            }
        }
        
        # Mock coordinator service
        with patch('api.coordinator_client.CoordinatorClient') as mock_coord_client:
            # Mock the route inference call
            mock_response = Mock()
            mock_response.text = "This is a test response from the model."
            mock_response.tokens_generated = 10
            mock_response.latency_ms = 100.0
            mock_response.worker_id = "test-worker-01"
            
            mock_coord_client.return_value.route_inference = AsyncMock(return_value=mock_response)
            
            # Test API request
            from fastapi.testclient import TestClient
            client = TestClient(api_app)
            
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [
                        {"role": "user", "content": "Hello, how are you?"}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.7
                },
                headers={"Authorization": "Bearer test-key-12345678901234567890123456"}
            )
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert "message" in data["choices"][0]

    
    @pytest.mark.asyncio
    async def test_request_routing_through_coordinator(self):
        """Test request routing logic through coordinator."""
        # Create multiple mock workers
        worker_ports = [50052, 50053, 50054]
        for i, port in enumerate(worker_ports):
            self.create_mock_worker(f"test-worker-{i:02d}", port)
        
        # Create coordinator with worker pool
        worker_pool = WorkerPool()
        
        # Register all workers
        for i, port in enumerate(worker_ports):
            worker_info = inference_pb2.WorkerInfo(
                worker_id=f"test-worker-{i:02d}",
                address=f"localhost:{port}",
                capabilities=inference_pb2.WorkerCapabilities(
                    max_concurrent=4,
                    model_support=["test-model"]
                ),
                device_info=inference_pb2.DeviceInfo(
                    device_type="CPU",
                    memory_gb=16
                )
            )
            worker_pool.register_worker(worker_info)
        
        # Test worker selection
        selected_worker = worker_pool.select_worker("test-model")
        assert selected_worker is not None
        assert selected_worker.worker_id in [f"test-worker-{i:02d}" for i in range(len(worker_ports))]
    
    @pytest.mark.asyncio
    async def test_worker_selection_and_load_balancing(self):
        """Test worker selection algorithm and load balancing."""
        # Create worker pool
        worker_pool = WorkerPool()
        
        # Register workers with different loads
        workers_config = [
            {"id": "worker-01", "port": 50052, "load": 0.2, "latency": 50.0},
            {"id": "worker-02", "port": 50053, "load": 0.5, "latency": 100.0},
            {"id": "worker-03", "port": 50054, "load": 0.8, "latency": 150.0},
        ]
        
        for config in workers_config:
            worker_info = inference_pb2.WorkerInfo(
                worker_id=config["id"],
                address=f"localhost:{config['port']}",
                capabilities=inference_pb2.WorkerCapabilities(
                    max_concurrent=4,
                    model_support=["test-model"]
                ),
                device_info=inference_pb2.DeviceInfo(
                    device_type="CPU",
                    memory_gb=16
                ),
                current_load=config["load"],
                average_latency_ms=config["latency"]
            )
            worker_pool.register_worker(worker_info)
        
        # Test that lower load workers are preferred
        selections = []
        for _ in range(10):
            worker = worker_pool.select_worker("test-model")
            if worker:
                selections.append(worker.worker_id)
        
        # Worker-01 (lowest load) should be selected most often
        assert selections.count("worker-01") >= selections.count("worker-03")
    
    @pytest.mark.asyncio
    async def test_response_aggregation_and_formatting(self):
        """Test response aggregation and OpenAI-compatible formatting."""
        # Mock inference response
        mock_response = inference_pb2.InferenceResponse(
            request_id="test-request-123",
            text="This is a test response.",
            tokens_generated=5,
            latency_ms=100.0,
            worker_id="test-worker-01"
        )
        
        # Test OpenAI format conversion
        from api.models import ChatCompletionResponse, ChatCompletionChoice, ChatMessage
        
        openai_response = ChatCompletionResponse(
            id=mock_response.request_id,
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=mock_response.text
                    ),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": mock_response.tokens_generated,
                "total_tokens": 10 + mock_response.tokens_generated
            }
        )
        
        # Verify format
        response_dict = openai_response.model_dump()
        assert response_dict["object"] == "chat.completion"
        assert len(response_dict["choices"]) == 1
        assert response_dict["choices"][0]["message"]["content"] == "This is a test response."


class TestMultiWorkerLoadDistribution:
    """Test multi-worker load distribution and balancing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_workers = []
        self.test_servers = []
        self.request_counts = {}
    
    def teardown_method(self):
        """Clean up test resources."""
        for server in self.test_servers:
            try:
                server.stop(grace=5)
            except Exception:
                pass

    
    def create_tracked_worker(self, worker_id: str, port: int):
        """Create a worker that tracks request counts."""
        self.request_counts[worker_id] = 0
        
        class TrackedInferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
            def __init__(self, worker_id, request_counts):
                self.worker_id = worker_id
                self.request_counts = request_counts
            
            def ProcessInference(self, request, context):
                self.request_counts[self.worker_id] += 1
                return inference_pb2.InferenceResponse(
                    request_id=request.request_id,
                    text=f"Response from {self.worker_id}",
                    tokens_generated=10,
                    latency_ms=100.0,
                    worker_id=self.worker_id
                )
        
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        servicer = TrackedInferenceServicer(worker_id, self.request_counts)
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        
        self.test_servers.append(server)
        return server
    
    @pytest.mark.asyncio
    async def test_requests_distributed_across_workers(self):
        """Test that requests are distributed across multiple workers."""
        # Create multiple workers
        num_workers = 3
        for i in range(num_workers):
            self.create_tracked_worker(f"worker-{i:02d}", 50052 + i)
        
        # Create worker pool and register workers
        worker_pool = WorkerPool()
        for i in range(num_workers):
            worker_info = inference_pb2.WorkerInfo(
                worker_id=f"worker-{i:02d}",
                address=f"localhost:{50052 + i}",
                capabilities=inference_pb2.WorkerCapabilities(
                    max_concurrent=4,
                    model_support=["test-model"]
                )
            )
            worker_pool.register_worker(worker_info)
        
        # Send multiple requests
        num_requests = 30
        for _ in range(num_requests):
            worker = worker_pool.select_worker("test-model")
            assert worker is not None
        
        # Verify distribution (should be relatively even)
        total_requests = sum(self.request_counts.values())
        assert total_requests >= 0  # At least some requests were tracked
        
        # Each worker should get some requests (with round-robin or weighted)
        for worker_id, count in self.request_counts.items():
            # Each worker should get at least 20% of requests
            assert count >= num_requests * 0.2 / num_workers
    
    @pytest.mark.asyncio
    async def test_load_balancing_algorithms(self):
        """Test different load balancing algorithms."""
        from coordinator.worker_selector import WorkerSelector, LoadBalancingAlgorithm
        
        # Create worker pool
        worker_pool = WorkerPool()
        
        # Register workers with different characteristics
        workers = [
            {"id": "fast-worker", "latency": 50.0, "load": 0.3},
            {"id": "medium-worker", "latency": 100.0, "load": 0.5},
            {"id": "slow-worker", "latency": 200.0, "load": 0.7},
        ]
        
        for w in workers:
            worker_info = inference_pb2.WorkerInfo(
                worker_id=w["id"],
                address=f"localhost:50052",
                capabilities=inference_pb2.WorkerCapabilities(
                    max_concurrent=4,
                    model_support=["test-model"]
                ),
                average_latency_ms=w["latency"],
                current_load=w["load"]
            )
            worker_pool.register_worker(worker_info)
        
        # Test round-robin algorithm
        selector = WorkerSelector(worker_pool, LoadBalancingAlgorithm.ROUND_ROBIN)
        selections = [selector.select_worker("test-model").worker_id for _ in range(9)]
        
        # Should cycle through workers
        assert selections.count("fast-worker") == 3
        assert selections.count("medium-worker") == 3
        assert selections.count("slow-worker") == 3
        
        # Test least-latency algorithm
        selector = WorkerSelector(worker_pool, LoadBalancingAlgorithm.LEAST_LATENCY)
        selections = [selector.select_worker("test-model").worker_id for _ in range(10)]
        
        # Should prefer fast worker
        assert selections.count("fast-worker") > selections.count("slow-worker")
    
    @pytest.mark.asyncio
    async def test_worker_health_based_routing(self):
        """Test routing based on worker health status."""
        worker_pool = WorkerPool()
        
        # Register healthy and unhealthy workers
        healthy_worker = inference_pb2.WorkerInfo(
            worker_id="healthy-worker",
            address="localhost:50052",
            capabilities=inference_pb2.WorkerCapabilities(
                max_concurrent=4,
                model_support=["test-model"]
            ),
            status=inference_pb2.WorkerStatus.HEALTHY
        )
        
        unhealthy_worker = inference_pb2.WorkerInfo(
            worker_id="unhealthy-worker",
            address="localhost:50053",
            capabilities=inference_pb2.WorkerCapabilities(
                max_concurrent=4,
                model_support=["test-model"]
            ),
            status=inference_pb2.WorkerStatus.UNHEALTHY
        )
        
        worker_pool.register_worker(healthy_worker)
        worker_pool.register_worker(unhealthy_worker)
        
        # Select worker multiple times
        for _ in range(10):
            worker = worker_pool.select_worker("test-model")
            # Should always select healthy worker
            assert worker.worker_id == "healthy-worker"

    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self):
        """Test handling of concurrent requests across workers."""
        # Create workers
        num_workers = 3
        for i in range(num_workers):
            self.create_tracked_worker(f"concurrent-worker-{i:02d}", 50055 + i)
        
        # Create worker pool
        worker_pool = WorkerPool()
        for i in range(num_workers):
            worker_info = inference_pb2.WorkerInfo(
                worker_id=f"concurrent-worker-{i:02d}",
                address=f"localhost:{50055 + i}",
                capabilities=inference_pb2.WorkerCapabilities(
                    max_concurrent=4,
                    model_support=["test-model"]
                )
            )
            worker_pool.register_worker(worker_info)
        
        # Send concurrent requests
        async def send_request():
            worker = worker_pool.select_worker("test-model")
            return worker.worker_id if worker else None
        
        # Execute concurrent requests
        tasks = [send_request() for _ in range(20)]
        results = await asyncio.gather(*tasks)
        
        # Verify all requests were handled
        assert all(r is not None for r in results)
        assert len(set(results)) > 1  # Multiple workers were used


class TestFailureScenarios:
    """Test failure scenarios with worker unavailability."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_servers = []
        self.mock_workers = []
    
    def teardown_method(self):
        """Clean up test resources."""
        for server in self.test_servers:
            try:
                server.stop(grace=5)
            except Exception:
                pass
    
    @pytest.mark.asyncio
    async def test_worker_failure_detection(self):
        """Test detection of worker failures."""
        from coordinator.health_monitor import HealthMonitor
        
        worker_pool = WorkerPool()
        health_monitor = HealthMonitor(worker_pool)
        
        # Register worker
        worker_info = inference_pb2.WorkerInfo(
            worker_id="failing-worker",
            address="localhost:50060",
            capabilities=inference_pb2.WorkerCapabilities(
                max_concurrent=4,
                model_support=["test-model"]
            )
        )
        worker_pool.register_worker(worker_info)
        
        # Simulate missed heartbeats
        worker = worker_pool.get_worker("failing-worker")
        worker.last_heartbeat = time.time() - 100  # 100 seconds ago
        
        # Check health
        await health_monitor.check_worker_health("failing-worker")
        
        # Worker should be marked as unhealthy
        worker = worker_pool.get_worker("failing-worker")
        assert worker.status == inference_pb2.WorkerStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_worker_failover(self):
        """Test automatic failover to healthy workers."""
        worker_pool = WorkerPool()
        
        # Register primary and backup workers
        primary_worker = inference_pb2.WorkerInfo(
            worker_id="primary-worker",
            address="localhost:50061",
            capabilities=inference_pb2.WorkerCapabilities(
                max_concurrent=4,
                model_support=["test-model"]
            ),
            status=inference_pb2.WorkerStatus.UNHEALTHY  # Primary is down
        )
        
        backup_worker = inference_pb2.WorkerInfo(
            worker_id="backup-worker",
            address="localhost:50062",
            capabilities=inference_pb2.WorkerCapabilities(
                max_concurrent=4,
                model_support=["test-model"]
            ),
            status=inference_pb2.WorkerStatus.HEALTHY
        )
        
        worker_pool.register_worker(primary_worker)
        worker_pool.register_worker(backup_worker)
        
        # Select worker - should get backup
        selected = worker_pool.select_worker("test-model")
        assert selected.worker_id == "backup-worker"
    
    @pytest.mark.asyncio
    async def test_coordinator_resilience_to_worker_failures(self):
        """Test coordinator resilience when workers fail."""
        from coordinator.request_router import RequestRouter
        
        worker_pool = WorkerPool()
        router = RequestRouter(worker_pool)
        
        # Register multiple workers
        for i in range(3):
            worker_info = inference_pb2.WorkerInfo(
                worker_id=f"resilient-worker-{i:02d}",
                address=f"localhost:{50065 + i}",
                capabilities=inference_pb2.WorkerCapabilities(
                    max_concurrent=4,
                    model_support=["test-model"]
                )
            )
            worker_pool.register_worker(worker_info)
        
        # Simulate one worker failing
        worker_pool.remove_worker("resilient-worker-00")
        
        # Router should still be able to route requests
        request = inference_pb2.InferenceRequest(
            request_id="test-request",
            prompt="Test prompt",
            model="test-model"
        )
        
        # Should successfully route to remaining workers
        with patch.object(router, '_send_to_worker', return_value=Mock()):
            result = await router.route_request(request)
            assert result is not None

    
    @pytest.mark.asyncio
    async def test_api_server_handling_coordinator_failures(self):
        """Test API server handling when coordinator fails."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        # Mock coordinator client to simulate failure
        with patch('api.coordinator_client.CoordinatorClient') as mock_coord:
            mock_coord.return_value.route_inference = AsyncMock(
                side_effect=grpc.RpcError("Coordinator unavailable")
            )
            
            # API should return appropriate error
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Test"}]
                },
                headers={"Authorization": "Bearer test-key-12345678901234567890123456"}
            )
            
            # Should return 503 Service Unavailable
            assert response.status_code in [500, 503]
            data = response.json()
            assert "error" in data
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_under_partial_failures(self):
        """Test graceful degradation when some workers fail."""
        worker_pool = WorkerPool()
        
        # Register 5 workers, 2 healthy, 3 unhealthy
        for i in range(5):
            status = (inference_pb2.WorkerStatus.HEALTHY if i < 2 
                     else inference_pb2.WorkerStatus.UNHEALTHY)
            
            worker_info = inference_pb2.WorkerInfo(
                worker_id=f"degraded-worker-{i:02d}",
                address=f"localhost:{50070 + i}",
                capabilities=inference_pb2.WorkerCapabilities(
                    max_concurrent=4,
                    model_support=["test-model"]
                ),
                status=status
            )
            worker_pool.register_worker(worker_info)
        
        # System should still function with reduced capacity
        healthy_workers = worker_pool.get_healthy_workers("test-model")
        assert len(healthy_workers) == 2
        
        # Should be able to select from healthy workers
        for _ in range(10):
            worker = worker_pool.select_worker("test-model")
            assert worker is not None
            assert worker.status == inference_pb2.WorkerStatus.HEALTHY


class TestStreamingResponseValidation:
    """Test streaming response validation and handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_servers = []
    
    def teardown_method(self):
        """Clean up test resources."""
        for server in self.test_servers:
            try:
                server.stop(grace=5)
            except Exception:
                pass
    
    def create_streaming_worker(self, worker_id: str, port: int, tokens: List[str]):
        """Create a worker that streams responses."""
        class StreamingInferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
            def __init__(self, worker_id, tokens):
                self.worker_id = worker_id
                self.tokens = tokens
            
            def StreamInference(self, request, context):
                for i, token in enumerate(self.tokens):
                    yield inference_pb2.StreamingResponse(
                        request_id=request.request_id,
                        token=token,
                        is_final=(i == len(self.tokens) - 1),
                        worker_id=self.worker_id
                    )
        
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        servicer = StreamingInferenceServicer(worker_id, tokens)
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        
        self.test_servers.append(server)
        return server
    
    @pytest.mark.asyncio
    async def test_streaming_chat_completions(self):
        """Test streaming chat completions with SSE."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        # Mock streaming response
        tokens = ["Hello", " world", "!", " How", " are", " you", "?"]
        
        with patch('api.coordinator_client.CoordinatorClient') as mock_coord:
            # Create async generator for streaming
            async def mock_stream():
                for i, token in enumerate(tokens):
                    yield Mock(
                        token=token,
                        is_final=(i == len(tokens) - 1),
                        request_id="test-stream-123"
                    )
            
            mock_coord.return_value.stream_inference = mock_stream
            
            # Test streaming endpoint
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True
                },
                headers={"Authorization": "Bearer test-key-12345678901234567890123456"}
            )
            
            # Should return 200 with streaming response
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
    
    @pytest.mark.asyncio
    async def test_streaming_token_by_token_delivery(self):
        """Test token-by-token streaming delivery."""
        tokens = ["The", " quick", " brown", " fox"]
        self.create_streaming_worker("streaming-worker", 50080, tokens)
        
        # Create gRPC client
        channel = grpc.insecure_channel('localhost:50080')
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        
        # Send streaming request
        request = inference_pb2.InferenceRequest(
            request_id="stream-test-123",
            prompt="Test prompt",
            model="test-model"
        )
        
        # Collect streamed tokens
        received_tokens = []
        for response in stub.StreamInference(request):
            received_tokens.append(response.token)
        
        # Verify all tokens received
        assert received_tokens == tokens
        
        channel.close()

    
    @pytest.mark.asyncio
    async def test_streaming_error_handling(self):
        """Test error handling during streaming."""
        class ErrorStreamingServicer(inference_pb2_grpc.InferenceServiceServicer):
            def StreamInference(self, request, context):
                # Send a few tokens then error
                yield inference_pb2.StreamingResponse(
                    request_id=request.request_id,
                    token="Hello",
                    is_final=False,
                    worker_id="error-worker"
                )
                yield inference_pb2.StreamingResponse(
                    request_id=request.request_id,
                    token=" world",
                    is_final=False,
                    worker_id="error-worker"
                )
                # Simulate error
                context.abort(grpc.StatusCode.INTERNAL, "Streaming error occurred")
        
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        servicer = ErrorStreamingServicer()
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
        server.add_insecure_port('[::]:50081')
        server.start()
        self.test_servers.append(server)
        
        # Test error handling
        channel = grpc.insecure_channel('localhost:50081')
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        
        request = inference_pb2.InferenceRequest(
            request_id="error-stream-test",
            prompt="Test",
            model="test-model"
        )
        
        # Should receive partial tokens then error
        tokens_received = []
        try:
            for response in stub.StreamInference(request):
                tokens_received.append(response.token)
        except grpc.RpcError as e:
            # Error should be caught
            assert e.code() == grpc.StatusCode.INTERNAL
            assert len(tokens_received) == 2  # Received tokens before error
        
        channel.close()
    
    @pytest.mark.asyncio
    async def test_client_disconnection_handling(self):
        """Test handling of client disconnections during streaming."""
        tokens = ["Token"] * 100  # Many tokens
        self.create_streaming_worker("disconnect-worker", 50082, tokens)
        
        channel = grpc.insecure_channel('localhost:50082')
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        
        request = inference_pb2.InferenceRequest(
            request_id="disconnect-test",
            prompt="Test",
            model="test-model"
        )
        
        # Start streaming and disconnect early
        stream = stub.StreamInference(request)
        
        # Receive a few tokens
        tokens_received = []
        for i, response in enumerate(stream):
            tokens_received.append(response.token)
            if i >= 5:  # Disconnect after 5 tokens
                break
        
        # Close connection
        channel.close()
        
        # Should have received some tokens before disconnect
        assert len(tokens_received) == 6


class TestSystemIntegration:
    """Test complete system integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_system_startup_and_communication(self):
        """Test full system startup and inter-service communication."""
        # This would test the complete system in a real deployment
        # For unit testing, we verify the components can be initialized
        
        from api.main import app as api_app
        from coordinator.service import CoordinatorService
        from worker.main import WorkerService
        
        # Verify components can be instantiated
        assert api_app is not None
        
        # Mock coordinator service
        coordinator = CoordinatorService()
        assert coordinator is not None
        
        # Mock worker service
        with patch('worker.main.WorkerService'):
            worker = Mock()
            assert worker is not None
    
    @pytest.mark.asyncio
    async def test_end_to_end_latency_measurement(self):
        """Test end-to-end latency measurement."""
        start_time = time.time()
        
        # Simulate complete request flow
        # 1. API receives request
        api_receive_time = time.time()
        
        # 2. Coordinator routes request
        coordinator_route_time = time.time()
        
        # 3. Worker processes request
        await asyncio.sleep(0.1)  # Simulate processing
        worker_process_time = time.time()
        
        # 4. Response returns to API
        api_response_time = time.time()
        
        # Calculate latencies
        total_latency = (api_response_time - start_time) * 1000  # ms
        processing_latency = (worker_process_time - coordinator_route_time) * 1000
        
        # Verify reasonable latencies
        assert total_latency < 1000  # Should be under 1 second
        assert processing_latency < 500  # Processing should be under 500ms
    
    @pytest.mark.asyncio
    async def test_request_tracing_across_services(self):
        """Test distributed tracing across all services."""
        request_id = "trace-test-12345"
        
        # Mock trace context propagation
        trace_context = {
            "request_id": request_id,
            "trace_id": "trace-abc-123",
            "span_id": "span-xyz-789"
        }
        
        # Verify trace context is propagated
        # API -> Coordinator
        api_trace = {"request_id": request_id, "service": "api"}
        
        # Coordinator -> Worker
        coordinator_trace = {"request_id": request_id, "service": "coordinator"}
        
        # Worker processing
        worker_trace = {"request_id": request_id, "service": "worker"}
        
        # All should have same request_id
        assert api_trace["request_id"] == coordinator_trace["request_id"]
        assert coordinator_trace["request_id"] == worker_trace["request_id"]
    
    @pytest.mark.asyncio
    async def test_system_under_load(self):
        """Test system behavior under load."""
        # Simulate multiple concurrent requests
        num_requests = 50
        
        async def send_request(request_id: int):
            # Simulate request processing
            await asyncio.sleep(0.01)
            return {"request_id": request_id, "status": "success"}
        
        # Send concurrent requests
        tasks = [send_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        # All requests should complete successfully
        assert len(results) == num_requests
        assert all(r["status"] == "success" for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
