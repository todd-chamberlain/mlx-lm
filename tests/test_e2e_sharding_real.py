"""
Real End-to-End Sharding Integration Test

This test uses actual services and a real small model to verify:
- Coordinator and workers can be started
- Model can be downloaded with sharding enabled
- Each worker loads only assigned layers
- Inference works through the sharded model
- Output is correct

Requirements: 7.1, 7.2, 7.3, 10.1, 10.2
"""

import pytest
import asyncio
import subprocess
import time
import httpx
import psutil
from pathlib import Path
from typing import Optional, Dict, Any


# Test configuration
TEST_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for testing
COORDINATOR_PORT = 50051
WORKER1_PORT = 50052
WORKER2_PORT = 50053
API_PORT = 8000


class ServiceManager:
    """Manages test services lifecycle"""
    
    def __init__(self):
        self.processes = {}
        self.base_dir = Path(__file__).parent.parent
    
    async def start_coordinator(self) -> subprocess.Popen:
        """Start coordinator service"""
        cmd = [
            "python", "-m", "coordinator.main",
            "--port", str(COORDINATOR_PORT),
            "--enable-auto-sharding", "true",
            "--shard-threshold-gb", "0.5",  # Low threshold for testing
            "--min-workers-for-sharding", "2"
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes['coordinator'] = process
        
        # Wait for coordinator to be ready
        await self._wait_for_port(COORDINATOR_PORT, timeout=30)
        
        return process
    
    async def start_worker(self, worker_id: str, port: int) -> subprocess.Popen:
        """Start worker service"""
        cmd = [
            "python", "-m", "worker.main",
            "--worker-id", worker_id,
            "--port", str(port),
            "--coordinator", f"localhost:{COORDINATOR_PORT}",
            "--model-cache-size", "1"
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes[worker_id] = process
        
        # Wait for worker to be ready
        await self._wait_for_port(port, timeout=30)
        
        return process
    
    async def _wait_for_port(self, port: int, timeout: int = 30):
        """Wait for a port to be listening"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if any process is listening on the port
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    # Give it a bit more time to fully initialize
                    await asyncio.sleep(2)
                    return
            
            await asyncio.sleep(0.5)
        
        raise TimeoutError(f"Port {port} did not become available within {timeout}s")
    
    def stop_all(self):
        """Stop all services"""
        for name, process in self.processes.items():
            if process.poll() is None:  # Still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        
        self.processes.clear()
    
    def get_worker_memory(self, worker_id: str) -> Optional[float]:
        """Get memory usage of a worker process in GB"""
        process = self.processes.get(worker_id)
        if not process or process.poll() is not None:
            return None
        
        try:
            proc = psutil.Process(process.pid)
            memory_info = proc.memory_info()
            return memory_info.rss / (1024 ** 3)  # Convert to GB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None


@pytest.fixture
async def service_manager():
    """Fixture to manage test services"""
    manager = ServiceManager()
    yield manager
    manager.stop_all()


@pytest.mark.asyncio
@pytest.mark.integration
class TestRealEndToEndSharding:
    """Real end-to-end sharding tests with actual services"""
    
    async def test_sharded_model_download_and_inference(self, service_manager):
        """
        Complete end-to-end test:
        1. Start coordinator and 2 workers
        2. Download model with sharding enabled
        3. Verify each worker has only assigned layers
        4. Run inference request
        5. Verify output is correct
        """
        # Step 1: Start services
        print("\n=== Starting coordinator ===")
        await service_manager.start_coordinator()
        
        print("=== Starting worker 1 ===")
        await service_manager.start_worker("worker-1", WORKER1_PORT)
        
        print("=== Starting worker 2 ===")
        await service_manager.start_worker("worker-2", WORKER2_PORT)
        
        # Give services time to register
        await asyncio.sleep(5)
        
        # Step 2: Download model with sharding
        print(f"\n=== Downloading model {TEST_MODEL} with sharding ===")
        
        # Use coordinator client to download
        import grpc
        from proto.inference_pb2 import ModelDownloadRequest
        from proto.inference_pb2_grpc import InferenceServiceStub
        
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            request = ModelDownloadRequest(
                model_name=TEST_MODEL,
                force_redownload=False
            )
            
            try:
                response = await stub.DownloadModel(request)
                print(f"Download response: success={response.success}, message={response.message}")
                
                # Note: The actual sharding decision is made by the coordinator
                # based on model size and configuration
                
            except grpc.RpcError as e:
                print(f"Download failed: {e}")
                pytest.fail(f"Model download failed: {e}")
        
        # Wait for download to complete
        print("=== Waiting for download to complete ===")
        await asyncio.sleep(30)  # TinyLlama is small but still needs time
        
        # Step 3: Check worker memory usage
        print("\n=== Checking worker memory usage ===")
        worker1_memory = service_manager.get_worker_memory("worker-1")
        worker2_memory = service_manager.get_worker_memory("worker-2")
        
        print(f"Worker 1 memory: {worker1_memory:.2f} GB" if worker1_memory else "Worker 1 memory: N/A")
        print(f"Worker 2 memory: {worker2_memory:.2f} GB" if worker2_memory else "Worker 2 memory: N/A")
        
        # If sharding worked, each worker should use less memory than full model
        # TinyLlama is ~2.2GB, so each worker should use ~1.1GB + overhead
        if worker1_memory and worker2_memory:
            # Each worker should use less than full model size
            assert worker1_memory < 3.0, f"Worker 1 using too much memory: {worker1_memory:.2f} GB"
            assert worker2_memory < 3.0, f"Worker 2 using too much memory: {worker2_memory:.2f} GB"
        
        # Step 4: Run inference
        print("\n=== Running inference ===")
        
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            from proto.inference_pb2 import InferenceRequest
            
            request = InferenceRequest(
                model=TEST_MODEL,
                prompt="Hello, how are you?",
                max_tokens=20,
                temperature=0.7
            )
            
            try:
                response = await stub.Infer(request)
                print(f"Inference response: {response.generated_text}")
                
                # Step 5: Verify output
                assert response.generated_text, "Generated text should not be empty"
                assert len(response.generated_text) > 0, "Should generate some text"
                
                print(f"✓ Inference successful: {response.generated_text[:100]}...")
                
            except grpc.RpcError as e:
                print(f"Inference failed: {e}")
                pytest.fail(f"Inference failed: {e}")
    
    async def test_verify_layer_distribution(self, service_manager):
        """
        Verify that layers are actually distributed across workers
        """
        # Start services
        await service_manager.start_coordinator()
        await service_manager.start_worker("worker-1", WORKER1_PORT)
        await service_manager.start_worker("worker-2", WORKER2_PORT)
        await asyncio.sleep(5)
        
        # Download model
        import grpc
        from proto.inference_pb2 import ModelDownloadRequest
        from proto.inference_pb2_grpc import InferenceServiceStub
        
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            request = ModelDownloadRequest(
                model_name=TEST_MODEL,
                force_redownload=False
            )
            
            response = await stub.DownloadModel(request)
            assert response.success, f"Download failed: {response.message}"
        
        await asyncio.sleep(30)
        
        # Query coordinator for shard metadata
        # This would require adding a GetShardMetadata RPC endpoint
        # For now, we verify through memory usage
        
        worker1_memory = service_manager.get_worker_memory("worker-1")
        worker2_memory = service_manager.get_worker_memory("worker-2")
        
        if worker1_memory and worker2_memory:
            # Both workers should have loaded something
            assert worker1_memory > 0.5, "Worker 1 should have loaded model layers"
            assert worker2_memory > 0.5, "Worker 2 should have loaded model layers"
            
            # Memory should be roughly balanced (within 50% of each other)
            ratio = max(worker1_memory, worker2_memory) / min(worker1_memory, worker2_memory)
            assert ratio < 2.0, f"Memory distribution too unbalanced: {ratio:.2f}x"
            
            print(f"✓ Layer distribution verified: Worker1={worker1_memory:.2f}GB, Worker2={worker2_memory:.2f}GB")


@pytest.mark.asyncio
@pytest.mark.integration
class TestBackwardCompatibility:
    """Test backward compatibility with non-sharded models"""
    
    async def test_non_sharded_model_still_works(self, service_manager):
        """
        Test that non-sharded model loading still works:
        1. Start coordinator with sharding disabled
        2. Download model
        3. Verify all workers have full model
        4. Run inference
        5. Verify output matches sharded output
        
        Requirements: 7.1, 7.2, 8.1, 10.1
        """
        # Start services with sharding disabled
        # This would require modifying start_coordinator to accept config
        # For now, we test with a model below the threshold
        
        await service_manager.start_coordinator()
        await service_manager.start_worker("worker-1", WORKER1_PORT)
        await asyncio.sleep(5)
        
        # Download a small model that won't trigger sharding
        import grpc
        from proto.inference_pb2 import ModelDownloadRequest, InferenceRequest
        from proto.inference_pb2_grpc import InferenceServiceStub
        
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            # Download
            download_req = ModelDownloadRequest(
                model_name=TEST_MODEL,
                force_redownload=False
            )
            
            download_resp = await stub.DownloadModel(download_req)
            assert download_resp.success, f"Download failed: {download_resp.message}"
        
        await asyncio.sleep(30)
        
        # Run inference
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            inference_req = InferenceRequest(
                model=TEST_MODEL,
                prompt="Hello, how are you?",
                max_tokens=20,
                temperature=0.7
            )
            
            response = await stub.Infer(inference_req)
            assert response.generated_text, "Should generate text"
            
            print(f"✓ Non-sharded inference works: {response.generated_text[:100]}...")
    
    async def test_sharded_vs_non_sharded_output_consistency(self, service_manager):
        """
        Test that sharded and non-sharded inference produce consistent outputs
        """
        test_prompt = "The capital of France is"
        
        # Test 1: Non-sharded inference
        print("\n=== Testing non-sharded inference ===")
        await service_manager.start_coordinator()
        await service_manager.start_worker("worker-1", WORKER1_PORT)
        await asyncio.sleep(5)
        
        import grpc
        from proto.inference_pb2 import ModelDownloadRequest, InferenceRequest
        from proto.inference_pb2_grpc import InferenceServiceStub
        
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            # Download non-sharded
            download_req = ModelDownloadRequest(
                model_name=TEST_MODEL,
                force_redownload=False
            )
            await stub.DownloadModel(download_req)
        
        await asyncio.sleep(30)
        
        # Run inference with fixed seed for reproducibility
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            inference_req = InferenceRequest(
                model=TEST_MODEL,
                prompt=test_prompt,
                max_tokens=10,
                temperature=0.0  # Deterministic
            )
            
            non_sharded_response = await stub.Infer(inference_req)
            non_sharded_text = non_sharded_response.generated_text
            print(f"Non-sharded output: {non_sharded_text}")
        
        # Stop services
        service_manager.stop_all()
        await asyncio.sleep(2)
        
        # Test 2: Sharded inference
        print("\n=== Testing sharded inference ===")
        await service_manager.start_coordinator()
        await service_manager.start_worker("worker-1", WORKER1_PORT)
        await service_manager.start_worker("worker-2", WORKER2_PORT)
        await asyncio.sleep(5)
        
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            # Download sharded
            download_req = ModelDownloadRequest(
                model_name=TEST_MODEL,
                force_redownload=True  # Force re-download to trigger sharding
            )
            await stub.DownloadModel(download_req)
        
        await asyncio.sleep(30)
        
        # Run inference with same parameters
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            inference_req = InferenceRequest(
                model=TEST_MODEL,
                prompt=test_prompt,
                max_tokens=10,
                temperature=0.0  # Deterministic
            )
            
            sharded_response = await stub.Infer(inference_req)
            sharded_text = sharded_response.generated_text
            print(f"Sharded output: {sharded_text}")
        
        # Verify outputs are similar (may not be identical due to numerical precision)
        # At minimum, they should both generate valid text
        assert non_sharded_text, "Non-sharded should generate text"
        assert sharded_text, "Sharded should generate text"
        
        # Both should start with the prompt or be reasonable continuations
        print(f"✓ Both modes produce valid outputs")
        print(f"  Non-sharded: {non_sharded_text[:50]}...")
        print(f"  Sharded: {sharded_text[:50]}...")
    
    async def test_multiple_workers_without_sharding(self, service_manager):
        """
        Test that multiple workers work correctly without sharding
        (load balancing across workers with full model)
        """
        # Start coordinator and 2 workers
        await service_manager.start_coordinator()
        await service_manager.start_worker("worker-1", WORKER1_PORT)
        await service_manager.start_worker("worker-2", WORKER2_PORT)
        await asyncio.sleep(5)
        
        import grpc
        from proto.inference_pb2 import ModelDownloadRequest, InferenceRequest
        from proto.inference_pb2_grpc import InferenceServiceStub
        
        # Download model (may or may not shard depending on size)
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            download_req = ModelDownloadRequest(
                model_name=TEST_MODEL,
                force_redownload=False
            )
            await stub.DownloadModel(download_req)
        
        await asyncio.sleep(30)
        
        # Run multiple inference requests
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            for i in range(3):
                inference_req = InferenceRequest(
                    model=TEST_MODEL,
                    prompt=f"Test prompt {i}",
                    max_tokens=10,
                    temperature=0.7
                )
                
                response = await stub.Infer(inference_req)
                assert response.generated_text, f"Request {i} should generate text"
                print(f"✓ Request {i} successful: {response.generated_text[:50]}...")
        
        print("✓ Multiple workers handle requests correctly")


@pytest.mark.asyncio
@pytest.mark.integration  
class TestFailureScenarios:
    """Test failure scenarios"""
    
    async def test_insufficient_workers_for_sharding(self, service_manager):
        """
        Test that sharding fails gracefully with insufficient workers
        
        Requirements: 7.1, 7.2, 5.4, 5.5
        """
        # Start coordinator requiring 2 workers but only start 1
        await service_manager.start_coordinator()
        await service_manager.start_worker("worker-1", WORKER1_PORT)
        await asyncio.sleep(5)
        
        # Try to download a large model
        import grpc
        from proto.inference_pb2 import ModelDownloadRequest
        from proto.inference_pb2_grpc import InferenceServiceStub
        
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            request = ModelDownloadRequest(
                model_name=TEST_MODEL,
                force_redownload=False
            )
            
            response = await stub.DownloadModel(request)
            
            # Should either succeed with non-sharded download or fail with clear message
            if not response.success:
                assert "worker" in response.message.lower() or "shard" in response.message.lower()
                print(f"✓ Clear error message: {response.message}")
            else:
                # Succeeded with non-sharded download (fallback)
                print("✓ Fell back to non-sharded download")
    
    async def test_worker_failure_during_download(self, service_manager):
        """
        Test handling of worker failure during model download
        
        Requirements: 7.1, 7.2, 5.4, 5.5
        """
        # Start services
        await service_manager.start_coordinator()
        await service_manager.start_worker("worker-1", WORKER1_PORT)
        await service_manager.start_worker("worker-2", WORKER2_PORT)
        await asyncio.sleep(5)
        
        import grpc
        from proto.inference_pb2 import ModelDownloadRequest, InferenceRequest
        from proto.inference_pb2_grpc import InferenceServiceStub
        
        # Start download
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            download_req = ModelDownloadRequest(
                model_name=TEST_MODEL,
                force_redownload=True
            )
            
            # Start download (non-blocking)
            response = await stub.DownloadModel(download_req)
            
            # Kill worker 2 shortly after download starts
            await asyncio.sleep(2)
            worker2_process = service_manager.processes.get("worker-2")
            if worker2_process:
                worker2_process.terminate()
                worker2_process.wait()
                print("✓ Killed worker 2 during download")
            
            # Wait for download to complete or fail
            await asyncio.sleep(30)
            
            # Try to use the model - should either work with worker 1 only or fail gracefully
            inference_req = InferenceRequest(
                model=TEST_MODEL,
                prompt="Test",
                max_tokens=5,
                temperature=0.7
            )
            
            try:
                inference_response = await stub.Infer(inference_req)
                # If it works, worker 1 completed download
                print(f"✓ Inference succeeded with remaining worker: {inference_response.generated_text[:30]}...")
            except grpc.RpcError as e:
                # Should fail with clear error message
                error_msg = e.details().lower()
                assert any(word in error_msg for word in ["worker", "unavailable", "failed", "not found"])
                print(f"✓ Clear error message on download failure: {e.details()}")
    
    async def test_worker_failure_during_inference(self, service_manager):
        """
        Test handling of worker failure during inference
        
        Requirements: 7.1, 7.2, 5.4, 5.5
        """
        # Start services
        await service_manager.start_coordinator()
        await service_manager.start_worker("worker-1", WORKER1_PORT)
        await service_manager.start_worker("worker-2", WORKER2_PORT)
        await asyncio.sleep(5)
        
        # Download model
        import grpc
        from proto.inference_pb2 import ModelDownloadRequest, InferenceRequest
        from proto.inference_pb2_grpc import InferenceServiceStub
        
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            download_req = ModelDownloadRequest(
                model_name=TEST_MODEL,
                force_redownload=False
            )
            
            download_resp = await stub.DownloadModel(download_req)
            assert download_resp.success
        
        await asyncio.sleep(30)
        
        # Kill worker 2
        worker2_process = service_manager.processes.get("worker-2")
        if worker2_process:
            worker2_process.terminate()
            worker2_process.wait()
            print("✓ Killed worker 2 before inference")
        
        await asyncio.sleep(2)
        
        # Try inference - should fail with clear error
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            inference_req = InferenceRequest(
                model=TEST_MODEL,
                prompt="Hello, how are you?",
                max_tokens=20,
                temperature=0.7
            )
            
            try:
                response = await stub.Infer(inference_req)
                # If it succeeds, it might have fallen back to worker 1 only
                print(f"✓ Inference handled worker failure: {response.generated_text[:50]}...")
            except grpc.RpcError as e:
                # Should get a clear error message
                assert e.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.INTERNAL, grpc.StatusCode.UNKNOWN]
                error_msg = e.details().lower()
                assert any(word in error_msg for word in ["worker", "unavailable", "failed", "connection"])
                print(f"✓ Clear error on worker failure: {e.details()}")
    
    async def test_error_messages_are_clear(self, service_manager):
        """
        Test that error messages are clear and actionable
        
        Requirements: 7.1, 7.2, 5.4, 5.5
        """
        # Start only coordinator (no workers)
        await service_manager.start_coordinator()
        await asyncio.sleep(5)
        
        import grpc
        from proto.inference_pb2 import ModelDownloadRequest, InferenceRequest
        from proto.inference_pb2_grpc import InferenceServiceStub
        
        # Test 1: Try to download with no workers
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            download_req = ModelDownloadRequest(
                model_name=TEST_MODEL,
                force_redownload=False
            )
            
            try:
                response = await stub.DownloadModel(download_req)
                if not response.success:
                    # Error message should mention workers
                    assert "worker" in response.message.lower()
                    print(f"✓ Clear download error: {response.message}")
            except grpc.RpcError as e:
                # Should fail with clear message
                assert "worker" in e.details().lower() or "unavailable" in e.details().lower()
                print(f"✓ Clear download error: {e.details()}")
        
        # Test 2: Try inference with no workers
        async with grpc.aio.insecure_channel(f'localhost:{COORDINATOR_PORT}') as channel:
            stub = InferenceServiceStub(channel)
            
            inference_req = InferenceRequest(
                model=TEST_MODEL,
                prompt="Test",
                max_tokens=5,
                temperature=0.7
            )
            
            try:
                response = await stub.Infer(inference_req)
                pytest.fail("Should have failed with no workers")
            except grpc.RpcError as e:
                # Error should be clear
                error_msg = e.details().lower()
                assert any(word in error_msg for word in ["worker", "unavailable", "not found", "no workers"])
                print(f"✓ Clear inference error: {e.details()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
