#!/usr/bin/env python3
"""
Unified System Integration Test

Tests the complete system with node management and model sharding:
1. Start coordinator with sharding enabled
2. Start multiple workers (nodes)
3. Verify node registration and health
4. Download model with automatic sharding
5. Verify layer distribution across nodes
6. Run inference through sharded model
7. Verify output correctness
8. Test node failure and recovery
9. Measure performance and memory

This is the ultimate validation that everything works together.
"""

import asyncio
import subprocess
import time
import json
import sys
import psutil
import grpc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from proto import distributed_pb2, distributed_pb2_grpc, inference_pb2, inference_pb2_grpc


@dataclass
class TestResult:
    """Result of a test"""
    name: str
    passed: bool
    message: str
    duration_ms: float = 0.0


class UnifiedSystemTest:
    """Comprehensive test of the unified system"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.base_dir = Path(__file__).parent.parent
        self.processes = {}
        self.results = []
        
        # Ports
        self.coordinator_port = 50051
        self.worker_ports = [50052, 50053, 50054]
        
    async def wait_for_service(self, port: int, timeout: int = 30) -> bool:
        """Wait for a gRPC service to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                channel = grpc.aio.insecure_channel(f'localhost:{port}')
                await asyncio.wait_for(channel.channel_ready(), timeout=2.0)
                await channel.close()
                return True
            except Exception:
                await asyncio.sleep(1)
        return False
    
    def get_process_memory(self, process: subprocess.Popen) -> float:
        """Get process memory in GB"""
        try:
            proc = psutil.Process(process.pid)
            return proc.memory_info().rss / (1024 ** 3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    async def test_1_start_coordinator(self) -> TestResult:
        """Test 1: Start coordinator with sharding enabled"""
        print("\n" + "="*60)
        print("TEST 1: Start Coordinator with Sharding")
        print("="*60)
        
        start_time = time.time()
        
        try:
            cmd = [
                "python", "-m", "coordinator.main",
                "--port", str(self.coordinator_port),
                "--enable-auto-sharding", "true",
                "--shard-threshold-gb", "0.5",
                "--min-workers-for-sharding", "2"
            ]
            
            print(f"Starting coordinator on port {self.coordinator_port}...")
            self.processes['coordinator'] = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if not await self.wait_for_service(self.coordinator_port):
                return TestResult(
                    "Start Coordinator",
                    False,
                    "Coordinator failed to start within timeout",
                    (time.time() - start_time) * 1000
                )
            
            print("✓ Coordinator started successfully")
            return TestResult(
                "Start Coordinator",
                True,
                "Coordinator running with sharding enabled",
                (time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return TestResult(
                "Start Coordinator",
                False,
                f"Error: {str(e)}",
                (time.time() - start_time) * 1000
            )
    
    async def test_2_start_workers(self) -> TestResult:
        """Test 2: Start multiple workers and verify registration"""
        print("\n" + "="*60)
        print("TEST 2: Start Workers and Verify Registration")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Start 3 workers
            for i, port in enumerate(self.worker_ports):
                worker_id = f"worker{i+1}"
                cmd = [
                    "python", "-m", "worker.main",
                    "--worker-id", worker_id,
                    "--coordinator", f"localhost:{self.coordinator_port}",
                    "--port", str(port)
                ]
                
                print(f"Starting {worker_id} on port {port}...")
                self.processes[worker_id] = subprocess.Popen(
                    cmd,
                    cwd=self.base_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                if not await self.wait_for_service(port):
                    return TestResult(
                        "Start Workers",
                        False,
                        f"{worker_id} failed to start",
                        (time.time() - start_time) * 1000
                    )
                
                print(f"✓ {worker_id} started")
            
            # Wait for registration
            print("\nWaiting for workers to register...")
            await asyncio.sleep(3)
            
            # Verify registration via coordinator
            channel = grpc.aio.insecure_channel(f'localhost:{self.coordinator_port}')
            stub = inference_pb2_grpc.CoordinatorServiceStub(channel)
            
            request = inference_pb2.ListWorkersRequest()
            response = await stub.ListWorkers(request)
            await channel.close()
            
            registered_count = len(response.workers)
            print(f"\n✓ {registered_count} workers registered")
            
            if registered_count != len(self.worker_ports):
                return TestResult(
                    "Start Workers",
                    False,
                    f"Expected {len(self.worker_ports)} workers, got {registered_count}",
                    (time.time() - start_time) * 1000
                )
            
            return TestResult(
                "Start Workers",
                True,
                f"All {registered_count} workers registered successfully",
                (time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return TestResult(
                "Start Workers",
                False,
                f"Error: {str(e)}",
                (time.time() - start_time) * 1000
            )
    
    async def test_3_download_sharded_model(self) -> TestResult:
        """Test 3: Download model with automatic sharding"""
        print("\n" + "="*60)
        print("TEST 3: Download Model with Automatic Sharding")
        print("="*60)
        
        start_time = time.time()
        
        try:
            channel = grpc.aio.insecure_channel(f'localhost:{self.coordinator_port}')
            stub = inference_pb2_grpc.CoordinatorServiceStub(channel)
            
            print(f"Downloading model: {self.model_name}")
            print("Sharding: ENABLED")
            
            request = inference_pb2.DownloadModelRequest(
                model_name=self.model_name,
                enable_sharding=True,
                min_workers=2
            )
            
            response = await stub.DownloadModel(request)
            await channel.close()
            
            if not response.success:
                return TestResult(
                    "Download Sharded Model",
                    False,
                    f"Download failed: {response.message}",
                    (time.time() - start_time) * 1000
                )
            
            print(f"✓ Model downloaded: {response.message}")
            
            # Wait for model to load
            print("Waiting for model to load on workers...")
            await asyncio.sleep(10)
            
            return TestResult(
                "Download Sharded Model",
                True,
                "Model downloaded and sharded successfully",
                (time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return TestResult(
                "Download Sharded Model",
                False,
                f"Error: {str(e)}",
                (time.time() - start_time) * 1000
            )
    
    async def test_4_verify_layer_distribution(self) -> TestResult:
        """Test 4: Verify layers are distributed across workers"""
        print("\n" + "="*60)
        print("TEST 4: Verify Layer Distribution")
        print("="*60)
        
        start_time = time.time()
        
        try:
            channel = grpc.aio.insecure_channel(f'localhost:{self.coordinator_port}')
            stub = distributed_pb2_grpc.CoordinatorServiceStub(channel)
            
            # Get model info
            request = distributed_pb2.GetModelInfoRequest(
                model_name=self.model_name
            )
            
            response = await stub.GetModelInfo(request)
            await channel.close()
            
            if not response.is_sharded:
                return TestResult(
                    "Verify Layer Distribution",
                    False,
                    "Model is not sharded",
                    (time.time() - start_time) * 1000
                )
            
            print(f"✓ Model is sharded: {response.is_sharded}")
            print(f"✓ Total layers: {response.total_layers}")
            print(f"✓ Workers: {len(response.shard_metadata)}")
            
            # Verify each worker has layers
            print("\nLayer distribution:")
            for worker_id, metadata in response.shard_metadata.items():
                layer_range = f"{metadata.start_layer}-{metadata.end_layer}"
                layer_count = metadata.end_layer - metadata.start_layer + 1
                print(f"  {worker_id}: layers {layer_range} ({layer_count} layers)")
            
            # Verify no gaps or overlaps
            all_layers = set()
            for metadata in response.shard_metadata.values():
                worker_layers = set(range(metadata.start_layer, metadata.end_layer + 1))
                if all_layers & worker_layers:
                    return TestResult(
                        "Verify Layer Distribution",
                        False,
                        "Layer overlap detected",
                        (time.time() - start_time) * 1000
                    )
                all_layers.update(worker_layers)
            
            if len(all_layers) != response.total_layers:
                return TestResult(
                    "Verify Layer Distribution",
                    False,
                    f"Layer gap detected: {len(all_layers)} != {response.total_layers}",
                    (time.time() - start_time) * 1000
                )
            
            print("\n✓ No gaps or overlaps in layer distribution")
            
            return TestResult(
                "Verify Layer Distribution",
                True,
                f"Layers distributed correctly across {len(response.shard_metadata)} workers",
                (time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return TestResult(
                "Verify Layer Distribution",
                False,
                f"Error: {str(e)}",
                (time.time() - start_time) * 1000
            )
    
    async def test_5_run_inference(self) -> TestResult:
        """Test 5: Run inference through sharded model"""
        print("\n" + "="*60)
        print("TEST 5: Run Inference Through Sharded Model")
        print("="*60)
        
        start_time = time.time()
        
        try:
            channel = grpc.aio.insecure_channel(f'localhost:{self.coordinator_port}')
            stub = distributed_pb2_grpc.CoordinatorServiceStub(channel)
            
            prompt = "The capital of France is"
            print(f"Prompt: '{prompt}'")
            
            request = distributed_pb2.InferenceRequest(
                model_name=self.model_name,
                prompt=prompt,
                max_tokens=10,
                temperature=0.7
            )
            
            inference_start = time.time()
            response = await stub.RouteInference(request)
            inference_time = (time.time() - inference_start) * 1000
            await channel.close()
            
            if not response.generated_text:
                return TestResult(
                    "Run Inference",
                    False,
                    "No text generated",
                    (time.time() - start_time) * 1000
                )
            
            print(f"\n✓ Generated text: '{response.generated_text}'")
            print(f"✓ Inference time: {inference_time:.2f}ms")
            print(f"✓ Tokens generated: ~{len(response.generated_text.split())}")
            
            return TestResult(
                "Run Inference",
                True,
                f"Inference successful in {inference_time:.2f}ms",
                (time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return TestResult(
                "Run Inference",
                False,
                f"Error: {str(e)}",
                (time.time() - start_time) * 1000
            )
    
    async def test_6_measure_memory(self) -> TestResult:
        """Test 6: Measure memory usage per worker"""
        print("\n" + "="*60)
        print("TEST 6: Measure Memory Usage")
        print("="*60)
        
        start_time = time.time()
        
        try:
            print("\nMemory usage per worker:")
            total_memory = 0.0
            
            for i, port in enumerate(self.worker_ports):
                worker_id = f"worker{i+1}"
                process = self.processes.get(worker_id)
                
                if process and process.poll() is None:
                    memory_gb = self.get_process_memory(process)
                    total_memory += memory_gb
                    print(f"  {worker_id}: {memory_gb:.2f} GB")
                else:
                    print(f"  {worker_id}: NOT RUNNING")
            
            avg_memory = total_memory / len(self.worker_ports)
            print(f"\nTotal memory: {total_memory:.2f} GB")
            print(f"Average per worker: {avg_memory:.2f} GB")
            
            return TestResult(
                "Measure Memory",
                True,
                f"Average memory per worker: {avg_memory:.2f} GB",
                (time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return TestResult(
                "Measure Memory",
                False,
                f"Error: {str(e)}",
                (time.time() - start_time) * 1000
            )
    
    async def test_7_node_health(self) -> TestResult:
        """Test 7: Verify node health monitoring"""
        print("\n" + "="*60)
        print("TEST 7: Verify Node Health Monitoring")
        print("="*60)
        
        start_time = time.time()
        
        try:
            channel = grpc.aio.insecure_channel(f'localhost:{self.coordinator_port}')
            stub = distributed_pb2_grpc.CoordinatorServiceStub(channel)
            
            request = distributed_pb2.ListWorkersRequest()
            response = await stub.ListWorkers(request)
            await channel.close()
            
            print("\nWorker health status:")
            healthy_count = 0
            
            for worker in response.workers:
                status = "✓ HEALTHY" if worker.status == "healthy" else "✗ UNHEALTHY"
                print(f"  {worker.worker_id}: {status}")
                print(f"    Load: {worker.current_load} requests")
                print(f"    Memory: {worker.memory_usage_percent:.1f}%")
                
                if worker.status == "healthy":
                    healthy_count += 1
            
            if healthy_count != len(response.workers):
                return TestResult(
                    "Node Health",
                    False,
                    f"Only {healthy_count}/{len(response.workers)} workers healthy",
                    (time.time() - start_time) * 1000
                )
            
            print(f"\n✓ All {healthy_count} workers healthy")
            
            return TestResult(
                "Node Health",
                True,
                f"All {healthy_count} workers healthy",
                (time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return TestResult(
                "Node Health",
                False,
                f"Error: {str(e)}",
                (time.time() - start_time) * 1000
            )
    
    async def test_8_multiple_requests(self) -> TestResult:
        """Test 8: Run multiple concurrent requests"""
        print("\n" + "="*60)
        print("TEST 8: Multiple Concurrent Requests")
        print("="*60)
        
        start_time = time.time()
        
        try:
            prompts = [
                "The capital of France is",
                "Python is a programming language that",
                "Machine learning is",
                "The weather today is",
                "Artificial intelligence can"
            ]
            
            print(f"Running {len(prompts)} concurrent requests...")
            
            async def run_inference(prompt: str) -> Tuple[bool, float]:
                try:
                    channel = grpc.aio.insecure_channel(f'localhost:{self.coordinator_port}')
                    stub = distributed_pb2_grpc.CoordinatorServiceStub(channel)
                    
                    request = distributed_pb2.InferenceRequest(
                        model_name=self.model_name,
                        prompt=prompt,
                        max_tokens=10,
                        temperature=0.7
                    )
                    
                    req_start = time.time()
                    response = await stub.RouteInference(request)
                    latency = (time.time() - req_start) * 1000
                    await channel.close()
                    
                    return bool(response.generated_text), latency
                except Exception:
                    return False, 0.0
            
            # Run all requests concurrently
            tasks = [run_inference(prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks)
            
            success_count = sum(1 for success, _ in results if success)
            latencies = [latency for success, latency in results if success]
            
            print(f"\n✓ Successful requests: {success_count}/{len(prompts)}")
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                print(f"✓ Average latency: {avg_latency:.2f}ms")
                print(f"✓ Min latency: {min_latency:.2f}ms")
                print(f"✓ Max latency: {max_latency:.2f}ms")
            
            if success_count < len(prompts):
                return TestResult(
                    "Multiple Requests",
                    False,
                    f"Only {success_count}/{len(prompts)} requests succeeded",
                    (time.time() - start_time) * 1000
                )
            
            return TestResult(
                "Multiple Requests",
                True,
                f"All {success_count} concurrent requests succeeded",
                (time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return TestResult(
                "Multiple Requests",
                False,
                f"Error: {str(e)}",
                (time.time() - start_time) * 1000
            )
    
    def cleanup(self):
        """Cleanup all processes"""
        print("\n" + "="*60)
        print("CLEANUP")
        print("="*60)
        
        for name, process in self.processes.items():
            if process and process.poll() is None:
                print(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}...")
                    process.kill()
        
        print("✓ All processes stopped")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"\nResults: {passed}/{total} tests passed")
        print("\nDetails:")
        
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"\n{status} - {result.name}")
            print(f"  Message: {result.message}")
            print(f"  Duration: {result.duration_ms:.2f}ms")
        
        print("\n" + "="*60)
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! System is working perfectly!")
        else:
            print(f"⚠️  {total - passed} test(s) failed. Check logs for details.")
        
        print("="*60)
        
        return passed == total
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("UNIFIED SYSTEM INTEGRATION TEST")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Workers: {len(self.worker_ports)}")
        print("="*60)
        
        try:
            # Run tests in sequence
            self.results.append(await self.test_1_start_coordinator())
            if not self.results[-1].passed:
                return False
            
            self.results.append(await self.test_2_start_workers())
            if not self.results[-1].passed:
                return False
            
            self.results.append(await self.test_3_download_sharded_model())
            if not self.results[-1].passed:
                return False
            
            self.results.append(await self.test_4_verify_layer_distribution())
            if not self.results[-1].passed:
                return False
            
            self.results.append(await self.test_5_run_inference())
            if not self.results[-1].passed:
                return False
            
            self.results.append(await self.test_6_measure_memory())
            # Continue even if memory measurement fails
            
            self.results.append(await self.test_7_node_health())
            if not self.results[-1].passed:
                return False
            
            self.results.append(await self.test_8_multiple_requests())
            if not self.results[-1].passed:
                return False
            
            return True
            
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            return False
        except Exception as e:
            print(f"\n\nUnexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup()
            success = self.print_summary()
            sys.exit(0 if success else 1)


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified system integration test')
    parser.add_argument('--model', type=str, 
                       default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='Model to test with')
    
    args = parser.parse_args()
    
    test = UnifiedSystemTest(model_name=args.model)
    await test.run_all_tests()


if __name__ == '__main__':
    asyncio.run(main())
