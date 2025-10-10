#!/usr/bin/env python3
"""
Memory Savings Measurement Script

Measures actual memory usage with sharded vs non-sharded model loading.
This script helps validate the memory savings from sharding.

Usage:
    python scripts/measure_memory_savings.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import asyncio
import subprocess
import time
import psutil
import grpc
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from proto import distributed_pb2, distributed_pb2_grpc


class MemoryMeasurement:
    """Measures memory usage of worker processes"""
    
    def __init__(self):
        self.measurements = {}
    
    def measure_process(self, process: subprocess.Popen, label: str) -> float:
        """Measure memory usage of a process in GB"""
        try:
            proc = psutil.Process(process.pid)
            memory_info = proc.memory_info()
            memory_gb = memory_info.rss / (1024 ** 3)
            self.measurements[label] = memory_gb
            return memory_gb
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all measurements"""
        return self.measurements.copy()


async def wait_for_service(port: int, timeout: int = 30) -> bool:
    """Wait for a gRPC service to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            channel = grpc.aio.insecure_channel(f'localhost:{port}')
            # Try to connect
            await asyncio.wait_for(channel.channel_ready(), timeout=2.0)
            await channel.close()
            return True
        except Exception:
            await asyncio.sleep(1)
    return False


async def download_model_sharded(
    coordinator_port: int,
    model_name: str,
    num_workers: int
) -> bool:
    """Download model with sharding enabled"""
    try:
        channel = grpc.aio.insecure_channel(f'localhost:{coordinator_port}')
        stub = distributed_pb2_grpc.CoordinatorServiceStub(channel)
        
        request = distributed_pb2.DownloadModelRequest(
            model_name=model_name,
            enable_sharding=True,
            min_workers=num_workers
        )
        
        response = await stub.DownloadModel(request)
        await channel.close()
        
        return response.success
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


async def download_model_non_sharded(
    coordinator_port: int,
    model_name: str
) -> bool:
    """Download model without sharding"""
    try:
        channel = grpc.aio.insecure_channel(f'localhost:{coordinator_port}')
        stub = distributed_pb2_grpc.CoordinatorServiceStub(channel)
        
        request = distributed_pb2.DownloadModelRequest(
            model_name=model_name,
            enable_sharding=False
        )
        
        response = await stub.DownloadModel(request)
        await channel.close()
        
        return response.success
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


async def measure_sharded_memory(model_name: str) -> Dict[str, float]:
    """Measure memory usage with sharded model"""
    print("\n=== Measuring Sharded Model Memory Usage ===\n")
    
    base_dir = Path(__file__).parent.parent
    processes = {}
    measurer = MemoryMeasurement()
    
    try:
        # Start coordinator
        print("Starting coordinator...")
        coordinator_cmd = [
            "python", "-m", "coordinator.main",
            "--port", "50051",
            "--enable-auto-sharding", "true",
            "--shard-threshold-gb", "0.5",
            "--min-workers-for-sharding", "2"
        ]
        processes['coordinator'] = subprocess.Popen(
            coordinator_cmd,
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if not await wait_for_service(50051):
            raise Exception("Coordinator failed to start")
        print("✓ Coordinator ready")
        
        # Start worker 1
        print("Starting worker 1...")
        worker1_cmd = [
            "python", "-m", "worker.main",
            "--worker-id", "worker1",
            "--coordinator", "localhost:50051",
            "--port", "50052"
        ]
        processes['worker1'] = subprocess.Popen(
            worker1_cmd,
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if not await wait_for_service(50052):
            raise Exception("Worker 1 failed to start")
        print("✓ Worker 1 ready")
        
        # Start worker 2
        print("Starting worker 2...")
        worker2_cmd = [
            "python", "-m", "worker.main",
            "--worker-id", "worker2",
            "--coordinator", "localhost:50051",
            "--port", "50053"
        ]
        processes['worker2'] = subprocess.Popen(
            worker2_cmd,
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if not await wait_for_service(50053):
            raise Exception("Worker 2 failed to start")
        print("✓ Worker 2 ready")
        
        # Wait for workers to register
        await asyncio.sleep(3)
        
        # Measure baseline memory
        baseline_worker1 = measurer.measure_process(processes['worker1'], 'sharded_worker1_baseline')
        baseline_worker2 = measurer.measure_process(processes['worker2'], 'sharded_worker2_baseline')
        print(f"Baseline memory - Worker 1: {baseline_worker1:.2f} GB, Worker 2: {baseline_worker2:.2f} GB")
        
        # Download model with sharding
        print(f"\nDownloading model '{model_name}' with sharding...")
        success = await download_model_sharded(50051, model_name, 2)
        
        if not success:
            raise Exception("Model download failed")
        
        # Wait for model to load
        print("Waiting for model to load...")
        await asyncio.sleep(10)
        
        # Measure memory after loading
        loaded_worker1 = measurer.measure_process(processes['worker1'], 'sharded_worker1_loaded')
        loaded_worker2 = measurer.measure_process(processes['worker2'], 'sharded_worker2_loaded')
        print(f"After loading - Worker 1: {loaded_worker1:.2f} GB, Worker 2: {loaded_worker2:.2f} GB")
        
        # Calculate memory used by model
        model_memory_worker1 = loaded_worker1 - baseline_worker1
        model_memory_worker2 = loaded_worker2 - baseline_worker2
        total_model_memory = model_memory_worker1 + model_memory_worker2
        
        measurer.measurements['sharded_model_memory_worker1'] = model_memory_worker1
        measurer.measurements['sharded_model_memory_worker2'] = model_memory_worker2
        measurer.measurements['sharded_total_model_memory'] = total_model_memory
        
        print(f"\nModel memory - Worker 1: {model_memory_worker1:.2f} GB, Worker 2: {model_memory_worker2:.2f} GB")
        print(f"Total model memory: {total_model_memory:.2f} GB")
        
        return measurer.get_summary()
        
    finally:
        # Cleanup
        print("\nCleaning up sharded test...")
        for name, process in processes.items():
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


async def measure_non_sharded_memory(model_name: str) -> Dict[str, float]:
    """Measure memory usage with non-sharded model"""
    print("\n=== Measuring Non-Sharded Model Memory Usage ===\n")
    
    base_dir = Path(__file__).parent.parent
    processes = {}
    measurer = MemoryMeasurement()
    
    try:
        # Start coordinator
        print("Starting coordinator...")
        coordinator_cmd = [
            "python", "-m", "coordinator.main",
            "--port", "50051",
            "--enable-auto-sharding", "false"
        ]
        processes['coordinator'] = subprocess.Popen(
            coordinator_cmd,
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if not await wait_for_service(50051):
            raise Exception("Coordinator failed to start")
        print("✓ Coordinator ready")
        
        # Start worker 1
        print("Starting worker 1...")
        worker1_cmd = [
            "python", "-m", "worker.main",
            "--worker-id", "worker1",
            "--coordinator", "localhost:50051",
            "--port", "50052"
        ]
        processes['worker1'] = subprocess.Popen(
            worker1_cmd,
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if not await wait_for_service(50052):
            raise Exception("Worker 1 failed to start")
        print("✓ Worker 1 ready")
        
        # Wait for worker to register
        await asyncio.sleep(3)
        
        # Measure baseline memory
        baseline_worker1 = measurer.measure_process(processes['worker1'], 'non_sharded_worker1_baseline')
        print(f"Baseline memory - Worker 1: {baseline_worker1:.2f} GB")
        
        # Download model without sharding
        print(f"\nDownloading model '{model_name}' without sharding...")
        success = await download_model_non_sharded(50051, model_name)
        
        if not success:
            raise Exception("Model download failed")
        
        # Wait for model to load
        print("Waiting for model to load...")
        await asyncio.sleep(10)
        
        # Measure memory after loading
        loaded_worker1 = measurer.measure_process(processes['worker1'], 'non_sharded_worker1_loaded')
        print(f"After loading - Worker 1: {loaded_worker1:.2f} GB")
        
        # Calculate memory used by model
        model_memory_worker1 = loaded_worker1 - baseline_worker1
        
        measurer.measurements['non_sharded_model_memory_worker1'] = model_memory_worker1
        
        print(f"\nModel memory - Worker 1: {model_memory_worker1:.2f} GB")
        
        return measurer.get_summary()
        
    finally:
        # Cleanup
        print("\nCleaning up non-sharded test...")
        for name, process in processes.items():
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


def print_comparison(sharded: Dict[str, float], non_sharded: Dict[str, float]):
    """Print comparison of memory usage"""
    print("\n" + "="*60)
    print("MEMORY USAGE COMPARISON")
    print("="*60)
    
    print("\nSharded Configuration (2 workers):")
    print(f"  Worker 1: {sharded.get('sharded_model_memory_worker1', 0):.2f} GB")
    print(f"  Worker 2: {sharded.get('sharded_model_memory_worker2', 0):.2f} GB")
    print(f"  Total:    {sharded.get('sharded_total_model_memory', 0):.2f} GB")
    
    print("\nNon-Sharded Configuration (1 worker):")
    print(f"  Worker 1: {non_sharded.get('non_sharded_model_memory_worker1', 0):.2f} GB")
    
    # Calculate savings
    sharded_per_worker_avg = (
        sharded.get('sharded_model_memory_worker1', 0) +
        sharded.get('sharded_model_memory_worker2', 0)
    ) / 2
    non_sharded_memory = non_sharded.get('non_sharded_model_memory_worker1', 0)
    
    if non_sharded_memory > 0:
        savings_percent = ((non_sharded_memory - sharded_per_worker_avg) / non_sharded_memory) * 100
        print(f"\nMemory Savings per Worker:")
        print(f"  Average per worker (sharded): {sharded_per_worker_avg:.2f} GB")
        print(f"  Full model (non-sharded):     {non_sharded_memory:.2f} GB")
        print(f"  Savings:                       {savings_percent:.1f}%")
    
    print("\n" + "="*60)


async def main():
    parser = argparse.ArgumentParser(description='Measure memory savings from sharding')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='Model to test (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)')
    parser.add_argument('--skip-sharded', action='store_true',
                       help='Skip sharded measurement')
    parser.add_argument('--skip-non-sharded', action='store_true',
                       help='Skip non-sharded measurement')
    
    args = parser.parse_args()
    
    sharded_results = {}
    non_sharded_results = {}
    
    try:
        if not args.skip_sharded:
            sharded_results = await measure_sharded_memory(args.model)
            await asyncio.sleep(5)  # Wait between tests
        
        if not args.skip_non_sharded:
            non_sharded_results = await measure_non_sharded_memory(args.model)
        
        if sharded_results and non_sharded_results:
            print_comparison(sharded_results, non_sharded_results)
        
    except KeyboardInterrupt:
        print("\n\nMeasurement interrupted by user")
    except Exception as e:
        print(f"\n\nError during measurement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
