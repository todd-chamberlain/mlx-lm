#!/usr/bin/env python3
"""
Performance Measurement Script

Measures inference latency and throughput with sharded vs non-sharded models.
This script helps quantify the performance overhead of sharding.

Usage:
    python scripts/measure_performance.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import asyncio
import subprocess
import time
import statistics
import grpc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from proto import distributed_pb2, distributed_pb2_grpc


class PerformanceMetrics:
    """Collects and analyzes performance metrics"""
    
    def __init__(self):
        self.latencies = []
        self.token_counts = []
    
    def add_measurement(self, latency_ms: float, token_count: int):
        """Add a measurement"""
        self.latencies.append(latency_ms)
        self.token_counts.append(token_count)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        if not self.latencies:
            return {}
        
        return {
            'mean_latency_ms': statistics.mean(self.latencies),
            'median_latency_ms': statistics.median(self.latencies),
            'min_latency_ms': min(self.latencies),
            'max_latency_ms': max(self.latencies),
            'stdev_latency_ms': statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0,
            'total_tokens': sum(self.token_counts),
            'mean_tokens': statistics.mean(self.token_counts),
            'num_requests': len(self.latencies)
        }


async def wait_for_service(port: int, timeout: int = 30) -> bool:
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


async def download_model(
    coordinator_port: int,
    model_name: str,
    enable_sharding: bool,
    num_workers: int = 2
) -> bool:
    """Download model"""
    try:
        channel = grpc.aio.insecure_channel(f'localhost:{coordinator_port}')
        stub = distributed_pb2_grpc.CoordinatorServiceStub(channel)
        
        request = distributed_pb2.DownloadModelRequest(
            model_name=model_name,
            enable_sharding=enable_sharding,
            min_workers=num_workers if enable_sharding else 1
        )
        
        response = await stub.DownloadModel(request)
        await channel.close()
        
        return response.success
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


async def run_inference(
    coordinator_port: int,
    model_name: str,
    prompt: str,
    max_tokens: int = 50
) -> Tuple[float, int]:
    """Run inference and measure latency"""
    try:
        channel = grpc.aio.insecure_channel(f'localhost:{coordinator_port}')
        stub = distributed_pb2_grpc.CoordinatorServiceStub(channel)
        
        request = distributed_pb2.InferenceRequest(
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        start_time = time.time()
        response = await stub.RouteInference(request)
        latency_ms = (time.time() - start_time) * 1000
        
        await channel.close()
        
        # Count tokens (approximate)
        token_count = len(response.generated_text.split())
        
        return latency_ms, token_count
    except Exception as e:
        print(f"Error during inference: {e}")
        return 0.0, 0


async def measure_performance(
    model_name: str,
    enable_sharding: bool,
    num_workers: int,
    num_requests: int = 10,
    prompts: Optional[List[str]] = None
) -> PerformanceMetrics:
    """Measure performance with given configuration"""
    
    config_name = "Sharded" if enable_sharding else "Non-Sharded"
    print(f"\n=== Measuring {config_name} Performance ===\n")
    
    base_dir = Path(__file__).parent.parent
    processes = {}
    metrics = PerformanceMetrics()
    
    try:
        # Start coordinator
        print("Starting coordinator...")
        coordinator_cmd = [
            "python", "-m", "coordinator.main",
            "--port", "50051",
            "--enable-auto-sharding", str(enable_sharding).lower(),
        ]
        if enable_sharding:
            coordinator_cmd.extend([
                "--shard-threshold-gb", "0.5",
                "--min-workers-for-sharding", str(num_workers)
            ])
        
        processes['coordinator'] = subprocess.Popen(
            coordinator_cmd,
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if not await wait_for_service(50051):
            raise Exception("Coordinator failed to start")
        print("✓ Coordinator ready")
        
        # Start workers
        for i in range(num_workers):
            worker_id = f"worker{i+1}"
            port = 50052 + i
            print(f"Starting {worker_id}...")
            
            worker_cmd = [
                "python", "-m", "worker.main",
                "--worker-id", worker_id,
                "--coordinator", "localhost:50051",
                "--port", str(port)
            ]
            processes[worker_id] = subprocess.Popen(
                worker_cmd,
                cwd=base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if not await wait_for_service(port):
                raise Exception(f"{worker_id} failed to start")
            print(f"✓ {worker_id} ready")
        
        # Wait for workers to register
        await asyncio.sleep(3)
        
        # Download model
        print(f"\nDownloading model '{model_name}'...")
        success = await download_model(50051, model_name, enable_sharding, num_workers)
        
        if not success:
            raise Exception("Model download failed")
        
        # Wait for model to load
        print("Waiting for model to load...")
        await asyncio.sleep(10)
        
        # Default prompts if none provided
        if not prompts:
            prompts = [
                "The capital of France is",
                "Explain quantum computing in simple terms:",
                "Write a haiku about programming:",
                "What is the meaning of life?",
                "Describe the color blue to someone who can't see:",
            ]
        
        # Run inference requests
        print(f"\nRunning {num_requests} inference requests...")
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            print(f"  Request {i+1}/{num_requests}: '{prompt[:50]}...'")
            
            latency_ms, token_count = await run_inference(50051, model_name, prompt)
            
            if latency_ms > 0:
                metrics.add_measurement(latency_ms, token_count)
                print(f"    Latency: {latency_ms:.2f}ms, Tokens: {token_count}")
            else:
                print(f"    Failed")
            
            # Small delay between requests
            await asyncio.sleep(0.5)
        
        return metrics
        
    finally:
        # Cleanup
        print(f"\nCleaning up {config_name.lower()} test...")
        for name, process in processes.items():
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


def print_comparison(sharded: Dict[str, float], non_sharded: Dict[str, float]):
    """Print comparison of performance metrics"""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    print("\nSharded Configuration:")
    print(f"  Mean Latency:   {sharded.get('mean_latency_ms', 0):.2f} ms")
    print(f"  Median Latency: {sharded.get('median_latency_ms', 0):.2f} ms")
    print(f"  Min Latency:    {sharded.get('min_latency_ms', 0):.2f} ms")
    print(f"  Max Latency:    {sharded.get('max_latency_ms', 0):.2f} ms")
    print(f"  Std Dev:        {sharded.get('stdev_latency_ms', 0):.2f} ms")
    print(f"  Total Tokens:   {sharded.get('total_tokens', 0):.0f}")
    print(f"  Requests:       {sharded.get('num_requests', 0):.0f}")
    
    print("\nNon-Sharded Configuration:")
    print(f"  Mean Latency:   {non_sharded.get('mean_latency_ms', 0):.2f} ms")
    print(f"  Median Latency: {non_sharded.get('median_latency_ms', 0):.2f} ms")
    print(f"  Min Latency:    {non_sharded.get('min_latency_ms', 0):.2f} ms")
    print(f"  Max Latency:    {non_sharded.get('max_latency_ms', 0):.2f} ms")
    print(f"  Std Dev:        {non_sharded.get('stdev_latency_ms', 0):.2f} ms")
    print(f"  Total Tokens:   {non_sharded.get('total_tokens', 0):.0f}")
    print(f"  Requests:       {non_sharded.get('num_requests', 0):.0f}")
    
    # Calculate overhead
    sharded_mean = sharded.get('mean_latency_ms', 0)
    non_sharded_mean = non_sharded.get('mean_latency_ms', 0)
    
    if non_sharded_mean > 0:
        overhead_ms = sharded_mean - non_sharded_mean
        overhead_percent = (overhead_ms / non_sharded_mean) * 100
        
        print(f"\nPerformance Overhead:")
        print(f"  Absolute:  {overhead_ms:+.2f} ms")
        print(f"  Relative:  {overhead_percent:+.1f}%")
        
        if overhead_percent < 10:
            print(f"  Assessment: ✓ Low overhead (< 10%)")
        elif overhead_percent < 25:
            print(f"  Assessment: ⚠ Moderate overhead (10-25%)")
        else:
            print(f"  Assessment: ✗ High overhead (> 25%)")
    
    # Calculate throughput
    if sharded.get('num_requests', 0) > 0:
        sharded_tokens_per_sec = sharded.get('total_tokens', 0) / (sharded.get('mean_latency_ms', 1) / 1000)
        non_sharded_tokens_per_sec = non_sharded.get('total_tokens', 0) / (non_sharded.get('mean_latency_ms', 1) / 1000)
        
        print(f"\nThroughput (tokens/sec):")
        print(f"  Sharded:     {sharded_tokens_per_sec:.2f}")
        print(f"  Non-Sharded: {non_sharded_tokens_per_sec:.2f}")
    
    print("\n" + "="*60)


async def main():
    parser = argparse.ArgumentParser(description='Measure performance overhead from sharding')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='Model to test (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)')
    parser.add_argument('--num-requests', type=int, default=10,
                       help='Number of inference requests to run (default: 10)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of workers for sharded test (default: 2)')
    parser.add_argument('--skip-sharded', action='store_true',
                       help='Skip sharded measurement')
    parser.add_argument('--skip-non-sharded', action='store_true',
                       help='Skip non-sharded measurement')
    
    args = parser.parse_args()
    
    sharded_metrics = None
    non_sharded_metrics = None
    
    try:
        if not args.skip_sharded:
            sharded_metrics = await measure_performance(
                args.model,
                enable_sharding=True,
                num_workers=args.num_workers,
                num_requests=args.num_requests
            )
            await asyncio.sleep(5)  # Wait between tests
        
        if not args.skip_non_sharded:
            non_sharded_metrics = await measure_performance(
                args.model,
                enable_sharding=False,
                num_workers=1,
                num_requests=args.num_requests
            )
        
        if sharded_metrics and non_sharded_metrics:
            print_comparison(
                sharded_metrics.get_summary(),
                non_sharded_metrics.get_summary()
            )
        
    except KeyboardInterrupt:
        print("\n\nMeasurement interrupted by user")
    except Exception as e:
        print(f"\n\nError during measurement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
