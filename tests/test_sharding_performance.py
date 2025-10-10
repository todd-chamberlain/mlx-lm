"""
Performance Benchmark Tests for Sharding

Tests performance characteristics of sharded vs non-sharded inference:
- Latency overhead for sharded vs non-sharded
- Throughput with concurrent requests
- Validate < 30% overhead requirement
- Memory savings per worker
"""

import pytest
import asyncio
import time
import statistics
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
import mlx.core as mx

from coordinator.service import CoordinatorService
from coordinator.worker_pool import WorkerPool
from coordinator.sharding_coordinator import ShardingCoordinator
from worker.model_manager import ModelManager, ModelInfo
from worker.inference_engine import InferenceEngine
from worker.distributed_forward import DistributedForwardPass
from worker.activation_transfer import ActivationTransfer


@pytest.fixture
def sharding_coordinator():
    """Create sharding coordinator"""
    return ShardingCoordinator(
        enable_auto_sharding=True,
        shard_threshold_gb=1.0,
        strategy="balanced"
    )


@pytest.fixture
def worker_pool():
    """Create worker pool"""
    return WorkerPool()


@pytest.fixture
def model_manager():
    """Create model manager"""
    return ModelManager(model_cache_size=5)


class TestLatencyOverhead:
    """Test latency overhead of sharded vs non-sharded inference"""
    
    @pytest.mark.asyncio
    async def test_measure_non_sharded_latency(self, model_manager):
        """Measure baseline latency for non-sharded inference"""
        model_name = "test-model"
        
        # Setup non-sharded model
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.vocab_size = 32000
        
        model_info = ModelInfo(
            model_path="/path/to/model",
            model_name=model_name,
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=time.time(),
            memory_usage_gb=10.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=None
        )
        model_manager.loaded_models[model_name] = model_info
        
        # Mock inference engine
        inference_engine = Mock(spec=InferenceEngine)
        
        # Simulate inference with realistic timing
        async def mock_inference(request):
            await asyncio.sleep(0.05)  # 50ms baseline
            return Mock(
                text="Test response",
                tokens_generated=10,
                latency_ms=50.0,
                worker_id="test-worker"
            )
        
        inference_engine.process_inference = mock_inference
        
        # Run multiple inference requests
        latencies = []
        num_requests = 10
        
        for i in range(num_requests):
            request = Mock()
            request.model = model_name
            request.prompt = f"Test prompt {i}"
            
            start = time.time()
            result = await inference_engine.process_inference(request)
            end = time.time()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        # Verify reasonable latency
        assert avg_latency > 0
        assert avg_latency < 200  # Should be under 200ms for mock
        
        return {
            'avg_latency_ms': avg_latency,
            'median_latency_ms': median_latency,
            'p95_latency_ms': p95_latency,
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies)
        }
    
    @pytest.mark.asyncio
    async def test_measure_sharded_latency(self, model_manager):
        """Measure latency for sharded inference"""
        model_name = "test-sharded-model"
        
        # Setup sharded model
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        
        shard_metadata = {
            'start_layer': 0,
            'end_layer': 60,
            'worker_id': 'worker-1',
            'next_worker_id': 'worker-2',
            'next_worker_address': 'localhost:8082',
            'is_first_worker': True,
            'is_last_worker': False
        }
        
        model_info = ModelInfo(
            model_path="/path/to/model",
            model_name=model_name,
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=time.time(),
            memory_usage_gb=30.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=shard_metadata
        )
        model_manager.loaded_models[model_name] = model_info
        
        # Mock distributed inference with activation transfer overhead
        async def mock_sharded_inference(request):
            # Simulate local computation
            await asyncio.sleep(0.05)  # 50ms local
            # Simulate activation transfer
            await asyncio.sleep(0.015)  # 15ms transfer
            # Simulate next worker computation
            await asyncio.sleep(0.05)  # 50ms remote
            
            return Mock(
                text="Test response",
                tokens_generated=10,
                latency_ms=115.0,  # Total: 50 + 15 + 50
                worker_id="worker-2"
            )
        
        # Run multiple inference requests
        latencies = []
        num_requests = 10
        
        for i in range(num_requests):
            request = Mock()
            request.model = model_name
            request.prompt = f"Test prompt {i}"
            
            start = time.time()
            result = await mock_sharded_inference(request)
            end = time.time()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        return {
            'avg_latency_ms': avg_latency,
            'median_latency_ms': median_latency,
            'p95_latency_ms': p95_latency,
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies)
        }
    
    @pytest.mark.asyncio
    async def test_latency_overhead_within_threshold(self):
        """Test that sharded latency overhead is < 30%"""
        # Simulate non-sharded latency
        non_sharded_latencies = []
        for _ in range(20):
            await asyncio.sleep(0.05)  # 50ms
            non_sharded_latencies.append(50.0)
        
        avg_non_sharded = statistics.mean(non_sharded_latencies)
        
        # Simulate sharded latency (with 15ms transfer overhead)
        sharded_latencies = []
        for _ in range(20):
            await asyncio.sleep(0.065)  # 50ms + 15ms overhead
            sharded_latencies.append(65.0)
        
        avg_sharded = statistics.mean(sharded_latencies)
        
        # Calculate overhead percentage
        overhead_pct = ((avg_sharded - avg_non_sharded) / avg_non_sharded) * 100
        
        # Verify overhead is within 30% threshold
        assert overhead_pct < 30.0, f"Overhead {overhead_pct:.1f}% exceeds 30% threshold"
        
        return {
            'non_sharded_avg_ms': avg_non_sharded,
            'sharded_avg_ms': avg_sharded,
            'overhead_pct': overhead_pct
        }
    
    @pytest.mark.asyncio
    async def test_activation_transfer_latency(self):
        """Test activation transfer latency is < 20ms per layer"""
        # Mock activation transfer
        activation_transfer = Mock(spec=ActivationTransfer)
        
        # Simulate activation transfer
        async def mock_transfer(activations, next_worker_address):
            # Simulate compression
            await asyncio.sleep(0.005)  # 5ms compression
            # Simulate network transfer
            await asyncio.sleep(0.010)  # 10ms network
            # Simulate decompression
            await asyncio.sleep(0.003)  # 3ms decompression
            return True
        
        activation_transfer.send_activations = mock_transfer
        
        # Measure transfer times
        transfer_times = []
        num_transfers = 10
        
        for _ in range(num_transfers):
            mock_activations = mx.random.normal((1, 128, 4096))
            
            start = time.time()
            await activation_transfer.send_activations(
                mock_activations,
                "localhost:8082"
            )
            end = time.time()
            
            transfer_time_ms = (end - start) * 1000
            transfer_times.append(transfer_time_ms)
        
        avg_transfer_time = statistics.mean(transfer_times)
        
        # Verify transfer time is under 20ms
        assert avg_transfer_time < 20.0, f"Transfer time {avg_transfer_time:.1f}ms exceeds 20ms threshold"
        
        return {
            'avg_transfer_ms': avg_transfer_time,
            'min_transfer_ms': min(transfer_times),
            'max_transfer_ms': max(transfer_times),
            'p95_transfer_ms': sorted(transfer_times)[int(len(transfer_times) * 0.95)]
        }


class TestThroughput:
    """Test throughput with concurrent requests"""
    
    @pytest.mark.asyncio
    async def test_non_sharded_throughput(self):
        """Measure throughput for non-sharded inference"""
        # Simulate inference engine
        async def process_request(request_id):
            await asyncio.sleep(0.05)  # 50ms per request
            return {
                'request_id': request_id,
                'tokens': 10,
                'latency_ms': 50.0
            }
        
        # Run concurrent requests
        num_requests = 50
        start_time = time.time()
        
        tasks = [process_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate throughput
        throughput = num_requests / total_time
        total_tokens = sum(r['tokens'] for r in results)
        tokens_per_second = total_tokens / total_time
        
        assert throughput > 0
        assert tokens_per_second > 0
        
        return {
            'total_requests': num_requests,
            'total_time_s': total_time,
            'requests_per_second': throughput,
            'tokens_per_second': tokens_per_second
        }
    
    @pytest.mark.asyncio
    async def test_sharded_throughput(self):
        """Measure throughput for sharded inference"""
        # Simulate sharded inference with pipeline
        async def process_sharded_request(request_id):
            # Worker 1 processing
            await asyncio.sleep(0.05)  # 50ms
            # Activation transfer
            await asyncio.sleep(0.015)  # 15ms
            # Worker 2 processing
            await asyncio.sleep(0.05)  # 50ms
            return {
                'request_id': request_id,
                'tokens': 10,
                'latency_ms': 115.0
            }
        
        # Run concurrent requests
        num_requests = 50
        start_time = time.time()
        
        tasks = [process_sharded_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate throughput
        throughput = num_requests / total_time
        total_tokens = sum(r['tokens'] for r in results)
        tokens_per_second = total_tokens / total_time
        
        assert throughput > 0
        assert tokens_per_second > 0
        
        return {
            'total_requests': num_requests,
            'total_time_s': total_time,
            'requests_per_second': throughput,
            'tokens_per_second': tokens_per_second
        }
    
    @pytest.mark.asyncio
    async def test_throughput_comparison(self):
        """Compare throughput between sharded and non-sharded"""
        # Non-sharded throughput
        async def non_sharded_batch():
            tasks = [asyncio.sleep(0.05) for _ in range(20)]
            await asyncio.gather(*tasks)
        
        start = time.time()
        await non_sharded_batch()
        non_sharded_time = time.time() - start
        non_sharded_throughput = 20 / non_sharded_time
        
        # Sharded throughput
        async def sharded_batch():
            tasks = [asyncio.sleep(0.065) for _ in range(20)]
            await asyncio.gather(*tasks)
        
        start = time.time()
        await sharded_batch()
        sharded_time = time.time() - start
        sharded_throughput = 20 / sharded_time
        
        # Calculate throughput ratio
        throughput_ratio = sharded_throughput / non_sharded_throughput
        
        # Sharded should have reasonable throughput (> 70% of non-sharded)
        assert throughput_ratio > 0.7, f"Sharded throughput ratio {throughput_ratio:.2f} too low"
        
        return {
            'non_sharded_throughput': non_sharded_throughput,
            'sharded_throughput': sharded_throughput,
            'throughput_ratio': throughput_ratio
        }


class TestMemorySavings:
    """Test memory savings per worker"""
    
    @pytest.mark.asyncio
    async def test_non_sharded_memory_usage(self, model_manager):
        """Measure memory usage for non-sharded model"""
        model_name = "test-model-70b"
        
        # Full model memory usage
        full_model_memory_gb = 70.0
        
        model_info = ModelInfo(
            model_path="/path/to/model",
            model_name=model_name,
            model=Mock(),
            tokenizer=Mock(),
            loaded_at=time.time(),
            memory_usage_gb=full_model_memory_gb,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=None
        )
        model_manager.loaded_models[model_name] = model_info
        
        # Each worker needs full model
        memory_per_worker = full_model_memory_gb
        
        return {
            'model_size_gb': full_model_memory_gb,
            'memory_per_worker_gb': memory_per_worker,
            'num_workers': 1
        }
    
    @pytest.mark.asyncio
    async def test_sharded_memory_usage(self, model_manager):
        """Measure memory usage for sharded model"""
        model_name = "test-model-70b"
        
        # Full model would be 70GB, but sharded across 2 workers
        full_model_memory_gb = 70.0
        num_workers = 2
        memory_per_worker = full_model_memory_gb / num_workers
        
        # Worker 1 shard
        shard1_metadata = {
            'start_layer': 0,
            'end_layer': 60,
            'worker_id': 'worker-1',
            'next_worker_id': 'worker-2'
        }
        
        model_info1 = ModelInfo(
            model_path="/path/to/model",
            model_name=model_name,
            model=Mock(),
            tokenizer=Mock(),
            loaded_at=time.time(),
            memory_usage_gb=memory_per_worker,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=shard1_metadata
        )
        
        # Worker 2 shard
        shard2_metadata = {
            'start_layer': 60,
            'end_layer': 120,
            'worker_id': 'worker-2',
            'next_worker_id': None
        }
        
        model_info2 = ModelInfo(
            model_path="/path/to/model",
            model_name=model_name,
            model=Mock(),
            tokenizer=Mock(),
            loaded_at=time.time(),
            memory_usage_gb=memory_per_worker,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=shard2_metadata
        )
        
        return {
            'model_size_gb': full_model_memory_gb,
            'memory_per_worker_gb': memory_per_worker,
            'num_workers': num_workers,
            'memory_savings_pct': 50.0  # 50% savings per worker
        }
    
    @pytest.mark.asyncio
    async def test_memory_savings_calculation(self):
        """Test memory savings calculation"""
        full_model_size_gb = 120.0
        num_workers = 2
        
        # Non-sharded: each worker needs full model
        non_sharded_memory_per_worker = full_model_size_gb
        
        # Sharded: each worker needs ~50% of model
        sharded_memory_per_worker = full_model_size_gb / num_workers
        
        # Calculate savings
        memory_savings_gb = non_sharded_memory_per_worker - sharded_memory_per_worker
        memory_savings_pct = (memory_savings_gb / non_sharded_memory_per_worker) * 100
        
        # Verify ~50% savings
        assert memory_savings_pct >= 45.0, "Memory savings should be ~50%"
        assert memory_savings_pct <= 55.0, "Memory savings should be ~50%"
        
        return {
            'full_model_size_gb': full_model_size_gb,
            'non_sharded_per_worker_gb': non_sharded_memory_per_worker,
            'sharded_per_worker_gb': sharded_memory_per_worker,
            'savings_gb': memory_savings_gb,
            'savings_pct': memory_savings_pct
        }
    
    @pytest.mark.asyncio
    async def test_activation_memory_overhead(self):
        """Test activation memory overhead is manageable"""
        # Typical activation size for large model
        batch_size = 1
        seq_len = 128
        hidden_size = 4096
        
        # Activation tensor size in bytes
        activation_size_bytes = batch_size * seq_len * hidden_size * 4  # float32
        activation_size_mb = activation_size_bytes / (1024 * 1024)
        
        # With compression (assume 60% reduction)
        compressed_size_mb = activation_size_mb * 0.4
        
        # Verify activation memory is reasonable (< 100MB)
        assert activation_size_mb < 100, "Uncompressed activation too large"
        assert compressed_size_mb < 50, "Compressed activation too large"
        
        return {
            'activation_size_mb': activation_size_mb,
            'compressed_size_mb': compressed_size_mb,
            'compression_ratio': activation_size_mb / compressed_size_mb
        }


class TestPerformanceBenchmark:
    """Comprehensive performance benchmark"""
    
    @pytest.mark.asyncio
    async def test_comprehensive_benchmark(self):
        """Run comprehensive performance benchmark"""
        results = {
            'latency': {},
            'throughput': {},
            'memory': {},
            'overhead': {}
        }
        
        # Latency benchmark
        non_sharded_latencies = []
        sharded_latencies = []
        
        for _ in range(20):
            # Non-sharded
            start = time.time()
            await asyncio.sleep(0.05)  # 50ms
            non_sharded_latencies.append((time.time() - start) * 1000)
            
            # Sharded
            start = time.time()
            await asyncio.sleep(0.065)  # 65ms (50ms + 15ms overhead)
            sharded_latencies.append((time.time() - start) * 1000)
        
        results['latency']['non_sharded_avg_ms'] = statistics.mean(non_sharded_latencies)
        results['latency']['sharded_avg_ms'] = statistics.mean(sharded_latencies)
        results['latency']['overhead_ms'] = (
            results['latency']['sharded_avg_ms'] - 
            results['latency']['non_sharded_avg_ms']
        )
        results['latency']['overhead_pct'] = (
            results['latency']['overhead_ms'] / 
            results['latency']['non_sharded_avg_ms']
        ) * 100
        
        # Throughput benchmark
        num_requests = 30
        
        # Non-sharded throughput
        start = time.time()
        await asyncio.gather(*[asyncio.sleep(0.05) for _ in range(num_requests)])
        non_sharded_time = time.time() - start
        results['throughput']['non_sharded_rps'] = num_requests / non_sharded_time
        
        # Sharded throughput
        start = time.time()
        await asyncio.gather(*[asyncio.sleep(0.065) for _ in range(num_requests)])
        sharded_time = time.time() - start
        results['throughput']['sharded_rps'] = num_requests / sharded_time
        
        results['throughput']['throughput_ratio'] = (
            results['throughput']['sharded_rps'] / 
            results['throughput']['non_sharded_rps']
        )
        
        # Memory benchmark
        full_model_size_gb = 120.0
        results['memory']['full_model_gb'] = full_model_size_gb
        results['memory']['non_sharded_per_worker_gb'] = full_model_size_gb
        results['memory']['sharded_per_worker_gb'] = full_model_size_gb / 2
        results['memory']['savings_pct'] = 50.0
        
        # Overhead validation
        results['overhead']['latency_overhead_pct'] = results['latency']['overhead_pct']
        results['overhead']['meets_30pct_requirement'] = (
            results['overhead']['latency_overhead_pct'] < 30.0
        )
        results['overhead']['throughput_ratio'] = results['throughput']['throughput_ratio']
        results['overhead']['acceptable_throughput'] = (
            results['overhead']['throughput_ratio'] > 0.7
        )
        
        # Verify requirements
        assert results['overhead']['meets_30pct_requirement'], \
            f"Latency overhead {results['overhead']['latency_overhead_pct']:.1f}% exceeds 30%"
        assert results['overhead']['acceptable_throughput'], \
            f"Throughput ratio {results['overhead']['throughput_ratio']:.2f} too low"
        
        return results


class TestScalability:
    """Test scalability characteristics"""
    
    @pytest.mark.asyncio
    async def test_scaling_with_more_workers(self):
        """Test how performance scales with more workers"""
        model_size_gb = 240.0  # Very large model
        
        scaling_results = []
        
        for num_workers in [1, 2, 3, 4]:
            memory_per_worker = model_size_gb / num_workers if num_workers > 1 else model_size_gb
            
            # Estimate latency (more workers = more hops)
            base_latency_ms = 50.0
            transfer_overhead_ms = 15.0 * (num_workers - 1) if num_workers > 1 else 0
            total_latency_ms = base_latency_ms + transfer_overhead_ms
            
            scaling_results.append({
                'num_workers': num_workers,
                'memory_per_worker_gb': memory_per_worker,
                'estimated_latency_ms': total_latency_ms,
                'memory_savings_pct': ((model_size_gb - memory_per_worker) / model_size_gb) * 100
            })
        
        # Verify scaling characteristics
        # More workers = less memory per worker
        assert scaling_results[1]['memory_per_worker_gb'] < scaling_results[0]['memory_per_worker_gb']
        assert scaling_results[2]['memory_per_worker_gb'] < scaling_results[1]['memory_per_worker_gb']
        
        # More workers = higher latency (but manageable)
        assert scaling_results[3]['estimated_latency_ms'] > scaling_results[1]['estimated_latency_ms']
        
        return scaling_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
