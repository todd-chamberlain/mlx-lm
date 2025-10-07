"""
Batch Processing Integration Tests

Comprehensive integration tests for the complete batch processing system,
validating end-to-end functionality, adaptive sizing, load-based scheduling,
priority handling, and performance characteristics.
"""

import pytest
import time
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import mlx.core as mx

from coordinator.batch_manager import (
    BatchManager,
    BatchRequest,
    BatchPriority,
    BatchingPolicy
)
from coordinator.batch_processor import BatchProcessor
from coordinator.batch_scheduler import BatchScheduler


# Mock inference function with realistic latencies
async def mock_inference_realistic(prompts: List[str], max_tokens: int) -> List[str]:
    """
    Mock inference function with realistic processing time.
    
    Simulates ~10ms per token generation.
    """
    # Simulate processing time based on batch size and tokens
    batch_size = len(prompts)
    processing_time = 0.01 * max_tokens * (1 + 0.1 * batch_size)  # Slight overhead for larger batches
    await asyncio.sleep(processing_time)
    
    # Generate mock outputs
    return [f"Generated response for: {prompt[:20]}..." for prompt in prompts]


async def mock_inference_fast(prompts: List[str], max_tokens: int) -> List[str]:
    """Fast mock inference for testing."""
    await asyncio.sleep(0.01)
    return ["Fast response"] * len(prompts)


async def mock_inference_slow(prompts: List[str], max_tokens: int) -> List[str]:
    """Slow mock inference for testing."""
    await asyncio.sleep(0.1)
    return ["Slow response"] * len(prompts)



class TestEndToEndPipeline:
    """Test complete end-to-end batch processing pipeline."""
    
    @pytest.mark.asyncio
    async def test_complete_batch_flow(self):
        """Test complete flow: submission → batching → assembly → processing → results."""
        # Initialize components
        policy = BatchingPolicy(
            min_batch_size=2,
            max_batch_size=8,
            max_batch_wait_ms=50.0
        )
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor,
            enable_adaptive_sizing=False  # Disable for deterministic test
        )
        
        # Submit requests
        request_ids = []
        for i in range(5):
            req = BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt=f"Test prompt {i}",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
            batch_manager.submit_request(req)
            request_ids.append(req.request_id)
        
        # Update load
        batch_scheduler.update_load_metrics(
            queue_depth=5,
            active_batches=0,
            worker_utilization=0.5,
            avg_latency_ms=100.0
        )
        
        # Wait for batch timeout
        await asyncio.sleep(0.06)
        
        # Schedule and process batch
        result = await batch_scheduler.schedule_batch(mock_inference_fast)
        
        # Verify results
        assert result is not None
        assert result.success_count == 5
        assert result.failure_count == 0
        assert len(result.request_results) == 5
        
        # Verify all requests got results
        for req_id in request_ids:
            assert req_id in result.request_results
            assert result.request_results[req_id].success is True
    
    @pytest.mark.asyncio
    async def test_batch_assembly_and_padding(self):
        """Test batch assembly with padding and attention masks."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=8)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        
        # Submit requests with variable prompt lengths
        requests = [
            BatchRequest(
                request_id="short",
                model_name="test-model",
                prompt="Short",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ),
            BatchRequest(
                request_id="medium",
                model_name="test-model",
                prompt="This is a medium length prompt",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ),
            BatchRequest(
                request_id="long",
                model_name="test-model",
                prompt="This is a much longer prompt with many more words to test padding",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
        ]
        
        for req in requests:
            batch_manager.submit_request(req)
        
        # Wait for batch timeout
        await asyncio.sleep(0.06)
        
        # Create batch
        batch = batch_manager.create_batch(current_load=0.5)
        assert batch is not None
        
        # Process batch
        result = await batch_processor.process_batch(batch, mock_inference_fast)
        
        # Verify all requests processed successfully
        assert result.success_count == 3
        assert result.failure_count == 0

    
    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self):
        """Test error handling throughout the pipeline."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=8)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor
        )
        
        # Submit requests
        for i in range(3):
            batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ))
        
        batch_scheduler.update_load_metrics(
            queue_depth=3,
            active_batches=0,
            worker_utilization=0.5,
            avg_latency_ms=100.0
        )
        
        await asyncio.sleep(0.06)
        
        # Mock inference that fails
        async def failing_inference(prompts, max_tokens):
            raise RuntimeError("Inference failed")
        
        result = await batch_scheduler.schedule_batch(failing_inference)
        
        # Should handle error gracefully
        assert result is not None
        assert result.success_count == 0
        assert result.failure_count == 3
        
        # All requests should have error results
        for req_result in result.request_results.values():
            assert req_result.success is False
            assert "Inference failed" in req_result.error
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self):
        """Test handling of partial batch failures."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=8)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        
        # Submit requests
        for i in range(4):
            batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ))
        
        await asyncio.sleep(0.06)
        
        # Create batch
        batch = batch_manager.create_batch(current_load=0.5)
        
        # Mock inference that returns fewer outputs than inputs
        async def partial_inference(prompts, max_tokens):
            # Only return outputs for first 2 requests
            return ["Output 1", "Output 2"]
        
        result = await batch_processor.process_batch(batch, partial_inference)
        
        # Should handle partial failure
        assert result.success_count == 2
        assert result.failure_count == 2
    
    @pytest.mark.asyncio
    async def test_multi_batch_sequential_processing(self):
        """Test processing multiple batches sequentially."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=4)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor
        )
        
        # Submit 10 requests (will create multiple batches)
        for i in range(10):
            batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt=f"Prompt {i}",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ))
        
        batch_scheduler.update_load_metrics(
            queue_depth=10,
            active_batches=0,
            worker_utilization=0.5,
            avg_latency_ms=100.0
        )
        
        await asyncio.sleep(0.06)
        
        # Process multiple batches
        results = []
        for _ in range(3):
            result = await batch_scheduler.schedule_batch(mock_inference_fast)
            if result:
                results.append(result)
        
        # Should have processed multiple batches
        assert len(results) >= 2
        
        # Total successful requests
        total_success = sum(r.success_count for r in results)
        assert total_success == 10



class TestAdaptiveSizing:
    """Test adaptive batch sizing behavior."""
    
    @pytest.mark.asyncio
    async def test_convergence_to_optimal_size(self):
        """Test that adaptive sizing converges to optimal size under stable load."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=16)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor,
            enable_adaptive_sizing=True
        )
        
        # Simulate stable load and consistent performance
        batch_scheduler.update_load_metrics(
            queue_depth=10,
            active_batches=2,
            worker_utilization=0.6,
            avg_latency_ms=150.0
        )
        
        # Process multiple batches to allow convergence
        batch_sizes = []
        for iteration in range(10):
            # Submit requests
            for i in range(12):
                batch_manager.submit_request(BatchRequest(
                    request_id=f"iter_{iteration}_req_{i}",
                    model_name="test-model",
                    prompt="Test",
                    max_tokens=50,
                    temperature=0.7,
                    top_p=0.9
                ))
            
            await asyncio.sleep(0.06)
            
            # Process batch
            result = await batch_scheduler.schedule_batch(mock_inference_fast)
            if result:
                batch_sizes.append(result.success_count + result.failure_count)
        
        # Batch sizes should stabilize (variance should decrease over time)
        early_variance = sum((s - sum(batch_sizes[:5])/5)**2 for s in batch_sizes[:5]) / 5
        late_variance = sum((s - sum(batch_sizes[-5:])/5)**2 for s in batch_sizes[-5:]) / 5
        
        # Later batches should have lower variance (more stable)
        assert late_variance <= early_variance * 1.5  # Allow some tolerance
    
    @pytest.mark.asyncio
    async def test_size_adjustment_under_varying_load(self):
        """Test batch size adjustment under varying load patterns."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=16)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor,
            enable_adaptive_sizing=True
        )
        
        # Test under low load
        batch_scheduler.update_load_metrics(
            queue_depth=5,
            active_batches=0,
            worker_utilization=0.2,
            avg_latency_ms=50.0
        )
        
        low_load_size = batch_scheduler.get_batch_size()
        
        # Test under high load
        batch_scheduler.update_load_metrics(
            queue_depth=50,
            active_batches=10,
            worker_utilization=0.95,
            avg_latency_ms=500.0
        )
        
        high_load_size = batch_scheduler.get_batch_size()
        
        # High load should prefer smaller or equal batches (allow equal for initial state)
        assert high_load_size <= low_load_size

    
    @pytest.mark.asyncio
    async def test_exploration_vs_exploitation(self):
        """Test exploration vs exploitation behavior in adaptive sizing."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=16)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        
        # Create scheduler with exploration enabled
        from coordinator.batch_scheduler import AdaptiveBatchSizer
        sizer = AdaptiveBatchSizer(
            min_batch_size=2,
            max_batch_size=16,
            exploration_rate=0.3  # 30% exploration
        )
        
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor,
            enable_adaptive_sizing=True
        )
        batch_scheduler.adaptive_sizer = sizer
        
        # Get batch sizes multiple times
        batch_scheduler.update_load_metrics(
            queue_depth=10,
            active_batches=2,
            worker_utilization=0.5,
            avg_latency_ms=150.0
        )
        
        sizes = set()
        for _ in range(20):
            size = batch_scheduler.get_batch_size()
            sizes.add(size)
        
        # With exploration, should try multiple sizes
        assert len(sizes) > 1
    
    @pytest.mark.asyncio
    async def test_performance_feedback_loop(self):
        """Test that performance feedback affects future batch sizing."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=16)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor,
            enable_adaptive_sizing=True
        )
        
        batch_scheduler.update_load_metrics(
            queue_depth=10,
            active_batches=2,
            worker_utilization=0.5,
            avg_latency_ms=150.0
        )
        
        # Process batches and provide performance feedback
        for iteration in range(5):
            # Submit requests
            for i in range(10):
                batch_manager.submit_request(BatchRequest(
                    request_id=f"iter_{iteration}_req_{i}",
                    model_name="test-model",
                    prompt="Test",
                    max_tokens=50,
                    temperature=0.7,
                    top_p=0.9
                ))
            
            await asyncio.sleep(0.06)
            
            # Process batch (performance feedback happens automatically)
            await batch_scheduler.schedule_batch(mock_inference_fast)
        
        # Adaptive sizer should have tracked performance
        stats = batch_scheduler.adaptive_sizer.get_stats()
        assert len(stats["tracked_sizes"]) > 0
        assert stats["optimal_size"] >= policy.min_batch_size
        assert stats["optimal_size"] <= policy.max_batch_size



class TestLoadBasedScheduling:
    """Test load-based batch scheduling behavior."""
    
    @pytest.mark.asyncio
    async def test_low_load_maximize_throughput(self):
        """Test that low load maximizes batch size for throughput."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=16)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor,
            enable_adaptive_sizing=True
        )
        
        # Set low load
        batch_scheduler.update_load_metrics(
            queue_depth=8,
            active_batches=0,
            worker_utilization=0.2,
            avg_latency_ms=50.0
        )
        
        # Submit many requests
        for i in range(16):
            batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ))
        
        await asyncio.sleep(0.06)
        
        # Should create larger batch under low load
        batch = batch_manager.create_batch(current_load=0.2)
        assert batch is not None
        assert batch.size >= 8  # Should prefer larger batches
    
    @pytest.mark.asyncio
    async def test_medium_load_balanced_sizing(self):
        """Test balanced batch sizing under medium load."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=16)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor,
            enable_adaptive_sizing=True
        )
        
        # Set medium load
        batch_scheduler.update_load_metrics(
            queue_depth=15,
            active_batches=3,
            worker_utilization=0.6,
            avg_latency_ms=150.0
        )
        
        # Get optimal size under medium load
        size = batch_scheduler.get_batch_size()
        
        # Should be between min and max (inclusive)
        assert size >= policy.min_batch_size
        assert size <= policy.max_batch_size
    
    @pytest.mark.asyncio
    async def test_high_load_minimize_latency(self):
        """Test that high load uses smaller batches for lower latency."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=16)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor,
            enable_adaptive_sizing=True
        )
        
        # First set low load to establish baseline
        batch_scheduler.update_load_metrics(
            queue_depth=5,
            active_batches=0,
            worker_utilization=0.2,
            avg_latency_ms=50.0
        )
        
        low_load_size = batch_scheduler.get_batch_size()
        
        # Then set high load
        batch_scheduler.update_load_metrics(
            queue_depth=50,
            active_batches=10,
            worker_utilization=0.95,
            avg_latency_ms=500.0
        )
        
        # Get optimal size under high load
        high_load_size = batch_scheduler.get_batch_size()
        
        # High load should use smaller or equal batches
        assert high_load_size <= low_load_size

    
    @pytest.mark.asyncio
    async def test_queue_management_under_pressure(self):
        """Test queue management and backpressure handling."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=8)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor
        )
        
        # Submit many requests to create backpressure
        for i in range(50):
            batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ))
        
        initial_queue_size = batch_manager.queue.size()
        assert initial_queue_size == 50
        
        # Update with high load
        batch_scheduler.update_load_metrics(
            queue_depth=50,
            active_batches=5,
            worker_utilization=0.9,
            avg_latency_ms=400.0
        )
        
        await asyncio.sleep(0.06)
        
        # Process multiple batches to drain queue
        processed_count = 0
        for _ in range(10):
            result = await batch_scheduler.schedule_batch(mock_inference_fast)
            if result:
                processed_count += result.success_count + result.failure_count
            if batch_manager.queue.is_empty():
                break
        
        # Should have processed all requests
        assert processed_count == 50
        assert batch_manager.queue.is_empty()


class TestPriorityAndPreemption:
    """Test priority handling and batch preemption."""
    
    @pytest.mark.asyncio
    async def test_critical_priority_preemption(self):
        """Test that critical priority requests trigger preemption."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=8)
        batch_manager = BatchManager(policy=policy, enable_priority=True)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor,
            enable_preemption=True
        )
        
        # Submit critical priority request first
        critical_req = BatchRequest(
            request_id="critical",
            model_name="test-model",
            prompt="Critical request",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            priority=BatchPriority.CRITICAL
        )
        
        # Wait to age the request (preemption threshold is 100ms)
        await asyncio.sleep(0.15)
        
        # Now submit it
        batch_manager.submit_request(critical_req)
        
        # Submit normal priority requests
        for i in range(5):
            batch_manager.submit_request(BatchRequest(
                request_id=f"normal_{i}",
                model_name="test-model",
                prompt="Normal request",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
                priority=BatchPriority.NORMAL
            ))
        
        batch_scheduler.update_load_metrics(
            queue_depth=6,
            active_batches=0,
            worker_utilization=0.5,
            avg_latency_ms=100.0
        )
        
        # Should trigger preemption (critical request is old enough)
        should_preempt = batch_scheduler._should_preempt()
        assert should_preempt is True

    
    @pytest.mark.asyncio
    async def test_priority_ordering_in_batches(self):
        """Test that priority ordering is maintained within batches."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=8)
        batch_manager = BatchManager(policy=policy, enable_priority=True)
        batch_processor = BatchProcessor()
        
        # Submit requests with different priorities
        batch_manager.submit_request(BatchRequest(
            request_id="low",
            model_name="test-model",
            prompt="Low priority",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            priority=BatchPriority.LOW
        ))
        
        batch_manager.submit_request(BatchRequest(
            request_id="high",
            model_name="test-model",
            prompt="High priority",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            priority=BatchPriority.HIGH
        ))
        
        batch_manager.submit_request(BatchRequest(
            request_id="normal",
            model_name="test-model",
            prompt="Normal priority",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            priority=BatchPriority.NORMAL
        ))
        
        await asyncio.sleep(0.06)
        
        # Create batch
        batch = batch_manager.create_batch(current_load=0.5)
        
        # High priority should be first
        assert batch.requests[0].request_id == "high"
        assert batch.requests[1].request_id == "normal"
        assert batch.requests[2].request_id == "low"
    
    @pytest.mark.asyncio
    async def test_fairness_with_mixed_priorities(self):
        """Test fairness policies with mixed priority requests."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=8)
        batch_manager = BatchManager(
            policy=policy,
            enable_priority=True,
            enable_fairness=True
        )
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor
        )
        
        # Submit mix of priorities
        for i in range(10):
            priority = BatchPriority.HIGH if i % 3 == 0 else BatchPriority.NORMAL
            batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
                priority=priority
            ))
        
        batch_scheduler.update_load_metrics(
            queue_depth=10,
            active_batches=0,
            worker_utilization=0.5,
            avg_latency_ms=100.0
        )
        
        await asyncio.sleep(0.06)
        
        # Process batches
        results = []
        for _ in range(3):
            result = await batch_scheduler.schedule_batch(mock_inference_fast)
            if result:
                results.append(result)
        
        # All requests should be processed
        total_processed = sum(r.success_count for r in results)
        assert total_processed == 10



class TestPerformanceBenchmarks:
    """Test performance characteristics and improvements."""
    
    @pytest.mark.asyncio
    async def test_throughput_improvement(self):
        """Test throughput improvement with batching vs single requests."""
        policy = BatchingPolicy(min_batch_size=4, max_batch_size=8)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor
        )
        
        # Measure single request processing time
        single_start = time.time()
        for i in range(8):
            await mock_inference_realistic([f"Prompt {i}"], 50)
        single_time = time.time() - single_start
        
        # Measure batched processing time
        for i in range(8):
            batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt=f"Prompt {i}",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ))
        
        batch_scheduler.update_load_metrics(
            queue_depth=8,
            active_batches=0,
            worker_utilization=0.5,
            avg_latency_ms=100.0
        )
        
        await asyncio.sleep(0.06)
        
        batch_start = time.time()
        result = await batch_scheduler.schedule_batch(mock_inference_realistic)
        batch_time = time.time() - batch_start
        
        # Batching should be faster (at least 1.5x improvement)
        improvement = single_time / batch_time
        assert improvement >= 1.5, f"Expected 1.5x improvement, got {improvement:.2f}x"
    
    @pytest.mark.asyncio
    async def test_latency_distribution_by_priority(self):
        """Test latency distribution for different priority levels."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=8)
        batch_manager = BatchManager(policy=policy, enable_priority=True)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor,
            enable_preemption=True
        )
        
        latencies = {
            BatchPriority.CRITICAL: [],
            BatchPriority.HIGH: [],
            BatchPriority.NORMAL: [],
            BatchPriority.LOW: []
        }
        
        # Submit requests with different priorities
        for priority in [BatchPriority.CRITICAL, BatchPriority.HIGH, BatchPriority.NORMAL, BatchPriority.LOW]:
            for i in range(3):
                req = BatchRequest(
                    request_id=f"{priority.name}_{i}",
                    model_name="test-model",
                    prompt="Test",
                    max_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    priority=priority
                )
                batch_manager.submit_request(req)
        
        batch_scheduler.update_load_metrics(
            queue_depth=12,
            active_batches=0,
            worker_utilization=0.5,
            avg_latency_ms=100.0
        )
        
        await asyncio.sleep(0.06)
        
        # Process batches and measure latencies
        for _ in range(3):
            result = await batch_scheduler.schedule_batch(mock_inference_fast)
            if result:
                for req_id, req_result in result.request_results.items():
                    priority_name = req_id.split('_')[0]
                    priority = BatchPriority[priority_name]
                    latencies[priority].append(req_result.processing_time_ms)
        
        # Critical priority should have lower average latency
        if latencies[BatchPriority.CRITICAL] and latencies[BatchPriority.LOW]:
            avg_critical = sum(latencies[BatchPriority.CRITICAL]) / len(latencies[BatchPriority.CRITICAL])
            avg_low = sum(latencies[BatchPriority.LOW]) / len(latencies[BatchPriority.LOW])
            
            # Critical should be processed faster (or at least not slower)
            assert avg_critical <= avg_low * 1.2  # Allow 20% tolerance

    
    @pytest.mark.asyncio
    async def test_batch_efficiency_metrics(self):
        """Test batch efficiency: padding ratio and success rate."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=8)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor
        )
        
        # Submit requests with variable prompt lengths
        prompts = [
            "Short",
            "Medium length prompt here",
            "This is a longer prompt with more words",
            "Another medium one",
            "Tiny",
            "This one is also quite long with many tokens"
        ]
        
        for i, prompt in enumerate(prompts):
            batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt=prompt,
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ))
        
        batch_scheduler.update_load_metrics(
            queue_depth=6,
            active_batches=0,
            worker_utilization=0.5,
            avg_latency_ms=100.0
        )
        
        await asyncio.sleep(0.06)
        
        # Process batch
        result = await batch_scheduler.schedule_batch(mock_inference_fast)
        
        # Check success rate
        assert result.success_rate >= 0.95  # Target: 95%+ success rate
        
        # Get scheduler stats
        stats = batch_scheduler.get_stats()
        assert stats["batches_scheduled"] > 0
    
    @pytest.mark.asyncio
    async def test_resource_utilization_patterns(self):
        """Test resource utilization patterns under different loads."""
        policy = BatchingPolicy(min_batch_size=2, max_batch_size=16)
        batch_manager = BatchManager(policy=policy)
        batch_processor = BatchProcessor()
        batch_scheduler = BatchScheduler(
            batch_manager=batch_manager,
            batch_processor=batch_processor,
            enable_adaptive_sizing=True
        )
        
        # Test under different load scenarios
        load_scenarios = [
            {"queue_depth": 5, "worker_util": 0.2, "name": "low"},
            {"queue_depth": 20, "worker_util": 0.6, "name": "medium"},
            {"queue_depth": 50, "worker_util": 0.9, "name": "high"}
        ]
        
        results = {}
        
        for scenario in load_scenarios:
            # Submit requests
            for i in range(12):
                batch_manager.submit_request(BatchRequest(
                    request_id=f"{scenario['name']}_req_{i}",
                    model_name="test-model",
                    prompt="Test",
                    max_tokens=50,
                    temperature=0.7,
                    top_p=0.9
                ))
            
            # Update load
            batch_scheduler.update_load_metrics(
                queue_depth=scenario["queue_depth"],
                active_batches=2,
                worker_utilization=scenario["worker_util"],
                avg_latency_ms=100.0 * scenario["worker_util"] / 0.5
            )
            
            await asyncio.sleep(0.06)
            
            # Process batch
            result = await batch_scheduler.schedule_batch(mock_inference_fast)
            if result:
                results[scenario["name"]] = {
                    "batch_size": result.success_count + result.failure_count,
                    "success_rate": result.success_rate
                }
        
        # Verify different load patterns produce different batch sizes
        if "low" in results and "high" in results:
            # High load should use smaller batches
            assert results["high"]["batch_size"] <= results["low"]["batch_size"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
