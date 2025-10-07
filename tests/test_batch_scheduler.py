"""
Tests for Batch Scheduling Optimization

Tests adaptive batch sizing, load-based scheduling, batch preemption,
and performance monitoring.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from coordinator.batch_scheduler import (
    BatchScheduler,
    AdaptiveBatchSizer,
    BatchPerformanceMetrics,
    LoadMetrics
)
from coordinator.batch_manager import BatchManager, Batch, BatchRequest, BatchPriority, BatchingPolicy
from coordinator.batch_processor import BatchProcessor, BatchResult
from coordinator.models import InferenceResult


class TestBatchPerformanceMetrics:
    """Test batch performance metrics."""
    
    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = BatchPerformanceMetrics(
            batch_size=16,
            processing_time_ms=100.0,
            throughput_tokens_per_sec=500.0,
            padding_ratio=0.2,
            success_rate=0.95
        )
        
        assert metrics.batch_size == 16
        assert metrics.processing_time_ms == 100.0
        assert metrics.throughput_tokens_per_sec == 500.0
        assert metrics.padding_ratio == 0.2
        assert metrics.success_rate == 0.95
    
    def test_efficiency_score(self):
        """Test efficiency score calculation."""
        # High efficiency: good throughput, low padding, high success
        metrics1 = BatchPerformanceMetrics(
            batch_size=16,
            processing_time_ms=100.0,
            throughput_tokens_per_sec=1000.0,
            padding_ratio=0.1,
            success_rate=1.0
        )
        
        # Low efficiency: low throughput, high padding, low success
        metrics2 = BatchPerformanceMetrics(
            batch_size=16,
            processing_time_ms=200.0,
            throughput_tokens_per_sec=200.0,
            padding_ratio=0.8,
            success_rate=0.5
        )
        
        assert metrics1.efficiency_score > metrics2.efficiency_score


class TestLoadMetrics:
    """Test load metrics."""
    
    def test_load_metrics_creation(self):
        """Test creating load metrics."""
        metrics = LoadMetrics(
            queue_depth=50,
            active_batches=5,
            worker_utilization=0.7,
            avg_latency_ms=150.0
        )
        
        assert metrics.queue_depth == 50
        assert metrics.active_batches == 5
        assert metrics.worker_utilization == 0.7
        assert metrics.avg_latency_ms == 150.0
    
    def test_load_level_calculation(self):
        """Test load level calculation."""
        # Low load
        low_load = LoadMetrics(
            queue_depth=5,
            active_batches=1,
            worker_utilization=0.2,
            avg_latency_ms=50.0
        )
        
        # High load
        high_load = LoadMetrics(
            queue_depth=150,
            active_batches=15,
            worker_utilization=0.95,
            avg_latency_ms=500.0
        )
        
        assert low_load.load_level < 0.3
        assert high_load.load_level > 0.8


class TestAdaptiveBatchSizer:
    """Test adaptive batch sizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sizer = AdaptiveBatchSizer(
            min_batch_size=2,
            max_batch_size=32,
            learning_rate=0.2,
            exploration_rate=0.0  # Disable exploration for deterministic tests
        )
    
    def test_sizer_initialization(self):
        """Test sizer initialization."""
        assert self.sizer.min_batch_size == 2
        assert self.sizer.max_batch_size == 32
        assert self.sizer.learning_rate == 0.2
        assert self.sizer._optimal_size == 32
    
    def test_update_performance(self):
        """Test updating performance metrics."""
        metrics = BatchPerformanceMetrics(
            batch_size=16,
            processing_time_ms=100.0,
            throughput_tokens_per_sec=500.0,
            padding_ratio=0.2,
            success_rate=0.95
        )
        
        self.sizer.update_performance(metrics)
        
        assert 16 in self.sizer._avg_efficiency
        assert self.sizer._avg_efficiency[16] > 0
    
    def test_optimal_size_selection(self):
        """Test optimal size selection based on performance."""
        # Add performance data for different sizes
        # Size 16 performs best
        for _ in range(5):
            self.sizer.update_performance(BatchPerformanceMetrics(
                batch_size=8,
                processing_time_ms=100.0,
                throughput_tokens_per_sec=300.0,
                padding_ratio=0.1,
                success_rate=0.9
            ))
            
            self.sizer.update_performance(BatchPerformanceMetrics(
                batch_size=16,
                processing_time_ms=100.0,
                throughput_tokens_per_sec=600.0,
                padding_ratio=0.2,
                success_rate=0.95
            ))
            
            self.sizer.update_performance(BatchPerformanceMetrics(
                batch_size=24,
                processing_time_ms=150.0,
                throughput_tokens_per_sec=400.0,
                padding_ratio=0.4,
                success_rate=0.85
            ))
        
        # Optimal size should be 16
        assert self.sizer._optimal_size == 16
    
    def test_get_optimal_size_low_load(self):
        """Test optimal size under low load."""
        # Set optimal size
        self.sizer._optimal_size = 16
        
        # Low load should return optimal size
        size = self.sizer.get_optimal_size(current_load=0.2)
        
        assert size == 16
    
    def test_get_optimal_size_high_load(self):
        """Test optimal size under high load."""
        # Set optimal size
        self.sizer._optimal_size = 16
        
        # High load should return smaller size
        size = self.sizer.get_optimal_size(current_load=0.9)
        
        assert size < 16
        assert size >= self.sizer.min_batch_size
    
    def test_exploration(self):
        """Test exploration behavior."""
        sizer_with_exploration = AdaptiveBatchSizer(
            min_batch_size=2,
            max_batch_size=32,
            exploration_rate=1.0  # Always explore
        )
        
        # Should return random sizes
        sizes = set()
        for _ in range(20):
            size = sizer_with_exploration.get_optimal_size(current_load=0.5)
            sizes.add(size)
        
        # Should have explored multiple sizes
        assert len(sizes) > 1
    
    def test_get_stats(self):
        """Test getting sizer statistics."""
        # Add some performance data
        self.sizer.update_performance(BatchPerformanceMetrics(
            batch_size=16,
            processing_time_ms=100.0,
            throughput_tokens_per_sec=500.0,
            padding_ratio=0.2,
            success_rate=0.95
        ))
        
        stats = self.sizer.get_stats()
        
        assert "optimal_size" in stats
        assert "tracked_sizes" in stats
        assert "avg_efficiency_by_size" in stats
        assert 16 in stats["tracked_sizes"]


class TestBatchScheduler:
    """Test batch scheduler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_manager = BatchManager(
            policy=BatchingPolicy(
                min_batch_size=2,
                max_batch_size=16,
                max_batch_wait_ms=50.0
            )
        )
        self.batch_processor = BatchProcessor()
        self.scheduler = BatchScheduler(
            batch_manager=self.batch_manager,
            batch_processor=self.batch_processor,
            enable_adaptive_sizing=True,
            enable_preemption=True
        )
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        assert self.scheduler.batch_manager == self.batch_manager
        assert self.scheduler.batch_processor == self.batch_processor
        assert self.scheduler.enable_adaptive_sizing is True
        assert self.scheduler.enable_preemption is True
        assert self.scheduler.adaptive_sizer is not None
    
    def test_update_load_metrics(self):
        """Test updating load metrics."""
        self.scheduler.update_load_metrics(
            queue_depth=50,
            active_batches=5,
            worker_utilization=0.7,
            avg_latency_ms=150.0
        )
        
        assert self.scheduler._current_load.queue_depth == 50
        assert self.scheduler._current_load.active_batches == 5
        assert self.scheduler._current_load.worker_utilization == 0.7
        assert self.scheduler._current_load.avg_latency_ms == 150.0
    
    def test_should_create_batch_empty_queue(self):
        """Test batch creation decision with empty queue."""
        should_create = self.scheduler.should_create_batch()
        
        assert should_create is False
    
    def test_should_create_batch_with_requests(self):
        """Test batch creation decision with requests."""
        # Add requests
        for i in range(5):
            self.batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ))
        
        # Update load
        self.scheduler.update_load_metrics(
            queue_depth=5,
            active_batches=0,
            worker_utilization=0.3,
            avg_latency_ms=100.0
        )
        
        # Wait for batch timeout
        time.sleep(0.06)
        
        should_create = self.scheduler.should_create_batch()
        
        assert should_create is True
    
    def test_should_create_batch_high_load(self):
        """Test batch creation under high load."""
        # Add requests (at least min batch size)
        for i in range(5):
            self.batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ))
        
        # Very high load - ensure load_level > 0.8
        # queue_depth=100 (normalized to 1.0), active_batches=10 (normalized to 1.0), worker_util=1.0
        # load = 1.0 * 0.4 + 1.0 * 0.3 + 1.0 * 0.3 = 1.0
        actual_queue_size = self.batch_manager.queue.size()
        self.scheduler.update_load_metrics(
            queue_depth=100,  # High queue depth
            active_batches=10,  # High active batches
            worker_utilization=1.0,  # Full utilization
            avg_latency_ms=500.0
        )
        
        should_create = self.scheduler.should_create_batch()
        
        # Should create batch aggressively under high load
        assert should_create is True
    
    def test_get_batch_size(self):
        """Test getting optimal batch size."""
        # Update load
        self.scheduler.update_load_metrics(
            queue_depth=10,
            active_batches=2,
            worker_utilization=0.5,
            avg_latency_ms=150.0
        )
        
        size = self.scheduler.get_batch_size()
        
        assert size >= self.batch_manager.policy.min_batch_size
        assert size <= self.batch_manager.policy.max_batch_size
    
    @pytest.mark.asyncio
    async def test_schedule_batch_success(self):
        """Test successful batch scheduling."""
        # Add requests
        for i in range(5):
            self.batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test prompt",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ))
        
        # Update load
        self.scheduler.update_load_metrics(
            queue_depth=5,
            active_batches=0,
            worker_utilization=0.3,
            avg_latency_ms=100.0
        )
        
        # Wait for batch timeout
        time.sleep(0.06)
        
        # Mock inference function
        async def mock_inference(prompts, max_tokens):
            return [f"Output {i}" for i in range(len(prompts))]
        
        # Schedule batch
        result = await self.scheduler.schedule_batch(mock_inference)
        
        assert result is not None
        assert result.success_count > 0
        assert self.scheduler.stats["batches_scheduled"] == 1
    
    @pytest.mark.asyncio
    async def test_schedule_batch_no_batch(self):
        """Test scheduling when no batch should be created."""
        # No requests in queue
        
        async def mock_inference(prompts, max_tokens):
            return []
        
        result = await self.scheduler.schedule_batch(mock_inference)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_schedule_batch_with_size_limit(self):
        """Test batch scheduling with size limiting."""
        # Add many requests
        for i in range(20):
            self.batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ))
        
        # Update load
        self.scheduler.update_load_metrics(
            queue_depth=20,
            active_batches=0,
            worker_utilization=0.3,
            avg_latency_ms=100.0
        )
        
        # Wait for batch timeout
        time.sleep(0.06)
        
        # Mock inference
        async def mock_inference(prompts, max_tokens):
            return ["Output"] * len(prompts)
        
        # Schedule batch
        result = await self.scheduler.schedule_batch(mock_inference)
        
        assert result is not None
        # Should be limited to max batch size
        assert result.success_count <= self.batch_manager.policy.max_batch_size
    
    @pytest.mark.asyncio
    async def test_preemption_for_critical_priority(self):
        """Test batch preemption for critical priority requests."""
        # Add critical priority request
        critical_req = BatchRequest(
            request_id="critical",
            model_name="test-model",
            prompt="Critical request",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            priority=BatchPriority.CRITICAL
        )
        
        # Wait a bit so it ages
        time.sleep(0.15)
        
        self.batch_manager.submit_request(critical_req)
        
        # Update load
        self.scheduler.update_load_metrics(
            queue_depth=1,
            active_batches=0,
            worker_utilization=0.3,
            avg_latency_ms=100.0
        )
        
        # Check preemption
        should_preempt = self.scheduler._should_preempt()
        
        assert should_preempt is True
    
    @pytest.mark.asyncio
    async def test_adaptive_sizing_updates(self):
        """Test that adaptive sizing is updated after batch processing."""
        # Add requests
        for i in range(8):
            self.batch_manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test prompt",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ))
        
        # Update load
        self.scheduler.update_load_metrics(
            queue_depth=8,
            active_batches=0,
            worker_utilization=0.5,
            avg_latency_ms=100.0
        )
        
        # Wait for batch timeout
        time.sleep(0.06)
        
        # Mock inference
        async def mock_inference(prompts, max_tokens):
            return ["Output"] * len(prompts)
        
        # Schedule batch
        result = await self.scheduler.schedule_batch(mock_inference)
        
        assert result is not None
        
        # Adaptive sizer should have been updated
        stats = self.scheduler.adaptive_sizer.get_stats()
        assert len(stats["tracked_sizes"]) > 0
    
    def test_get_stats(self):
        """Test getting scheduler statistics."""
        # Update load
        self.scheduler.update_load_metrics(
            queue_depth=10,
            active_batches=2,
            worker_utilization=0.6,
            avg_latency_ms=150.0
        )
        
        stats = self.scheduler.get_stats()
        
        assert "batches_scheduled" in stats
        assert "preemptions" in stats
        assert "adaptive_adjustments" in stats
        assert "current_load" in stats
        assert "adaptive_sizer" in stats
        assert stats["current_load"]["queue_depth"] == 10
    
    def test_get_load_history(self):
        """Test getting load history."""
        # Add load metrics over time
        for i in range(5):
            self.scheduler.update_load_metrics(
                queue_depth=i * 10,
                active_batches=i,
                worker_utilization=i * 0.2,
                avg_latency_ms=100.0 + i * 50
            )
            time.sleep(0.01)
        
        # Get history
        history = self.scheduler.get_load_history(window_seconds=1.0)
        
        assert len(history) == 5
        assert all(isinstance(load, LoadMetrics) for load in history)
    
    def test_scheduler_without_adaptive_sizing(self):
        """Test scheduler with adaptive sizing disabled."""
        scheduler = BatchScheduler(
            batch_manager=self.batch_manager,
            batch_processor=self.batch_processor,
            enable_adaptive_sizing=False
        )
        
        assert scheduler.adaptive_sizer is None
        
        # Should still work
        size = scheduler.get_batch_size()
        assert size >= self.batch_manager.policy.min_batch_size
    
    def test_scheduler_without_preemption(self):
        """Test scheduler with preemption disabled."""
        scheduler = BatchScheduler(
            batch_manager=self.batch_manager,
            batch_processor=self.batch_processor,
            enable_preemption=False
        )
        
        assert scheduler.enable_preemption is False
        
        # Preemption check should return False
        should_preempt = scheduler._should_preempt()
        assert should_preempt is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
