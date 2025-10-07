"""
Tests for Batch Manager

Tests batch request queuing, batch assembly, dynamic sizing,
priority scheduling, and fairness policies.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from coordinator.batch_manager import (
    BatchRequest,
    Batch,
    BatchPriority,
    BatchingPolicy,
    BatchQueue,
    BatchManager
)


class TestBatchRequest:
    """Test batch request functionality."""
    
    def test_batch_request_creation(self):
        """Test creating batch request."""
        request = BatchRequest(
            request_id="req_1",
            model_name="test-model",
            prompt="Hello, world!",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
        
        assert request.request_id == "req_1"
        assert request.priority == BatchPriority.NORMAL
        assert request.prompt_length > 0
        assert request.estimated_tokens == 100
    
    def test_batch_request_age(self):
        """Test request age calculation."""
        request = BatchRequest(
            request_id="req_1",
            model_name="test-model",
            prompt="Test",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
        
        time.sleep(0.1)
        assert request.age_seconds >= 0.1
    
    def test_batch_request_expiration(self):
        """Test request expiration."""
        request = BatchRequest(
            request_id="req_1",
            model_name="test-model",
            prompt="Test",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            timeout_seconds=0.1
        )
        
        assert not request.is_expired
        time.sleep(0.15)
        assert request.is_expired
    
    def test_batch_request_priority_ordering(self):
        """Test priority-based ordering."""
        req_low = BatchRequest(
            request_id="low",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
            priority=BatchPriority.LOW
        )
        
        req_high = BatchRequest(
            request_id="high",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
            priority=BatchPriority.HIGH
        )
        
        # Higher priority should be "less than" (comes first)
        assert req_high < req_low


class TestBatch:
    """Test batch functionality."""
    
    def test_batch_creation(self):
        """Test creating batch."""
        requests = [
            BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt=f"Prompt {i}",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
            for i in range(5)
        ]
        
        batch = Batch(
            batch_id="batch_1",
            requests=requests
        )
        
        assert batch.batch_id == "batch_1"
        assert batch.size == 5
    
    def test_batch_max_prompt_length(self):
        """Test batch max prompt length calculation."""
        requests = [
            BatchRequest(
                request_id="req_1",
                model_name="test",
                prompt="Short",
                max_tokens=10,
                temperature=0.7,
                top_p=0.9
            ),
            BatchRequest(
                request_id="req_2",
                model_name="test",
                prompt="This is a much longer prompt with many words",
                max_tokens=10,
                temperature=0.7,
                top_p=0.9
            )
        ]
        
        batch = Batch(batch_id="batch_1", requests=requests)
        
        assert batch.max_prompt_length == max(r.prompt_length for r in requests)
    
    def test_batch_total_estimated_tokens(self):
        """Test batch total estimated tokens."""
        requests = [
            BatchRequest(
                request_id=f"req_{i}",
                model_name="test",
                prompt="test",
                max_tokens=50 * (i + 1),
                temperature=0.7,
                top_p=0.9
            )
            for i in range(3)
        ]
        
        batch = Batch(batch_id="batch_1", requests=requests)
        
        assert batch.total_estimated_tokens == 50 + 100 + 150


class TestBatchingPolicy:
    """Test batching policy functionality."""
    
    def test_policy_initialization(self):
        """Test policy initialization."""
        policy = BatchingPolicy(
            min_batch_size=2,
            max_batch_size=16,
            max_batch_wait_ms=100.0
        )
        
        assert policy.min_batch_size == 2
        assert policy.max_batch_size == 16
        assert policy.max_batch_wait_ms == 100.0
    
    def test_should_create_batch_at_max_size(self):
        """Test batch creation at max size."""
        policy = BatchingPolicy(max_batch_size=10)
        
        should_create = policy.should_create_batch(
            queue_size=10,
            oldest_request_age_ms=10.0,
            current_load=0.5
        )
        
        assert should_create is True
    
    def test_should_create_batch_after_wait(self):
        """Test batch creation after wait time."""
        policy = BatchingPolicy(
            min_batch_size=2,
            max_batch_wait_ms=50.0
        )
        
        should_create = policy.should_create_batch(
            queue_size=3,
            oldest_request_age_ms=60.0,
            current_load=0.5
        )
        
        assert should_create is True
    
    def test_should_not_create_batch_below_min(self):
        """Test batch not created below minimum size."""
        policy = BatchingPolicy(
            min_batch_size=5,
            max_batch_wait_ms=50.0
        )
        
        should_create = policy.should_create_batch(
            queue_size=3,
            oldest_request_age_ms=10.0,
            current_load=0.5
        )
        
        assert should_create is False
    
    def test_should_create_batch_under_high_load(self):
        """Test batch creation under high load."""
        policy = BatchingPolicy(min_batch_size=2)
        
        should_create = policy.should_create_batch(
            queue_size=3,
            oldest_request_age_ms=10.0,
            current_load=0.9
        )
        
        assert should_create is True
    
    def test_get_optimal_batch_size_high_load(self):
        """Test optimal batch size under high load."""
        policy = BatchingPolicy(
            min_batch_size=2,
            max_batch_size=32,
            enable_dynamic_sizing=True
        )
        
        optimal = policy.get_optimal_batch_size(current_load=0.95)
        
        # Should prefer smaller batches under high load
        assert optimal < policy.max_batch_size
    
    def test_get_optimal_batch_size_low_load(self):
        """Test optimal batch size under low load."""
        policy = BatchingPolicy(
            min_batch_size=2,
            max_batch_size=32,
            enable_dynamic_sizing=True
        )
        
        optimal = policy.get_optimal_batch_size(current_load=0.2)
        
        # Should prefer larger batches under low load
        assert optimal == policy.max_batch_size
    
    def test_update_performance(self):
        """Test performance-based policy updates."""
        policy = BatchingPolicy(
            min_batch_size=2,
            max_batch_size=32,
            enable_dynamic_sizing=True
        )
        
        # Simulate performance data showing size 16 is optimal
        for _ in range(10):
            policy.update_performance(batch_size=8, throughput=100.0)
            policy.update_performance(batch_size=16, throughput=200.0)
            policy.update_performance(batch_size=24, throughput=150.0)
        
        # Optimal size should adjust toward 16
        optimal = policy.get_optimal_batch_size(current_load=0.5)
        assert 12 <= optimal <= 20


class TestBatchQueue:
    """Test batch queue functionality."""
    
    def test_queue_initialization(self):
        """Test queue initialization."""
        queue = BatchQueue(enable_priority=True)
        
        assert queue.is_empty()
        assert queue.size() == 0
    
    def test_enqueue_and_dequeue(self):
        """Test basic enqueue and dequeue."""
        queue = BatchQueue(enable_priority=False)
        
        request = BatchRequest(
            request_id="req_1",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9
        )
        
        queue.enqueue(request)
        assert queue.size() == 1
        
        batch = queue.dequeue_batch(max_size=10)
        assert len(batch) == 1
        assert batch[0].request_id == "req_1"
        assert queue.is_empty()
    
    def test_priority_ordering(self):
        """Test priority-based ordering."""
        queue = BatchQueue(enable_priority=True)
        
        # Add requests with different priorities
        queue.enqueue(BatchRequest(
            request_id="low",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
            priority=BatchPriority.LOW
        ))
        
        queue.enqueue(BatchRequest(
            request_id="high",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
            priority=BatchPriority.HIGH
        ))
        
        queue.enqueue(BatchRequest(
            request_id="normal",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
            priority=BatchPriority.NORMAL
        ))
        
        # Dequeue should return high priority first
        batch = queue.dequeue_batch(max_size=3)
        assert batch[0].request_id == "high"
        assert batch[1].request_id == "normal"
        assert batch[2].request_id == "low"
    
    def test_fifo_ordering(self):
        """Test FIFO ordering without priority."""
        queue = BatchQueue(enable_priority=False)
        
        for i in range(5):
            queue.enqueue(BatchRequest(
                request_id=f"req_{i}",
                model_name="test",
                prompt="test",
                max_tokens=10,
                temperature=0.7,
                top_p=0.9
            ))
        
        batch = queue.dequeue_batch(max_size=5)
        
        # Should be in FIFO order
        for i, request in enumerate(batch):
            assert request.request_id == f"req_{i}"
    
    def test_dequeue_batch_size_limit(self):
        """Test dequeue respects max size."""
        queue = BatchQueue(enable_priority=False)
        
        for i in range(10):
            queue.enqueue(BatchRequest(
                request_id=f"req_{i}",
                model_name="test",
                prompt="test",
                max_tokens=10,
                temperature=0.7,
                top_p=0.9
            ))
        
        batch = queue.dequeue_batch(max_size=5)
        
        assert len(batch) == 5
        assert queue.size() == 5
    
    def test_peek_oldest(self):
        """Test peeking at oldest request."""
        queue = BatchQueue(enable_priority=False)
        
        first_request = BatchRequest(
            request_id="first",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9
        )
        
        queue.enqueue(first_request)
        time.sleep(0.01)
        
        queue.enqueue(BatchRequest(
            request_id="second",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9
        ))
        
        oldest = queue.peek_oldest()
        assert oldest.request_id == "first"
        assert queue.size() == 2  # Peek doesn't remove
    
    def test_expired_request_removal(self):
        """Test automatic removal of expired requests."""
        queue = BatchQueue(enable_priority=False)
        
        # Add request with short timeout
        queue.enqueue(BatchRequest(
            request_id="expired",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
            timeout_seconds=0.05
        ))
        
        # Add normal request
        queue.enqueue(BatchRequest(
            request_id="valid",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
            timeout_seconds=10.0
        ))
        
        # Wait for first to expire
        time.sleep(0.1)
        
        # Dequeue should skip expired
        batch = queue.dequeue_batch(max_size=10)
        assert len(batch) == 1
        assert batch[0].request_id == "valid"
        assert queue.stats["total_expired"] == 1
    
    def test_clear_queue(self):
        """Test clearing queue."""
        queue = BatchQueue(enable_priority=False)
        
        for i in range(5):
            queue.enqueue(BatchRequest(
                request_id=f"req_{i}",
                model_name="test",
                prompt="test",
                max_tokens=10,
                temperature=0.7,
                top_p=0.9
            ))
        
        assert queue.size() == 5
        queue.clear()
        assert queue.is_empty()
    
    def test_thread_safety(self):
        """Test thread-safe queue operations."""
        queue = BatchQueue(enable_priority=False)
        
        def enqueue_worker(worker_id: int):
            for i in range(50):
                queue.enqueue(BatchRequest(
                    request_id=f"worker_{worker_id}_req_{i}",
                    model_name="test",
                    prompt="test",
                    max_tokens=10,
                    temperature=0.7,
                    top_p=0.9
                ))
        
        def dequeue_worker():
            for _ in range(25):
                queue.dequeue_batch(max_size=5)
                time.sleep(0.001)
        
        # Create threads
        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=enqueue_worker, args=(i,)))
        threads.append(threading.Thread(target=dequeue_worker))
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Queue should be in consistent state
        assert queue.stats["total_enqueued"] == 150
        assert queue.stats["total_dequeued"] <= 150


class TestBatchManager:
    """Test batch manager functionality."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = BatchManager()
        
        assert manager.enable_priority is True
        assert manager.enable_fairness is True
        assert manager.queue.is_empty()
    
    def test_submit_request(self):
        """Test submitting request."""
        manager = BatchManager()
        
        request = BatchRequest(
            request_id="req_1",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9
        )
        
        manager.submit_request(request)
        
        assert manager.queue.size() == 1
    
    def test_create_batch(self):
        """Test creating batch."""
        policy = BatchingPolicy(
            min_batch_size=2,
            max_batch_size=10,
            max_batch_wait_ms=10.0
        )
        manager = BatchManager(policy=policy)
        
        # Submit requests
        for i in range(5):
            manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test",
                prompt="test",
                max_tokens=10,
                temperature=0.7,
                top_p=0.9
            ))
        
        # Wait for batch timeout
        time.sleep(0.02)
        
        # Create batch
        batch = manager.create_batch(current_load=0.5)
        
        assert batch is not None
        assert batch.size >= policy.min_batch_size
        assert batch.size <= policy.max_batch_size
    
    def test_create_batch_at_max_size(self):
        """Test batch creation at max size."""
        policy = BatchingPolicy(
            min_batch_size=2,
            max_batch_size=5
        )
        manager = BatchManager(policy=policy)
        
        # Submit max size requests
        for i in range(5):
            manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test",
                prompt="test",
                max_tokens=10,
                temperature=0.7,
                top_p=0.9
            ))
        
        # Should create batch immediately
        batch = manager.create_batch(current_load=0.5)
        
        assert batch is not None
        assert batch.size == 5
    
    def test_complete_batch(self):
        """Test completing batch."""
        policy = BatchingPolicy(
            min_batch_size=2,
            max_batch_size=10,
            max_batch_wait_ms=10.0
        )
        manager = BatchManager(policy=policy)
        
        # Submit and create batch
        for i in range(3):
            manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test",
                prompt="test",
                max_tokens=10,
                temperature=0.7,
                top_p=0.9
            ))
        
        # Wait for batch timeout
        time.sleep(0.02)
        
        batch = manager.create_batch(current_load=0.5)
        assert batch is not None
        
        # Complete batch
        manager.complete_batch(batch.batch_id, throughput=100.0)
        
        stats = manager.get_stats()
        assert stats["total_batches_created"] == 1
    
    def test_get_stats(self):
        """Test getting manager statistics."""
        manager = BatchManager()
        
        # Submit requests
        for i in range(5):
            manager.submit_request(BatchRequest(
                request_id=f"req_{i}",
                model_name="test",
                prompt="test",
                max_tokens=10,
                temperature=0.7,
                top_p=0.9
            ))
        
        stats = manager.get_stats()
        
        assert "queue_size" in stats
        assert "total_batches_created" in stats
        assert "avg_batch_size" in stats
        assert stats["queue_size"] == 5
    
    def test_priority_scheduling(self):
        """Test priority-based scheduling."""
        policy = BatchingPolicy(
            min_batch_size=2,
            max_batch_size=10,
            max_batch_wait_ms=10.0
        )
        manager = BatchManager(policy=policy, enable_priority=True)
        
        # Submit requests with different priorities
        manager.submit_request(BatchRequest(
            request_id="low",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
            priority=BatchPriority.LOW
        ))
        
        manager.submit_request(BatchRequest(
            request_id="high",
            model_name="test",
            prompt="test",
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
            priority=BatchPriority.HIGH
        ))
        
        # Wait for batch timeout
        time.sleep(0.02)
        
        # Create batch
        batch = manager.create_batch(current_load=0.5)
        
        assert batch is not None
        # High priority should be first
        assert batch.requests[0].request_id == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
