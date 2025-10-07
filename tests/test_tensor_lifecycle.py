"""
Tests for Tensor Lifecycle Management.

This module tests tensor lifecycle tracking, eager deallocation,
memory leak detection, and profiling capabilities.
"""

import pytest
import time
import mlx.core as mx
from common.tensor_lifecycle import (
    TensorLifecycleManager,
    TensorInfo,
    MemoryLeakReport,
    TensorLifecycleStats,
    get_lifecycle_manager,
    reset_lifecycle_manager
)


@pytest.fixture
def lifecycle_manager():
    """Create a fresh lifecycle manager for testing."""
    manager = TensorLifecycleManager(
        leak_detection_threshold_seconds=1.0,  # Short threshold for testing
        enable_profiling=True
    )
    yield manager
    manager.clear_all()


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return mx.zeros((10, 20), dtype=mx.float32)


class TestTensorLifecycleManager:
    """Test TensorLifecycleManager functionality."""
    
    def test_track_tensor(self, lifecycle_manager, sample_tensor):
        """Test basic tensor tracking."""
        tensor_id = "test_tensor_1"
        
        lifecycle_manager.track_tensor(tensor_id, sample_tensor, context="test")
        
        # Verify tensor is tracked
        info = lifecycle_manager.get_tensor_info(tensor_id)
        assert info is not None
        assert info.tensor_id == tensor_id
        assert info.shape == (10, 20)
        assert "float32" in info.dtype  # Handle both 'float32' and 'mlx.core.float32'
        assert info.reference_count == 1
        assert info.context == "test"
        
        # Verify stats
        stats = lifecycle_manager.get_stats()
        assert stats.total_allocations == 1
        assert stats.current_tracked == 1
        assert stats.total_bytes_allocated > 0
    
    def test_track_multiple_tensors(self, lifecycle_manager):
        """Test tracking multiple tensors."""
        tensors = [
            ("tensor_1", mx.zeros((5, 5), dtype=mx.float32), "context_a"),
            ("tensor_2", mx.zeros((10, 10), dtype=mx.float16), "context_b"),
            ("tensor_3", mx.zeros((3, 3), dtype=mx.int32), "context_a")
        ]
        
        for tensor_id, tensor, context in tensors:
            lifecycle_manager.track_tensor(tensor_id, tensor, context)
        
        # Verify all tracked
        stats = lifecycle_manager.get_stats()
        assert stats.total_allocations == 3
        assert stats.current_tracked == 3
        
        # Verify each tensor
        for tensor_id, tensor, context in tensors:
            info = lifecycle_manager.get_tensor_info(tensor_id)
            assert info is not None
            assert info.context == context
    
    def test_increment_reference(self, lifecycle_manager, sample_tensor):
        """Test incrementing reference count."""
        tensor_id = "test_tensor"
        
        lifecycle_manager.track_tensor(tensor_id, sample_tensor)
        assert lifecycle_manager.get_reference_count(tensor_id) == 1
        
        lifecycle_manager.increment_reference(tensor_id)
        assert lifecycle_manager.get_reference_count(tensor_id) == 2
        
        lifecycle_manager.increment_reference(tensor_id)
        assert lifecycle_manager.get_reference_count(tensor_id) == 3
    
    def test_release_tensor_with_references(self, lifecycle_manager, sample_tensor):
        """Test releasing tensor with multiple references."""
        tensor_id = "test_tensor"
        
        lifecycle_manager.track_tensor(tensor_id, sample_tensor)
        lifecycle_manager.increment_reference(tensor_id)
        lifecycle_manager.increment_reference(tensor_id)
        
        assert lifecycle_manager.get_reference_count(tensor_id) == 3
        
        # Release once - should not deallocate
        deallocated = lifecycle_manager.release_tensor(tensor_id)
        assert not deallocated
        assert lifecycle_manager.get_reference_count(tensor_id) == 2
        
        # Release again - should not deallocate
        deallocated = lifecycle_manager.release_tensor(tensor_id)
        assert not deallocated
        assert lifecycle_manager.get_reference_count(tensor_id) == 1
        
        # Release final time - should deallocate
        deallocated = lifecycle_manager.release_tensor(tensor_id)
        assert deallocated
        assert lifecycle_manager.get_reference_count(tensor_id) == 0
        assert lifecycle_manager.get_tensor_info(tensor_id) is None
    
    def test_eager_deallocation(self, lifecycle_manager, sample_tensor):
        """Test eager deallocation when reference count reaches zero."""
        tensor_id = "test_tensor"
        
        lifecycle_manager.track_tensor(tensor_id, sample_tensor)
        
        stats_before = lifecycle_manager.get_stats()
        assert stats_before.current_tracked == 1
        
        # Release tensor - should eagerly deallocate
        deallocated = lifecycle_manager.release_tensor(tensor_id)
        assert deallocated
        
        stats_after = lifecycle_manager.get_stats()
        assert stats_after.current_tracked == 0
        assert stats_after.total_deallocations == 1
        assert stats_after.eager_deallocations == 1
        
        # Verify tensor is no longer tracked
        assert lifecycle_manager.get_tensor_info(tensor_id) is None
    
    def test_memory_accounting(self, lifecycle_manager):
        """Test memory byte accounting."""
        # Track tensors of different sizes
        tensors = [
            ("small", mx.zeros((5, 5), dtype=mx.float32)),  # 100 bytes
            ("medium", mx.zeros((10, 10), dtype=mx.float32)),  # 400 bytes
            ("large", mx.zeros((20, 20), dtype=mx.float32))  # 1600 bytes
        ]
        
        total_expected_bytes = 0
        for tensor_id, tensor in tensors:
            lifecycle_manager.track_tensor(tensor_id, tensor)
            info = lifecycle_manager.get_tensor_info(tensor_id)
            total_expected_bytes += info.size_bytes
        
        stats = lifecycle_manager.get_stats()
        assert stats.total_bytes_allocated == total_expected_bytes
        assert stats.current_bytes_tracked == total_expected_bytes
        
        # Release one tensor
        lifecycle_manager.release_tensor("small")
        
        stats = lifecycle_manager.get_stats()
        assert stats.total_deallocations == 1
        assert stats.current_bytes_tracked < total_expected_bytes
    
    def test_leak_detection(self, lifecycle_manager, sample_tensor):
        """Test memory leak detection."""
        # Track tensors with old timestamps
        old_tensor_id = "old_tensor"
        lifecycle_manager.track_tensor(old_tensor_id, sample_tensor, context="leak_test")
        
        # Manually set old timestamp
        info = lifecycle_manager.tracked_tensors[old_tensor_id]
        info.created_at = time.time() - 7200  # 2 hours ago
        info.last_accessed = time.time() - 3600  # 1 hour ago
        
        # Track a recent tensor
        recent_tensor_id = "recent_tensor"
        lifecycle_manager.track_tensor(recent_tensor_id, sample_tensor, context="normal")
        
        # Detect leaks
        leak_report = lifecycle_manager.detect_leaks()
        
        assert len(leak_report.leaked_tensors) == 1
        assert leak_report.leaked_tensors[0].tensor_id == old_tensor_id
        assert leak_report.total_leaked_bytes > 0
        assert "leak_test" in leak_report.leak_contexts
        assert leak_report.oldest_leak_age_seconds > 7000
    
    def test_cleanup_leaked_tensors(self, lifecycle_manager, sample_tensor):
        """Test cleaning up leaked tensors."""
        # Create leaked tensors
        for i in range(3):
            tensor_id = f"leaked_tensor_{i}"
            lifecycle_manager.track_tensor(tensor_id, sample_tensor, context="leak")
            
            # Make them old
            info = lifecycle_manager.tracked_tensors[tensor_id]
            info.created_at = time.time() - 7200
            info.last_accessed = time.time() - 3600
        
        # Create normal tensor
        lifecycle_manager.track_tensor("normal", sample_tensor, context="normal")
        
        assert lifecycle_manager.get_stats().current_tracked == 4
        
        # Cleanup leaked tensors
        cleaned = lifecycle_manager.cleanup_leaked_tensors()
        
        assert cleaned == 3
        assert lifecycle_manager.get_stats().current_tracked == 1
        assert lifecycle_manager.get_tensor_info("normal") is not None
    
    def test_memory_profiling(self, lifecycle_manager):
        """Test memory profiling capabilities."""
        # Track tensors in different contexts
        contexts = ["inference", "cache", "activation"]
        for i, context in enumerate(contexts):
            for j in range(3):
                tensor_id = f"{context}_tensor_{j}"
                tensor = mx.zeros((10 * (i + 1), 10), dtype=mx.float32)
                lifecycle_manager.track_tensor(tensor_id, tensor, context=context)
        
        # Get memory profile
        profile = lifecycle_manager.get_memory_profile()
        
        assert 'stats' in profile
        assert 'current_memory_by_context' in profile
        assert 'current_count_by_context' in profile
        assert 'total_allocations_by_context' in profile
        
        # Verify context tracking
        for context in contexts:
            assert context in profile['current_memory_by_context']
            assert profile['current_count_by_context'][context] == 3
            assert profile['total_allocations_by_context'][context] == 3
    
    def test_allocation_rate(self, lifecycle_manager, sample_tensor):
        """Test allocation rate calculation."""
        # Track and release tensors
        for i in range(10):
            tensor_id = f"tensor_{i}"
            lifecycle_manager.track_tensor(tensor_id, sample_tensor)
            
            if i < 5:
                lifecycle_manager.release_tensor(tensor_id)
        
        stats = lifecycle_manager.get_stats()
        assert stats.total_allocations == 10
        assert stats.total_deallocations == 5
        assert stats.get_allocation_rate() == 2.0
    
    def test_memory_efficiency(self, lifecycle_manager):
        """Test memory efficiency calculation."""
        # Track tensors of different sizes
        tensors = [
            mx.zeros((10, 10), dtype=mx.float32),
            mx.zeros((20, 20), dtype=mx.float32),
            mx.zeros((5, 5), dtype=mx.float32)
        ]
        
        for i, tensor in enumerate(tensors):
            tensor_id = f"tensor_{i}"
            lifecycle_manager.track_tensor(tensor_id, tensor)
        
        # Release first two
        lifecycle_manager.release_tensor("tensor_0")
        lifecycle_manager.release_tensor("tensor_1")
        
        stats = lifecycle_manager.get_stats()
        efficiency = stats.get_memory_efficiency()
        
        # Should have deallocated most of the memory
        assert 0.0 < efficiency < 1.0
    
    def test_thread_safety(self, lifecycle_manager, sample_tensor):
        """Test thread-safe operations."""
        import threading
        
        def track_and_release():
            for i in range(10):
                tensor_id = f"thread_{threading.current_thread().ident}_{i}"
                lifecycle_manager.track_tensor(tensor_id, sample_tensor)
                lifecycle_manager.release_tensor(tensor_id)
        
        threads = [threading.Thread(target=track_and_release) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All tensors should be deallocated
        stats = lifecycle_manager.get_stats()
        assert stats.current_tracked == 0
        assert stats.total_allocations == 50
        assert stats.total_deallocations == 50
    
    def test_clear_all(self, lifecycle_manager, sample_tensor):
        """Test clearing all tracked tensors."""
        # Track multiple tensors
        for i in range(5):
            tensor_id = f"tensor_{i}"
            lifecycle_manager.track_tensor(tensor_id, sample_tensor)
        
        assert lifecycle_manager.get_stats().current_tracked == 5
        
        # Clear all
        cleared = lifecycle_manager.clear_all()
        
        assert cleared == 5
        assert lifecycle_manager.get_stats().current_tracked == 0
    
    def test_tensor_size_calculation(self, lifecycle_manager):
        """Test tensor size calculation for different dtypes."""
        test_cases = [
            (mx.zeros((10, 10), dtype=mx.float32), 400),  # 100 * 4
            (mx.zeros((10, 10), dtype=mx.float16), 200),  # 100 * 2
            (mx.zeros((10, 10), dtype=mx.int32), 400),    # 100 * 4
            (mx.zeros((10, 10), dtype=mx.int8), 100),     # 100 * 1
        ]
        
        for i, (tensor, expected_size) in enumerate(test_cases):
            tensor_id = f"tensor_{i}"
            lifecycle_manager.track_tensor(tensor_id, tensor)
            
            info = lifecycle_manager.get_tensor_info(tensor_id)
            assert info.size_bytes == expected_size
    
    def test_context_tracking(self, lifecycle_manager, sample_tensor):
        """Test context-based tracking."""
        contexts = ["inference", "cache", "activation", "gradient"]
        
        for i, context in enumerate(contexts):
            tensor_id = f"tensor_{i}"
            lifecycle_manager.track_tensor(tensor_id, sample_tensor, context=context)
        
        # Verify context tracking
        profile = lifecycle_manager.get_memory_profile()
        
        for context in contexts:
            assert context in profile['current_count_by_context']
            assert profile['current_count_by_context'][context] == 1


class TestGlobalLifecycleManager:
    """Test global lifecycle manager singleton."""
    
    def test_get_lifecycle_manager(self):
        """Test getting global lifecycle manager."""
        reset_lifecycle_manager()
        
        manager1 = get_lifecycle_manager()
        manager2 = get_lifecycle_manager()
        
        # Should return same instance
        assert manager1 is manager2
    
    def test_reset_lifecycle_manager(self):
        """Test resetting global lifecycle manager."""
        manager1 = get_lifecycle_manager()
        
        reset_lifecycle_manager()
        
        manager2 = get_lifecycle_manager()
        
        # Should return different instance after reset
        assert manager1 is not manager2


class TestInferenceEngineIntegration:
    """Test integration with InferenceEngine."""
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager."""
        class MockModelManager:
            def get_model(self, model_name):
                # Return mock model and tokenizer
                class MockModel:
                    pass
                
                class MockTokenizer:
                    def encode(self, text):
                        return [1, 2, 3, 4, 5]
                
                return (MockModel(), MockTokenizer())
        
        return MockModelManager()
    
    def test_lifecycle_tracking_in_inference(self, mock_model_manager):
        """Test that inference engine tracks tensors."""
        from worker.inference_engine import InferenceEngine
        
        reset_lifecycle_manager()
        
        engine = InferenceEngine(
            model_manager=mock_model_manager,
            worker_id="test_worker",
            enable_gc_optimization=False,
            enable_lifecycle_tracking=True
        )
        
        assert engine.lifecycle_manager is not None
        
        # Get metrics should include lifecycle stats
        metrics = engine.get_metrics()
        assert 'tensor_lifecycle' in metrics
    
    def test_memory_leak_detection_in_engine(self, mock_model_manager):
        """Test memory leak detection through engine."""
        from worker.inference_engine import InferenceEngine
        
        reset_lifecycle_manager()
        
        engine = InferenceEngine(
            model_manager=mock_model_manager,
            worker_id="test_worker",
            enable_gc_optimization=False,
            enable_lifecycle_tracking=True
        )
        
        # Detect leaks
        leak_info = engine.detect_memory_leaks()
        
        assert leak_info['enabled'] is True
        assert 'leaked_tensor_count' in leak_info
    
    def test_memory_profiling_in_engine(self, mock_model_manager):
        """Test memory profiling through engine."""
        from worker.inference_engine import InferenceEngine
        
        reset_lifecycle_manager()
        
        engine = InferenceEngine(
            model_manager=mock_model_manager,
            worker_id="test_worker",
            enable_gc_optimization=False,
            enable_lifecycle_tracking=True
        )
        
        # Get memory profile
        profile = engine.get_memory_profile()
        
        assert profile['enabled'] is True
        assert 'stats' in profile


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
