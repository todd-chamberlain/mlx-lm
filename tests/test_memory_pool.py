"""
Tests for Memory Pool System

Tests tensor memory pooling, reuse, memory pressure detection,
fragmentation monitoring, and idle tensor cleanup.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import mlx.core as mx

from common.memory_pool import (
    TensorMemoryPool,
    PooledTensor,
    MemoryPoolStats,
    MemoryPressureLevel,
    get_global_memory_pool,
    configure_global_memory_pool
)


class TestPooledTensor:
    """Test pooled tensor functionality."""
    
    def test_pooled_tensor_creation(self):
        """Test creating pooled tensor."""
        tensor = mx.zeros((10, 20))
        
        pooled = PooledTensor(
            tensor=tensor,
            shape=(10, 20),
            dtype=mx.float32,
            size_bytes=800,
            pool_key="(10, 20)_float32",
            allocated_at=time.time(),
            last_used=time.time()
        )
        
        assert pooled.shape == (10, 20)
        assert pooled.dtype == mx.float32
        assert pooled.size_bytes == 800
        assert pooled.use_count == 0
        assert pooled.in_use == False
    
    def test_mark_used(self):
        """Test marking tensor as used."""
        tensor = mx.zeros((5, 5))
        pooled = PooledTensor(
            tensor=tensor,
            shape=(5, 5),
            dtype=mx.float32,
            size_bytes=100,
            pool_key="test",
            allocated_at=time.time(),
            last_used=time.time()
        )
        
        initial_use_count = pooled.use_count
        pooled.mark_used()
        
        assert pooled.in_use == True
        assert pooled.use_count == initial_use_count + 1
    
    def test_mark_free(self):
        """Test marking tensor as free."""
        tensor = mx.zeros((5, 5))
        pooled = PooledTensor(
            tensor=tensor,
            shape=(5, 5),
            dtype=mx.float32,
            size_bytes=100,
            pool_key="test",
            allocated_at=time.time(),
            last_used=time.time()
        )
        
        pooled.mark_used()
        pooled.mark_free()
        
        assert pooled.in_use == False
    
    def test_age_calculation(self):
        """Test age calculation."""
        tensor = mx.zeros((5, 5))
        pooled = PooledTensor(
            tensor=tensor,
            shape=(5, 5),
            dtype=mx.float32,
            size_bytes=100,
            pool_key="test",
            allocated_at=time.time() - 10.0,
            last_used=time.time()
        )
        
        assert pooled.age_seconds >= 10.0
    
    def test_idle_calculation(self):
        """Test idle time calculation."""
        tensor = mx.zeros((5, 5))
        pooled = PooledTensor(
            tensor=tensor,
            shape=(5, 5),
            dtype=mx.float32,
            size_bytes=100,
            pool_key="test",
            allocated_at=time.time(),
            last_used=time.time() - 5.0
        )
        
        assert pooled.idle_seconds >= 5.0


class TestMemoryPoolStats:
    """Test memory pool statistics."""
    
    def test_stats_initialization(self):
        """Test stats initialization."""
        stats = MemoryPoolStats()
        
        assert stats.total_allocated_bytes == 0
        assert stats.total_in_use_bytes == 0
        assert stats.total_free_bytes == 0
        assert stats.allocations == 0
        assert stats.reuses == 0
    
    def test_utilization_rate(self):
        """Test utilization rate calculation."""
        stats = MemoryPoolStats()
        stats.total_allocated_bytes = 1000
        stats.total_in_use_bytes = 750
        
        assert abs(stats.utilization_rate - 0.75) < 0.001
    
    def test_reuse_rate(self):
        """Test reuse rate calculation."""
        stats = MemoryPoolStats()
        stats.cache_hits = 80
        stats.cache_misses = 20
        
        assert abs(stats.reuse_rate - 0.8) < 0.001
    
    def test_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = MemoryPoolStats()
        stats.total_allocated_bytes = 1024 * 1024
        stats.allocations = 10
        stats.reuses = 5
        
        stats_dict = stats.to_dict()
        
        assert "total_allocated_mb" in stats_dict
        assert "allocations" in stats_dict
        assert "reuses" in stats_dict
        assert "utilization_rate" in stats_dict


class TestTensorMemoryPool:
    """Test tensor memory pool functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pool = TensorMemoryPool(
            max_pool_size_bytes=10 * 1024 * 1024,  # 10MB
            max_idle_seconds=60.0,
            enable_auto_cleanup=False  # Disable for testing
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.pool.stop()
    
    def test_pool_initialization(self):
        """Test pool initialization."""
        assert self.pool.max_pool_size_bytes == 10 * 1024 * 1024
        assert self.pool.max_idle_seconds == 60.0
        assert self.pool.stats.total_tensors == 0
    
    def test_allocate_new_tensor(self):
        """Test allocating new tensor."""
        tensor = self.pool.allocate((10, 20), mx.float32)
        
        assert tensor.shape == (10, 20)
        assert tensor.dtype == mx.float32
        assert self.pool.stats.allocations == 1
        assert self.pool.stats.total_tensors == 1
        assert self.pool.stats.in_use_tensors == 1
    
    def test_allocate_and_release(self):
        """Test allocating and releasing tensor."""
        tensor = self.pool.allocate((10, 20), mx.float32)
        
        assert self.pool.stats.in_use_tensors == 1
        assert self.pool.stats.free_tensors == 0
        
        self.pool.release(tensor)
        
        assert self.pool.stats.in_use_tensors == 0
        assert self.pool.stats.free_tensors == 1
    
    def test_tensor_reuse(self):
        """Test tensor reuse from pool."""
        # Allocate and release
        tensor1 = self.pool.allocate((10, 20), mx.float32)
        self.pool.release(tensor1)
        
        # Allocate again with same shape/dtype
        tensor2 = self.pool.allocate((10, 20), mx.float32)
        
        # Should reuse same tensor
        assert tensor1 is tensor2
        assert self.pool.stats.reuses == 1
        assert self.pool.stats.cache_hits == 1
        assert self.pool.stats.allocations == 1  # Only one allocation
    
    def test_different_shapes_different_pools(self):
        """Test that different shapes use different pools."""
        tensor1 = self.pool.allocate((10, 20), mx.float32)
        tensor2 = self.pool.allocate((5, 10), mx.float32)
        
        assert tensor1 is not tensor2
        assert self.pool.stats.allocations == 2
        assert self.pool.stats.total_tensors == 2
    
    def test_different_dtypes_different_pools(self):
        """Test that different dtypes use different pools."""
        tensor1 = self.pool.allocate((10, 20), mx.float32)
        tensor2 = self.pool.allocate((10, 20), mx.float16)
        
        assert tensor1 is not tensor2
        assert self.pool.stats.allocations == 2
    
    def test_multiple_tensors_same_pool(self):
        """Test multiple tensors in same pool."""
        tensor1 = self.pool.allocate((10, 20), mx.float32)
        tensor2 = self.pool.allocate((10, 20), mx.float32)
        
        # Both should be allocated (first is in use)
        assert tensor1 is not tensor2
        assert self.pool.stats.allocations == 2
        assert self.pool.stats.in_use_tensors == 2
    
    def test_reuse_with_multiple_tensors(self):
        """Test reuse when multiple tensors available."""
        # Allocate and release multiple tensors
        tensor1 = self.pool.allocate((10, 20), mx.float32)
        tensor2 = self.pool.allocate((10, 20), mx.float32)
        
        self.pool.release(tensor1)
        self.pool.release(tensor2)
        
        # Allocate again
        tensor3 = self.pool.allocate((10, 20), mx.float32)
        
        # Should reuse one of the released tensors
        assert tensor3 is tensor1 or tensor3 is tensor2
        assert self.pool.stats.reuses == 1
    
    def test_clear_pool(self):
        """Test clearing pool."""
        # Allocate some tensors
        for i in range(5):
            self.pool.allocate((10, 20), mx.float32)
        
        assert self.pool.stats.total_tensors == 5
        
        self.pool.clear()
        
        assert self.pool.stats.total_tensors == 0
        assert self.pool.stats.total_allocated_bytes == 0
    
    def test_memory_pressure_low(self):
        """Test low memory pressure detection."""
        # Allocate small amount with different shapes and release some
        tensor1 = self.pool.allocate((10, 10), mx.float32)
        tensor2 = self.pool.allocate((11, 11), mx.float32)
        tensor3 = self.pool.allocate((12, 12), mx.float32)
        
        # Release most to keep utilization low
        self.pool.release(tensor1)
        self.pool.release(tensor2)
        
        pressure = self.pool.get_memory_pressure()
        
        assert pressure == MemoryPressureLevel.LOW
    
    def test_memory_pressure_high(self):
        """Test high memory pressure detection."""
        # Fill pool to ~90%
        target_size = int(self.pool.max_pool_size_bytes * 0.9)
        tensor_size = 100 * 100 * 4  # 40KB per tensor
        num_tensors = target_size // tensor_size
        
        for i in range(num_tensors):
            self.pool.allocate((100, 100), mx.float32)
        
        pressure = self.pool.get_memory_pressure()
        
        assert pressure in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]
    
    def test_cleanup_idle_tensors(self):
        """Test cleanup of idle tensors."""
        # Allocate and release tensors with different shapes to avoid reuse
        for i in range(5):
            tensor = self.pool.allocate((10 + i, 20 + i), mx.float32)
            self.pool.release(tensor)
        
        assert self.pool.stats.free_tensors == 5
        
        # Clean up with 0 idle time (should remove all)
        self.pool.cleanup_idle_tensors(max_idle_seconds=0.0)
        
        assert self.pool.stats.free_tensors == 0
        assert self.pool.stats.total_tensors == 0
    
    def test_cleanup_respects_in_use(self):
        """Test that cleanup doesn't remove in-use tensors."""
        # Allocate tensors
        tensor1 = self.pool.allocate((10, 20), mx.float32)
        tensor2 = self.pool.allocate((10, 20), mx.float32)
        
        # Release one
        self.pool.release(tensor2)
        
        # Clean up
        self.pool.cleanup_idle_tensors(max_idle_seconds=0.0)
        
        # In-use tensor should remain
        assert self.pool.stats.in_use_tensors == 1
        assert self.pool.stats.total_tensors == 1
    
    def test_fragmentation_ratio(self):
        """Test fragmentation ratio calculation."""
        # No fragmentation initially
        assert self.pool.get_fragmentation_ratio() == 0.0
        
        # Allocate tensors with different shapes
        self.pool.allocate((10, 10), mx.float32)
        self.pool.allocate((20, 20), mx.float32)
        self.pool.allocate((30, 30), mx.float32)
        
        # Should have some fragmentation (3 pools, 3 tensors)
        fragmentation = self.pool.get_fragmentation_ratio()
        assert fragmentation > 0.0
    
    def test_get_stats(self):
        """Test getting pool statistics."""
        # Allocate some tensors
        tensor1 = self.pool.allocate((10, 20), mx.float32)
        tensor2 = self.pool.allocate((10, 20), mx.float32)
        self.pool.release(tensor1)
        
        stats = self.pool.get_stats()
        
        assert "total_allocated_mb" in stats
        assert "in_use_tensors" in stats
        assert "free_tensors" in stats
        assert "memory_pressure" in stats
        assert "fragmentation_ratio" in stats
        assert stats["in_use_tensors"] == 1
        assert stats["free_tensors"] == 1
    
    def test_get_pool_info(self):
        """Test getting pool information."""
        # Allocate tensors
        self.pool.allocate((10, 20), mx.float32)
        self.pool.allocate((10, 20), mx.float32)
        self.pool.allocate((5, 10), mx.float32)
        
        pool_info = self.pool.get_pool_info()
        
        assert len(pool_info) == 2  # Two different shapes
        assert all("pool_key" in info for info in pool_info)
        assert all("total_tensors" in info for info in pool_info)
        assert all("in_use" in info for info in pool_info)
    
    def test_pool_size_limit(self):
        """Test that pool respects size limit."""
        # Try to allocate more than pool size
        tensor_size = 100 * 100 * 4  # 40KB per tensor
        num_tensors = (self.pool.max_pool_size_bytes // tensor_size) + 10
        
        for i in range(num_tensors):
            tensor = self.pool.allocate((100, 100), mx.float32)
            if i < num_tensors - 5:
                self.pool.release(tensor)
        
        # Should not exceed pool size (much)
        assert self.pool.stats.total_allocated_bytes <= self.pool.max_pool_size_bytes * 1.1
    
    def test_thread_safety(self):
        """Test thread-safe pool operations."""
        def worker(worker_id: int):
            for i in range(50):
                tensor = self.pool.allocate((10, 20), mx.float32)
                time.sleep(0.001)
                self.pool.release(tensor)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Pool should be in consistent state
        stats = self.pool.get_stats()
        assert stats["total_allocated_mb"] <= self.pool.max_pool_size_bytes / (1024**2)
    
    def test_auto_cleanup_thread(self):
        """Test automatic cleanup thread."""
        pool = TensorMemoryPool(
            max_pool_size_bytes=10 * 1024 * 1024,
            max_idle_seconds=0.1,  # Very short for testing
            cleanup_interval_seconds=0.2,
            enable_auto_cleanup=True
        )
        
        try:
            # Allocate and release tensors with different shapes to avoid reuse
            for i in range(5):
                tensor = pool.allocate((10 + i, 20 + i), mx.float32)
                pool.release(tensor)
            
            assert pool.stats.free_tensors == 5
            
            # Wait for cleanup
            time.sleep(0.5)
            
            # Should have cleaned up idle tensors
            assert pool.stats.free_tensors < 5
        finally:
            pool.stop()


class TestGlobalMemoryPool:
    """Test global memory pool functionality."""
    
    def test_get_global_memory_pool(self):
        """Test getting global memory pool."""
        pool1 = get_global_memory_pool()
        pool2 = get_global_memory_pool()
        
        # Should return same instance
        assert pool1 is pool2
    
    def test_configure_global_memory_pool(self):
        """Test configuring global memory pool."""
        configure_global_memory_pool(
            max_pool_size_bytes=4 * 1024 * 1024 * 1024,
            max_idle_seconds=600.0,
            enable_auto_cleanup=False
        )
        
        pool = get_global_memory_pool()
        
        assert pool.max_pool_size_bytes == 4 * 1024 * 1024 * 1024
        assert pool.max_idle_seconds == 600.0
        assert pool.enable_auto_cleanup == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
