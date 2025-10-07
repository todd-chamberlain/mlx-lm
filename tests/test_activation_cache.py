"""
Tests for activation cache system.

Tests LRU cache functionality, cache key generation, eviction policies,
TTL expiration, distributed cache coordination, and cache statistics.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import mlx.core as mx

from common.activation_cache import (
    LRUActivationCache,
    CacheEntry,
    CacheStats,
    DistributedCacheCoordinator,
    get_global_cache,
    configure_global_cache
)


class TestCacheEntry:
    """Test cache entry functionality."""
    
    def test_cache_entry_creation(self):
        """Test creating cache entry."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            size_bytes=1024,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.size_bytes == 1024
        assert entry.access_count == 0
        assert entry.hit_count == 0
    
    def test_cache_entry_update_access(self):
        """Test updating cache entry access statistics."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            size_bytes=1024,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        initial_access_time = entry.last_accessed
        initial_count = entry.access_count
        
        time.sleep(0.01)
        entry.update_access()
        
        assert entry.last_accessed > initial_access_time
        assert entry.access_count == initial_count + 1


class TestCacheStats:
    """Test cache statistics functionality."""
    
    def test_cache_stats_initialization(self):
        """Test cache stats initialization."""
        stats = CacheStats()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.total_size_bytes == 0
        assert stats.entry_count == 0
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats()
        stats.hits = 80
        stats.misses = 20
        
        assert abs(stats.hit_rate - 0.8) < 0.001
        assert abs(stats.miss_rate - 0.2) < 0.001
    
    def test_hit_rate_with_no_requests(self):
        """Test hit rate with no requests."""
        stats = CacheStats()
        
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 1.0
    
    def test_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = CacheStats()
        stats.hits = 100
        stats.misses = 50
        stats.evictions = 10
        stats.total_size_bytes = 1024 * 1024
        stats.entry_count = 25
        
        stats_dict = stats.to_dict()
        
        assert stats_dict["hits"] == 100
        assert stats_dict["misses"] == 50
        assert stats_dict["evictions"] == 10
        assert stats_dict["total_size_bytes"] == 1024 * 1024
        assert stats_dict["entry_count"] == 25
        assert abs(stats_dict["hit_rate"] - 0.6667) < 0.001


class TestLRUActivationCache:
    """Test LRU activation cache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = LRUActivationCache(
            max_size_bytes=1024 * 1024,  # 1MB
            max_entries=10,
            ttl_seconds=60.0
        )
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        assert self.cache.max_size_bytes == 1024 * 1024
        assert self.cache.max_entries == 10
        assert self.cache.ttl_seconds == 60.0
        assert self.cache.stats.entry_count == 0
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        key1 = self.cache.generate_key("model1", "test prompt")
        key2 = self.cache.generate_key("model1", "test prompt")
        key3 = self.cache.generate_key("model2", "test prompt")
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different key
        assert key1 != key3
        
        # Key should have expected format
        assert key1.startswith("act_")
    
    def test_generate_cache_key_with_layer(self):
        """Test cache key generation with layer index."""
        key1 = self.cache.generate_key("model1", "prompt", layer_idx=0)
        key2 = self.cache.generate_key("model1", "prompt", layer_idx=1)
        
        # Different layers should generate different keys
        assert key1 != key2
    
    def test_generate_cache_key_with_kwargs(self):
        """Test cache key generation with additional parameters."""
        key1 = self.cache.generate_key("model1", "prompt", temperature=0.7)
        key2 = self.cache.generate_key("model1", "prompt", temperature=0.8)
        
        # Different parameters should generate different keys
        assert key1 != key2
    
    def test_put_and_get(self):
        """Test putting and getting values from cache."""
        key = "test_key"
        value = "test_value"
        
        self.cache.put(key, value, size_bytes=100)
        retrieved = self.cache.get(key)
        
        assert retrieved == value
        assert self.cache.stats.hits == 1
        assert self.cache.stats.misses == 0
    
    def test_cache_miss(self):
        """Test cache miss."""
        result = self.cache.get("nonexistent_key")
        
        assert result is None
        assert self.cache.stats.misses == 1
        assert self.cache.stats.hits == 0
    
    def test_cache_with_mlx_array(self):
        """Test caching MLX arrays."""
        key = "mlx_array_key"
        value = mx.array([1.0, 2.0, 3.0])
        
        self.cache.put(key, value)
        retrieved = self.cache.get(key)
        
        assert isinstance(retrieved, mx.array)
        assert mx.array_equal(retrieved, value).item()
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        # Fill cache to capacity
        for i in range(10):
            self.cache.put(f"key_{i}", f"value_{i}", size_bytes=1000)
        
        assert self.cache.stats.entry_count == 10
        
        # Add one more entry, should evict oldest
        self.cache.put("key_new", "value_new", size_bytes=1000)
        
        assert self.cache.stats.entry_count == 10
        assert self.cache.stats.evictions == 1
        
        # First key should be evicted
        assert self.cache.get("key_0") is None
        
        # New key should be present
        assert self.cache.get("key_new") == "value_new"
    
    def test_lru_ordering(self):
        """Test that accessing entries updates LRU order."""
        # Add entries
        self.cache.put("key_1", "value_1", size_bytes=1000)
        self.cache.put("key_2", "value_2", size_bytes=1000)
        self.cache.put("key_3", "value_3", size_bytes=1000)
        
        # Access key_1 to make it most recently used
        self.cache.get("key_1")
        
        # Fill cache to trigger eviction
        for i in range(8):
            self.cache.put(f"key_new_{i}", f"value_new_{i}", size_bytes=1000)
        
        # key_2 should be evicted (least recently used)
        # key_1 should still be present (was accessed)
        assert self.cache.get("key_2") is None
        assert self.cache.get("key_1") == "value_1"
    
    def test_size_based_eviction(self):
        """Test eviction based on size limit."""
        # Add entries that fill cache
        for i in range(5):
            self.cache.put(f"key_{i}", f"value_{i}", size_bytes=200 * 1024)
        
        # Add large entry that requires eviction
        self.cache.put("large_key", "large_value", size_bytes=300 * 1024)
        
        # Should have evicted entries to make space
        assert self.cache.stats.evictions > 0
        assert self.cache.stats.total_size_bytes <= self.cache.max_size_bytes
    
    def test_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        # Create cache with short TTL
        cache = LRUActivationCache(
            max_size_bytes=1024 * 1024,
            max_entries=10,
            ttl_seconds=0.1  # 100ms TTL
        )
        
        cache.put("key", "value", size_bytes=100)
        
        # Should be present immediately
        assert cache.get("key") == "value"
        
        # Wait for TTL to expire
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("key") is None
        assert cache.stats.misses == 1
    
    def test_invalidate(self):
        """Test cache invalidation."""
        self.cache.put("key", "value", size_bytes=100)
        
        assert self.cache.get("key") == "value"
        
        # Invalidate entry
        result = self.cache.invalidate("key")
        
        assert result == True
        assert self.cache.get("key") is None
    
    def test_invalidate_nonexistent(self):
        """Test invalidating nonexistent entry."""
        result = self.cache.invalidate("nonexistent")
        
        assert result == False
    
    def test_clear(self):
        """Test clearing cache."""
        # Add entries
        for i in range(5):
            self.cache.put(f"key_{i}", f"value_{i}", size_bytes=1000)
        
        assert self.cache.stats.entry_count == 5
        
        # Clear cache
        self.cache.clear()
        
        assert self.cache.stats.entry_count == 0
        assert self.cache.stats.total_size_bytes == 0
        
        # All entries should be gone
        for i in range(5):
            assert self.cache.get(f"key_{i}") is None
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        self.cache.put("key1", "value1", size_bytes=100)
        self.cache.get("key1")
        self.cache.get("nonexistent")
        
        stats = self.cache.get_stats()
        
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entry_count"] == 1
        assert stats["total_size_bytes"] == 100
    
    def test_get_entries_info(self):
        """Test getting cache entries information."""
        self.cache.put("key1", "value1", size_bytes=100)
        self.cache.put("key2", "value2", size_bytes=200)
        
        entries = self.cache.get_entries_info()
        
        assert len(entries) == 2
        assert all("key" in entry for entry in entries)
        assert all("size_bytes" in entry for entry in entries)
        assert all("age_seconds" in entry for entry in entries)
    
    def test_thread_safety(self):
        """Test thread-safe cache operations."""
        def worker(worker_id: int):
            for i in range(100):
                key = f"worker_{worker_id}_key_{i}"
                self.cache.put(key, f"value_{i}", size_bytes=100)
                self.cache.get(key)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Cache should be in consistent state
        stats = self.cache.get_stats()
        assert stats["entry_count"] <= self.cache.max_entries
        assert stats["total_size_bytes"] <= self.cache.max_size_bytes
    
    def test_size_estimation_mlx_array(self):
        """Test size estimation for MLX arrays."""
        array = mx.array([1.0, 2.0, 3.0, 4.0])
        estimated_size = self.cache._estimate_size(array)
        
        assert estimated_size == array.nbytes
    
    def test_size_estimation_list_of_arrays(self):
        """Test size estimation for list of MLX arrays."""
        arrays = [
            mx.array([1.0, 2.0]),
            mx.array([3.0, 4.0, 5.0])
        ]
        estimated_size = self.cache._estimate_size(arrays)
        
        expected_size = sum(arr.nbytes for arr in arrays)
        assert estimated_size == expected_size
    
    def test_entry_too_large(self):
        """Test handling of entries too large for cache."""
        # Try to add entry larger than max cache size
        large_value = "x" * (self.cache.max_size_bytes + 1000)
        
        self.cache.put("large_key", large_value, size_bytes=len(large_value))
        
        # Entry should not be added
        assert self.cache.get("large_key") is None
        assert self.cache.stats.entry_count == 0


class TestDistributedCacheCoordinator:
    """Test distributed cache coordinator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.coordinator = DistributedCacheCoordinator()
        self.cache1 = LRUActivationCache(max_size_bytes=1024 * 1024, max_entries=10)
        self.cache2 = LRUActivationCache(max_size_bytes=1024 * 1024, max_entries=10)
    
    def test_coordinator_initialization(self):
        """Test coordinator initialization."""
        assert len(self.coordinator.worker_caches) == 0
    
    def test_register_worker_cache(self):
        """Test registering worker cache."""
        self.coordinator.register_worker_cache("worker1", self.cache1)
        
        assert "worker1" in self.coordinator.worker_caches
        assert self.coordinator.worker_caches["worker1"] == self.cache1
    
    def test_unregister_worker_cache(self):
        """Test unregistering worker cache."""
        self.coordinator.register_worker_cache("worker1", self.cache1)
        self.coordinator.unregister_worker_cache("worker1")
        
        assert "worker1" not in self.coordinator.worker_caches
    
    def test_invalidate_all(self):
        """Test invalidating cache entry across all workers."""
        # Register workers
        self.coordinator.register_worker_cache("worker1", self.cache1)
        self.coordinator.register_worker_cache("worker2", self.cache2)
        
        # Add same key to both caches
        self.cache1.put("shared_key", "value1", size_bytes=100)
        self.cache2.put("shared_key", "value2", size_bytes=100)
        
        # Invalidate across all workers
        self.coordinator.invalidate_all("shared_key")
        
        # Key should be removed from both caches
        assert self.cache1.get("shared_key") is None
        assert self.cache2.get("shared_key") is None
    
    def test_clear_all(self):
        """Test clearing all worker caches."""
        # Register workers
        self.coordinator.register_worker_cache("worker1", self.cache1)
        self.coordinator.register_worker_cache("worker2", self.cache2)
        
        # Add entries to both caches
        self.cache1.put("key1", "value1", size_bytes=100)
        self.cache2.put("key2", "value2", size_bytes=100)
        
        # Clear all caches
        self.coordinator.clear_all()
        
        # Both caches should be empty
        assert self.cache1.stats.entry_count == 0
        assert self.cache2.stats.entry_count == 0
    
    def test_get_aggregate_stats(self):
        """Test getting aggregated cache statistics."""
        # Register workers
        self.coordinator.register_worker_cache("worker1", self.cache1)
        self.coordinator.register_worker_cache("worker2", self.cache2)
        
        # Add entries and access them
        self.cache1.put("key1", "value1", size_bytes=100)
        self.cache1.get("key1")
        self.cache1.get("nonexistent")
        
        self.cache2.put("key2", "value2", size_bytes=200)
        self.cache2.get("key2")
        
        # Get aggregate stats
        stats = self.coordinator.get_aggregate_stats()
        
        assert stats["worker_count"] == 2
        assert stats["total_hits"] == 2
        assert stats["total_misses"] == 1
        assert stats["total_entries"] == 2
        assert stats["total_size_bytes"] == 300
        assert "worker1" in stats["workers"]
        assert "worker2" in stats["workers"]
    
    def test_aggregate_hit_rate(self):
        """Test aggregate hit rate calculation."""
        # Register workers
        self.coordinator.register_worker_cache("worker1", self.cache1)
        self.coordinator.register_worker_cache("worker2", self.cache2)
        
        # Worker 1: 8 hits, 2 misses
        for i in range(10):
            self.cache1.put(f"key_{i}", f"value_{i}", size_bytes=100)
        for i in range(8):
            self.cache1.get(f"key_{i}")
        self.cache1.get("miss1")
        self.cache1.get("miss2")
        
        # Worker 2: 6 hits, 4 misses
        for i in range(10):
            self.cache2.put(f"key_{i}", f"value_{i}", size_bytes=100)
        for i in range(6):
            self.cache2.get(f"key_{i}")
        for i in range(4):
            self.cache2.get(f"miss_{i}")
        
        stats = self.coordinator.get_aggregate_stats()
        
        # Aggregate: 14 hits, 6 misses = 70% hit rate
        assert abs(stats["aggregate_hit_rate"] - 0.7) < 0.01


class TestGlobalCache:
    """Test global cache instance functionality."""
    
    def test_get_global_cache(self):
        """Test getting global cache instance."""
        cache1 = get_global_cache()
        cache2 = get_global_cache()
        
        # Should return same instance
        assert cache1 is cache2
    
    def test_configure_global_cache(self):
        """Test configuring global cache."""
        configure_global_cache(
            max_size_bytes=2 * 1024 * 1024,
            max_entries=500,
            ttl_seconds=1800.0
        )
        
        cache = get_global_cache()
        
        assert cache.max_size_bytes == 2 * 1024 * 1024
        assert cache.max_entries == 500
        assert cache.ttl_seconds == 1800.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
