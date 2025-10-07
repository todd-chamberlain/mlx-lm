"""
Tests for KV cache optimization system.

Tests KV cache functionality, cache sharing, extension, compression,
and shared cache management.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import mlx.core as mx
import numpy as np

from common.kv_cache import (
    KVCache,
    KVCacheEntry,
    SharedKVCacheManager,
    get_global_kv_cache,
    configure_global_kv_cache
)


class TestKVCacheEntry:
    """Test KV cache entry functionality."""
    
    def test_kv_cache_entry_creation(self):
        """Test creating KV cache entry."""
        key_cache = mx.array([[1.0, 2.0], [3.0, 4.0]])
        value_cache = mx.array([[5.0, 6.0], [7.0, 8.0]])
        
        entry = KVCacheEntry(
            key_cache=key_cache,
            value_cache=value_cache,
            sequence_length=2,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        assert entry.sequence_length == 2
        assert entry.access_count == 0
        assert entry.shared_count == 0
    
    def test_kv_cache_entry_size(self):
        """Test KV cache entry size calculation."""
        key_cache = mx.array([[1.0, 2.0], [3.0, 4.0]])
        value_cache = mx.array([[5.0, 6.0], [7.0, 8.0]])
        
        entry = KVCacheEntry(
            key_cache=key_cache,
            value_cache=value_cache,
            sequence_length=2,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        expected_size = key_cache.nbytes + value_cache.nbytes
        assert entry.size_bytes == expected_size
    
    def test_kv_cache_entry_age(self):
        """Test KV cache entry age calculation."""
        entry = KVCacheEntry(
            key_cache=mx.array([[1.0]]),
            value_cache=mx.array([[2.0]]),
            sequence_length=1,
            created_at=time.time() - 10.0,
            last_accessed=time.time()
        )
        
        assert entry.age_seconds >= 10.0
    
    def test_kv_cache_entry_update_access(self):
        """Test updating KV cache entry access."""
        entry = KVCacheEntry(
            key_cache=mx.array([[1.0]]),
            value_cache=mx.array([[2.0]]),
            sequence_length=1,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        initial_access_time = entry.last_accessed
        initial_count = entry.access_count
        
        time.sleep(0.01)
        entry.update_access()
        
        assert entry.last_accessed > initial_access_time
        assert entry.access_count == initial_count + 1


class TestKVCache:
    """Test KV cache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = KVCache(
            max_cache_size_bytes=10 * 1024 * 1024,  # 10MB
            max_sequence_length=1024,
            enable_compression=False
        )
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        assert self.cache.max_cache_size_bytes == 10 * 1024 * 1024
        assert self.cache.max_sequence_length == 1024
        assert self.cache.enable_compression == False
        assert self.cache.stats["entry_count"] == 0
    
    def test_put_and_get(self):
        """Test putting and getting KV cache."""
        key_cache = mx.array([[1.0, 2.0], [3.0, 4.0]])
        value_cache = mx.array([[5.0, 6.0], [7.0, 8.0]])
        
        self.cache.put("test_key", key_cache, value_cache, sequence_length=2)
        
        result = self.cache.get("test_key")
        
        assert result is not None
        retrieved_key, retrieved_value = result
        assert mx.array_equal(retrieved_key, key_cache).item()
        assert mx.array_equal(retrieved_value, value_cache).item()
        assert self.cache.stats["hits"] == 1
    
    def test_cache_miss(self):
        """Test cache miss."""
        result = self.cache.get("nonexistent_key")
        
        assert result is None
        assert self.cache.stats["misses"] == 1
    
    def test_cache_with_required_length(self):
        """Test cache retrieval with required length."""
        key_cache = mx.array([[1.0, 2.0], [3.0, 4.0]])
        value_cache = mx.array([[5.0, 6.0], [7.0, 8.0]])
        
        self.cache.put("test_key", key_cache, value_cache, sequence_length=2)
        
        # Should hit with exact length
        result = self.cache.get("test_key", required_length=2)
        assert result is not None
        
        # Should hit with shorter required length
        result = self.cache.get("test_key", required_length=1)
        assert result is not None
        
        # Should miss with longer required length
        result = self.cache.get("test_key", required_length=3)
        assert result is None
    
    def test_cache_extend(self):
        """Test extending KV cache."""
        # Initial cache
        key_cache = mx.array([[1.0, 2.0], [3.0, 4.0]])
        value_cache = mx.array([[5.0, 6.0], [7.0, 8.0]])
        
        self.cache.put("test_key", key_cache, value_cache, sequence_length=2)
        
        # Extend cache
        new_key = mx.array([[9.0, 10.0]])
        new_value = mx.array([[11.0, 12.0]])
        
        success = self.cache.extend("test_key", new_key, new_value, new_sequence_length=3)
        
        assert success == True
        assert self.cache.stats["extensions"] == 1
        
        # Verify extended cache
        result = self.cache.get("test_key")
        assert result is not None
        retrieved_key, retrieved_value = result
        assert retrieved_key.shape[0] == 3
        assert retrieved_value.shape[0] == 3
    
    def test_cache_extend_nonexistent(self):
        """Test extending nonexistent cache."""
        new_key = mx.array([[1.0, 2.0]])
        new_value = mx.array([[3.0, 4.0]])
        
        success = self.cache.extend("nonexistent", new_key, new_value, new_sequence_length=1)
        
        assert success == False
    
    def test_cache_share(self):
        """Test marking cache as shared."""
        key_cache = mx.array([[1.0, 2.0]])
        value_cache = mx.array([[3.0, 4.0]])
        
        self.cache.put("test_key", key_cache, value_cache, sequence_length=1)
        
        # Mark as shared
        success = self.cache.share("test_key")
        assert success == True
        assert self.cache.stats["shared_accesses"] == 1
        
        # Get entry and check shared count
        entries = self.cache.get_entries_info()
        assert len(entries) == 1
        assert entries[0]["shared_count"] == 1
    
    def test_cache_unshare(self):
        """Test unmarking cache as shared."""
        key_cache = mx.array([[1.0, 2.0]])
        value_cache = mx.array([[3.0, 4.0]])
        
        self.cache.put("test_key", key_cache, value_cache, sequence_length=1)
        
        # Share and unshare
        self.cache.share("test_key")
        success = self.cache.unshare("test_key")
        
        assert success == True
        
        # Check shared count
        entries = self.cache.get_entries_info()
        assert entries[0]["shared_count"] == 0
    
    def test_cache_invalidate(self):
        """Test cache invalidation."""
        key_cache = mx.array([[1.0, 2.0]])
        value_cache = mx.array([[3.0, 4.0]])
        
        self.cache.put("test_key", key_cache, value_cache, sequence_length=1)
        
        # Invalidate
        success = self.cache.invalidate("test_key")
        assert success == True
        
        # Should be gone
        result = self.cache.get("test_key")
        assert result is None
    
    def test_cache_clear(self):
        """Test clearing cache."""
        # Add multiple entries
        for i in range(5):
            key_cache = mx.array([[float(i), float(i+1)]])
            value_cache = mx.array([[float(i+2), float(i+3)]])
            self.cache.put(f"key_{i}", key_cache, value_cache, sequence_length=1)
        
        assert self.cache.stats["entry_count"] == 5
        
        # Clear cache
        self.cache.clear()
        
        assert self.cache.stats["entry_count"] == 0
        assert self.cache.stats["total_size_bytes"] == 0
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        # Fill cache with larger entries to trigger eviction
        for i in range(200):
            key_cache = mx.ones((100, 128), dtype=mx.float32)
            value_cache = mx.ones((100, 128), dtype=mx.float32)
            self.cache.put(f"key_{i}", key_cache, value_cache, sequence_length=100)
        
        # Should have evicted some entries
        assert self.cache.stats["evictions"] > 0
        assert self.cache.stats["total_size_bytes"] <= self.cache.max_cache_size_bytes
    
    def test_eviction_respects_shared_entries(self):
        """Test that shared entries are evicted last."""
        # Add entries
        for i in range(5):
            key_cache = mx.ones((100, 64), dtype=mx.float32)
            value_cache = mx.ones((100, 64), dtype=mx.float32)
            self.cache.put(f"key_{i}", key_cache, value_cache, sequence_length=100)
        
        # Mark first entry as shared
        self.cache.share("key_0")
        
        # Add more entries to trigger eviction
        for i in range(5, 10):
            key_cache = mx.ones((100, 64), dtype=mx.float32)
            value_cache = mx.ones((100, 64), dtype=mx.float32)
            self.cache.put(f"key_{i}", key_cache, value_cache, sequence_length=100)
        
        # Shared entry should still be present
        result = self.cache.get("key_0")
        assert result is not None
    
    def test_sequence_length_limit(self):
        """Test sequence length limit."""
        # Try to add entry exceeding max sequence length
        key_cache = mx.ones((2000, 64), dtype=mx.float32)
        value_cache = mx.ones((2000, 64), dtype=mx.float32)
        
        self.cache.put("long_key", key_cache, value_cache, sequence_length=2000)
        
        # Should not be added
        result = self.cache.get("long_key")
        assert result is None
    
    def test_cache_compression(self):
        """Test cache compression."""
        cache_with_compression = KVCache(
            max_cache_size_bytes=10 * 1024 * 1024,
            max_sequence_length=1024,
            enable_compression=True,
            compression_threshold=10
        )
        
        # Add entry that should be compressed
        key_cache = mx.ones((20, 64), dtype=mx.float32)
        value_cache = mx.ones((20, 64), dtype=mx.float32)
        
        original_size = key_cache.nbytes + value_cache.nbytes
        
        cache_with_compression.put("test_key", key_cache, value_cache, sequence_length=20)
        
        # Get entry info
        entries = cache_with_compression.get_entries_info()
        assert len(entries) == 1
        
        # Compressed size should be smaller (float32 -> float16)
        assert entries[0]["size_bytes"] < original_size
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        # Add entries and access them
        key_cache = mx.array([[1.0, 2.0]])
        value_cache = mx.array([[3.0, 4.0]])
        
        self.cache.put("key1", key_cache, value_cache, sequence_length=1)
        self.cache.get("key1")
        self.cache.get("nonexistent")
        
        stats = self.cache.get_stats()
        
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entry_count"] == 1
        assert "hit_rate" in stats
        assert "avg_entry_size_mb" in stats
    
    def test_get_entries_info(self):
        """Test getting cache entries information."""
        # Add entries
        for i in range(3):
            key_cache = mx.array([[float(i)]])
            value_cache = mx.array([[float(i+1)]])
            self.cache.put(f"key_{i}", key_cache, value_cache, sequence_length=1)
        
        entries = self.cache.get_entries_info()
        
        assert len(entries) == 3
        assert all("key" in entry for entry in entries)
        assert all("sequence_length" in entry for entry in entries)
        assert all("size_bytes" in entry for entry in entries)
    
    def test_warm_cache(self):
        """Test cache warming."""
        key_cache = mx.array([[1.0, 2.0], [3.0, 4.0]])
        value_cache = mx.array([[5.0, 6.0], [7.0, 8.0]])
        
        self.cache.warm_cache("warm_key", key_cache, value_cache, sequence_length=2)
        
        # Should be in cache
        result = self.cache.get("warm_key")
        assert result is not None
    
    def test_thread_safety(self):
        """Test thread-safe cache operations."""
        def worker(worker_id: int):
            for i in range(50):
                key_cache = mx.ones((5, 32), dtype=mx.float32)
                value_cache = mx.ones((5, 32), dtype=mx.float32)
                cache_key = f"worker_{worker_id}_key_{i}"
                self.cache.put(cache_key, key_cache, value_cache, sequence_length=5)
                self.cache.get(cache_key)
        
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
        assert stats["total_size_bytes"] <= self.cache.max_cache_size_bytes


class TestSharedKVCacheManager:
    """Test shared KV cache manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.kv_cache = KVCache(
            max_cache_size_bytes=10 * 1024 * 1024,
            max_sequence_length=1024
        )
        self.manager = SharedKVCacheManager(self.kv_cache)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        assert self.manager.kv_cache == self.kv_cache
    
    def test_acquire_and_release(self):
        """Test acquiring and releasing shared cache."""
        # Add cache entry
        key_cache = mx.array([[1.0, 2.0]])
        value_cache = mx.array([[3.0, 4.0]])
        self.kv_cache.put("test_key", key_cache, value_cache, sequence_length=1)
        
        # Acquire
        result = self.manager.acquire("test_key")
        assert result is not None
        assert self.manager.get_shared_count("test_key") == 1
        
        # Release
        self.manager.release("test_key")
        assert self.manager.get_shared_count("test_key") == 0
    
    def test_multiple_acquires(self):
        """Test multiple acquires of same cache."""
        # Add cache entry
        key_cache = mx.array([[1.0, 2.0]])
        value_cache = mx.array([[3.0, 4.0]])
        self.kv_cache.put("test_key", key_cache, value_cache, sequence_length=1)
        
        # Multiple acquires
        self.manager.acquire("test_key")
        self.manager.acquire("test_key")
        self.manager.acquire("test_key")
        
        assert self.manager.get_shared_count("test_key") == 3
        
        # Release all
        self.manager.release("test_key")
        self.manager.release("test_key")
        self.manager.release("test_key")
        
        assert self.manager.get_shared_count("test_key") == 0
    
    def test_acquire_nonexistent(self):
        """Test acquiring nonexistent cache."""
        result = self.manager.acquire("nonexistent")
        
        assert result is None
        assert self.manager.get_shared_count("nonexistent") == 0


class TestGlobalKVCache:
    """Test global KV cache instance functionality."""
    
    def test_get_global_kv_cache(self):
        """Test getting global KV cache instance."""
        cache1 = get_global_kv_cache()
        cache2 = get_global_kv_cache()
        
        # Should return same instance
        assert cache1 is cache2
    
    def test_configure_global_kv_cache(self):
        """Test configuring global KV cache."""
        configure_global_kv_cache(
            max_cache_size_bytes=4 * 1024 * 1024 * 1024,
            max_sequence_length=8192,
            enable_compression=True
        )
        
        cache = get_global_kv_cache()
        
        assert cache.max_cache_size_bytes == 4 * 1024 * 1024 * 1024
        assert cache.max_sequence_length == 8192
        assert cache.enable_compression == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
