"""
Integration tests for the complete caching system.

Tests the integration of activation cache, KV cache, and cache coordination
to ensure they work together correctly and provide expected performance benefits.
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch
import mlx.core as mx

from common.activation_cache import (
    LRUActivationCache,
    DistributedCacheCoordinator as ActivationCoordinator,
    get_global_cache,
    configure_global_cache
)
from common.kv_cache import (
    KVCache,
    SharedKVCacheManager,
    get_global_kv_cache,
    configure_global_kv_cache
)
from coordinator.cache_coordinator import CacheCoordinator


class TestCacheIntegration:
    """Test integration of all caching components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create cache instances
        self.activation_cache = LRUActivationCache(
            max_size_bytes=10 * 1024 * 1024,
            max_entries=100,
            ttl_seconds=300.0
        )
        
        self.kv_cache = KVCache(
            max_cache_size_bytes=10 * 1024 * 1024,
            max_sequence_length=1024,
            enable_compression=False
        )
        
        self.cache_coordinator = CacheCoordinator(
            enable_cache_routing=True,
            cache_affinity_weight=0.3,
            cache_state_ttl=60.0
        )
    
    def test_activation_and_kv_cache_together(self):
        """Test using activation and KV cache together."""
        # Simulate inference with both caches
        model_name = "test-model"
        prompt = "Hello, world!"
        
        # Generate activation cache key
        act_key = self.activation_cache.generate_key(model_name, prompt)
        
        # Generate KV cache key (similar pattern)
        kv_key = f"kv_{act_key}"
        
        # First inference - cache miss
        act_result = self.activation_cache.get(act_key)
        kv_result = self.kv_cache.get(kv_key)
        
        assert act_result is None
        assert kv_result is None
        
        # Simulate inference and cache results
        activation = mx.ones((10, 64), dtype=mx.float32)
        key_cache = mx.ones((5, 32), dtype=mx.float32)
        value_cache = mx.ones((5, 32), dtype=mx.float32)
        
        self.activation_cache.put(act_key, activation)
        self.kv_cache.put(kv_key, key_cache, value_cache, sequence_length=5)
        
        # Second inference - cache hit
        act_result = self.activation_cache.get(act_key)
        kv_result = self.kv_cache.get(kv_key)
        
        assert act_result is not None
        assert kv_result is not None
        assert mx.array_equal(act_result, activation).item()
        
        # Verify statistics
        assert self.activation_cache.stats.hits == 1
        assert self.activation_cache.stats.misses == 1
        assert self.kv_cache.stats["hits"] == 1
        assert self.kv_cache.stats["misses"] == 1
    
    def test_cache_coordinator_with_worker_caches(self):
        """Test cache coordinator managing worker caches."""
        # Register workers with coordinator
        self.cache_coordinator.register_worker("worker-0")
        self.cache_coordinator.register_worker("worker-1")
        
        # Simulate workers caching data
        cache_key = "shared_prompt_key"
        
        # Worker 0 has the cache
        self.cache_coordinator.update_worker_cache_state(
            "worker-0",
            cached_keys={cache_key}
        )
        
        # Check cache-aware routing
        from coordinator.models import WorkerInfo, WorkerStatus, DeviceCapabilities, WorkerMetrics
        
        workers = [
            WorkerInfo(
                worker_id="worker-0",
                address="localhost:5000",
                device_capabilities=DeviceCapabilities(
                    model="test-model",
                    memory_gb=16,
                    gpu_cores=10,
                    os_version="14.0",
                    mlx_version="0.1.0"
                ),
                max_concurrent=4,
                supported_models=["test-model"],
                status=WorkerStatus.HEALTHY,
                metrics=WorkerMetrics()
            ),
            WorkerInfo(
                worker_id="worker-1",
                address="localhost:5001",
                device_capabilities=DeviceCapabilities(
                    model="test-model",
                    memory_gb=16,
                    gpu_cores=10,
                    os_version="14.0",
                    mlx_version="0.1.0"
                ),
                max_concurrent=4,
                supported_models=["test-model"],
                status=WorkerStatus.HEALTHY,
                metrics=WorkerMetrics()
            )
        ]
        
        # Route request with cache affinity
        decision = self.cache_coordinator.select_worker_with_cache_affinity(
            cache_key,
            workers
        )
        
        assert decision is not None
        assert decision.worker_id == "worker-0"
        assert decision.has_cache is True
    
    def test_multi_turn_conversation_caching(self):
        """Test caching effectiveness for multi-turn conversations."""
        model_name = "test-model"
        base_prompt = "You are a helpful assistant."
        
        # Turn 1
        turn1_prompt = base_prompt + "\nUser: Hello"
        turn1_key = self.activation_cache.generate_key(model_name, turn1_prompt)
        
        # Cache miss on first turn
        result = self.activation_cache.get(turn1_key)
        assert result is None
        
        # Simulate inference and cache
        turn1_activation = mx.ones((10, 64), dtype=mx.float32)
        self.activation_cache.put(turn1_key, turn1_activation)
        
        # Turn 2 - extends conversation
        turn2_prompt = turn1_prompt + "\nAssistant: Hi!\nUser: How are you?"
        turn2_key = self.activation_cache.generate_key(model_name, turn2_prompt)
        
        # Different key, so cache miss
        result = self.activation_cache.get(turn2_key)
        assert result is None
        
        # But we can use KV cache extension
        kv_key = f"kv_{turn1_key}"
        key_cache = mx.ones((5, 32), dtype=mx.float32)
        value_cache = mx.ones((5, 32), dtype=mx.float32)
        self.kv_cache.put(kv_key, key_cache, value_cache, sequence_length=5)
        
        # Extend KV cache for turn 2
        new_key = mx.ones((3, 32), dtype=mx.float32)
        new_value = mx.ones((3, 32), dtype=mx.float32)
        success = self.kv_cache.extend(kv_key, new_key, new_value, new_sequence_length=8)
        
        assert success is True
        
        # Verify extended cache
        result = self.kv_cache.get(kv_key, required_length=8)
        assert result is not None
    
    def test_cache_sharing_across_similar_requests(self):
        """Test cache sharing for similar requests."""
        shared_manager = SharedKVCacheManager(self.kv_cache)
        
        # Create shared KV cache
        cache_key = "shared_prefix"
        key_cache = mx.ones((10, 32), dtype=mx.float32)
        value_cache = mx.ones((10, 32), dtype=mx.float32)
        
        self.kv_cache.put(cache_key, key_cache, value_cache, sequence_length=10)
        
        # Multiple requests acquire the same cache
        result1 = shared_manager.acquire(cache_key)
        result2 = shared_manager.acquire(cache_key)
        result3 = shared_manager.acquire(cache_key)
        
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        
        # Check shared count
        assert shared_manager.get_shared_count(cache_key) == 3
        
        # Release references
        shared_manager.release(cache_key)
        shared_manager.release(cache_key)
        
        assert shared_manager.get_shared_count(cache_key) == 1
        
        shared_manager.release(cache_key)
        assert shared_manager.get_shared_count(cache_key) == 0
    
    def test_cache_invalidation_propagation(self):
        """Test cache invalidation across all cache layers."""
        model_name = "test-model"
        prompt = "Test prompt"
        
        # Create cache entries at all levels
        act_key = self.activation_cache.generate_key(model_name, prompt)
        kv_key = f"kv_{act_key}"
        
        # Add to activation cache
        activation = mx.ones((10, 64), dtype=mx.float32)
        self.activation_cache.put(act_key, activation)
        
        # Add to KV cache
        key_cache = mx.ones((5, 32), dtype=mx.float32)
        value_cache = mx.ones((5, 32), dtype=mx.float32)
        self.kv_cache.put(kv_key, key_cache, value_cache, sequence_length=5)
        
        # Register with coordinator
        self.cache_coordinator.register_worker("worker-0")
        self.cache_coordinator.update_worker_cache_state(
            "worker-0",
            cached_keys={act_key}
        )
        
        # Verify all caches have data
        assert self.activation_cache.get(act_key) is not None
        assert self.kv_cache.get(kv_key) is not None
        assert len(self.cache_coordinator.get_workers_with_cache(act_key)) == 1
        
        # Invalidate at all levels
        self.activation_cache.invalidate(act_key)
        self.kv_cache.invalidate(kv_key)
        self.cache_coordinator.invalidate_cache_key(act_key)
        
        # Verify all caches are cleared
        assert self.activation_cache.get(act_key) is None
        assert self.kv_cache.get(kv_key) is None
        assert len(self.cache_coordinator.get_workers_with_cache(act_key)) == 0
    
    def test_memory_pressure_coordination(self):
        """Test cache behavior under memory pressure."""
        # Create small caches to trigger eviction
        small_act_cache = LRUActivationCache(
            max_size_bytes=1024 * 100,  # 100KB
            max_entries=10
        )
        
        small_kv_cache = KVCache(
            max_cache_size_bytes=1024 * 50,  # 50KB - smaller to trigger eviction
            max_sequence_length=1024
        )
        
        # Fill activation cache
        for i in range(20):
            key = f"act_key_{i}"
            value = mx.ones((10, 64), dtype=mx.float32)
            small_act_cache.put(key, value)
        
        # Should have evicted some entries
        assert small_act_cache.stats.evictions > 0
        assert small_act_cache.stats.entry_count <= 10
        
        # Fill KV cache with larger entries to trigger eviction
        for i in range(20):
            key = f"kv_key_{i}"
            key_cache = mx.ones((20, 64), dtype=mx.float32)  # Larger entries
            value_cache = mx.ones((20, 64), dtype=mx.float32)
            small_kv_cache.put(key, key_cache, value_cache, sequence_length=20)
        
        # Should have evicted some entries or limited by size
        # KV cache doesn't have max_entries, only max_cache_size_bytes
        assert small_kv_cache.stats["total_size_bytes"] <= small_kv_cache.max_cache_size_bytes


class TestCachePerformanceImpact:
    """Test performance impact of caching."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.activation_cache = LRUActivationCache(
            max_size_bytes=100 * 1024 * 1024,
            max_entries=1000
        )
        
        self.kv_cache = KVCache(
            max_cache_size_bytes=100 * 1024 * 1024,
            max_sequence_length=4096
        )
    
    def test_cache_hit_latency(self):
        """Test that cache hits are faster than misses."""
        model_name = "test-model"
        prompt = "Test prompt for latency"
        
        key = self.activation_cache.generate_key(model_name, prompt)
        
        # Simulate expensive computation
        large_activation = mx.ones((100, 512), dtype=mx.float32)
        
        # Cache the result
        self.activation_cache.put(key, large_activation)
        
        # Measure cache hit time
        start = time.time()
        for _ in range(100):
            result = self.activation_cache.get(key)
        hit_time = time.time() - start
        
        # Cache hits should be very fast
        assert hit_time < 0.1  # 100 cache hits in < 100ms
        assert result is not None
    
    def test_cache_effectiveness_repeated_prompts(self):
        """Test cache effectiveness with repeated prompts."""
        model_name = "test-model"
        prompts = [
            "Hello, how are you?",
            "What is the weather?",
            "Tell me a joke",
            "Hello, how are you?",  # Repeat
            "What is the weather?",  # Repeat
        ]
        
        for prompt in prompts:
            key = self.activation_cache.generate_key(model_name, prompt)
            
            result = self.activation_cache.get(key)
            if result is None:
                # Simulate inference
                activation = mx.ones((10, 64), dtype=mx.float32)
                self.activation_cache.put(key, activation)
        
        # Should have 2 hits (2 repeated prompts)
        assert self.activation_cache.stats.hits == 2
        assert self.activation_cache.stats.misses == 3
        assert self.activation_cache.stats.hit_rate == 0.4
    
    def test_kv_cache_extension_efficiency(self):
        """Test KV cache extension is efficient."""
        cache_key = "conversation_key"
        
        # Initial cache
        key_cache = mx.ones((10, 32), dtype=mx.float32)
        value_cache = mx.ones((10, 32), dtype=mx.float32)
        self.kv_cache.put(cache_key, key_cache, value_cache, sequence_length=10)
        
        # Extend multiple times (simulating conversation)
        for i in range(5):
            new_key = mx.ones((5, 32), dtype=mx.float32)
            new_value = mx.ones((5, 32), dtype=mx.float32)
            new_length = 10 + (i + 1) * 5
            
            success = self.kv_cache.extend(
                cache_key,
                new_key,
                new_value,
                new_sequence_length=new_length
            )
            assert success is True
        
        # Verify final cache
        result = self.kv_cache.get(cache_key, required_length=35)
        assert result is not None
        
        # Should have 5 extensions
        assert self.kv_cache.stats["extensions"] == 5


class TestCacheConsistency:
    """Test cache consistency across distributed workers."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.coordinator = CacheCoordinator(
            enable_cache_routing=True,
            cache_affinity_weight=0.3
        )
        
        # Create worker caches
        self.worker_caches = {
            "worker-0": LRUActivationCache(max_size_bytes=10 * 1024 * 1024, max_entries=100),
            "worker-1": LRUActivationCache(max_size_bytes=10 * 1024 * 1024, max_entries=100),
            "worker-2": LRUActivationCache(max_size_bytes=10 * 1024 * 1024, max_entries=100),
        }
        
        # Register workers
        for worker_id in self.worker_caches:
            self.coordinator.register_worker(worker_id)
    
    def test_cache_replication_tracking(self):
        """Test tracking cache replication across workers."""
        cache_key = "replicated_key"
        
        # Add cache to multiple workers
        for worker_id in ["worker-0", "worker-1"]:
            self.coordinator.update_worker_cache_state(
                worker_id,
                cached_keys={cache_key}
            )
        
        # Check replication
        workers_with_cache = self.coordinator.get_workers_with_cache(cache_key)
        assert len(workers_with_cache) == 2
        assert "worker-0" in workers_with_cache
        assert "worker-1" in workers_with_cache
        
        # Get distribution stats
        distribution = self.coordinator.get_cache_distribution()
        assert distribution["total_unique_keys"] == 1
        assert distribution["average_replication_factor"] == 2.0
    
    def test_cache_warming_coordination(self):
        """Test coordinated cache warming."""
        cache_key = "warm_key"
        
        # Only worker-0 has the cache
        self.coordinator.update_worker_cache_state(
            "worker-0",
            cached_keys={cache_key}
        )
        
        # Suggest warming to achieve replication of 3
        suggested = self.coordinator.suggest_cache_warming(
            cache_key,
            target_replication=3
        )
        
        assert len(suggested) == 2
        assert "worker-0" not in suggested  # Already has it
    
    def test_cache_invalidation_broadcast(self):
        """Test cache invalidation across all workers."""
        cache_key = "invalid_key"
        
        # Add cache to all workers
        for worker_id in self.worker_caches:
            self.coordinator.update_worker_cache_state(
                worker_id,
                cached_keys={cache_key}
            )
        
        # Verify all have it
        assert len(self.coordinator.get_workers_with_cache(cache_key)) == 3
        
        # Invalidate
        affected = self.coordinator.invalidate_cache_key(cache_key)
        
        assert affected == 3
        assert len(self.coordinator.get_workers_with_cache(cache_key)) == 0


class TestCacheThreadSafety:
    """Test thread safety of caching system."""
    
    def test_concurrent_cache_access(self):
        """Test concurrent access to caches."""
        activation_cache = LRUActivationCache(
            max_size_bytes=10 * 1024 * 1024,
            max_entries=100
        )
        
        kv_cache = KVCache(
            max_cache_size_bytes=10 * 1024 * 1024,
            max_sequence_length=1024
        )
        
        def worker(worker_id: int):
            for i in range(50):
                # Activation cache operations
                act_key = f"worker_{worker_id}_act_{i}"
                activation = mx.ones((10, 64), dtype=mx.float32)
                activation_cache.put(act_key, activation)
                activation_cache.get(act_key)
                
                # KV cache operations
                kv_key = f"worker_{worker_id}_kv_{i}"
                key_cache = mx.ones((5, 32), dtype=mx.float32)
                value_cache = mx.ones((5, 32), dtype=mx.float32)
                kv_cache.put(kv_key, key_cache, value_cache, sequence_length=5)
                kv_cache.get(kv_key)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Caches should be in consistent state
        act_stats = activation_cache.get_stats()
        kv_stats = kv_cache.get_stats()
        
        assert act_stats["entry_count"] <= 100
        # KV cache doesn't have max_entries limit, only size limit
        assert kv_stats["total_size_bytes"] <= kv_cache.max_cache_size_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
