"""
Tests for Distributed Cache Coordination
"""

import pytest
import time
from coordinator.cache_coordinator import (
    CacheCoordinator,
    CacheState,
    CacheRoutingDecision
)
from coordinator.models import WorkerInfo, WorkerStatus, DeviceCapabilities, WorkerMetrics


@pytest.fixture
def cache_coordinator():
    """Create a cache coordinator for testing."""
    return CacheCoordinator(
        enable_cache_routing=True,
        cache_affinity_weight=0.3,
        cache_state_ttl=60.0
    )


@pytest.fixture
def sample_workers():
    """Create sample workers for testing."""
    workers = []
    for i in range(3):
        worker = WorkerInfo(
            worker_id=f"worker-{i}",
            address=f"localhost:500{i}",
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
        workers.append(worker)
    return workers


class TestCacheCoordinator:
    """Test cache coordinator functionality."""
    
    def test_initialization(self, cache_coordinator):
        """Test cache coordinator initialization."""
        assert cache_coordinator.enable_cache_routing is True
        assert cache_coordinator.cache_affinity_weight == 0.3
        assert cache_coordinator.cache_state_ttl == 60.0
        assert len(cache_coordinator._worker_cache_states) == 0
    
    def test_register_worker(self, cache_coordinator):
        """Test worker registration."""
        cache_coordinator.register_worker("worker-1")
        
        assert "worker-1" in cache_coordinator._worker_cache_states
        assert cache_coordinator._worker_cache_states["worker-1"].worker_id == "worker-1"
    
    def test_unregister_worker(self, cache_coordinator):
        """Test worker unregistration."""
        # Register and add cache state
        cache_coordinator.register_worker("worker-1")
        cache_coordinator.update_worker_cache_state(
            "worker-1",
            cached_keys={"key1", "key2"}
        )
        
        # Unregister
        cache_coordinator.unregister_worker("worker-1")
        
        assert "worker-1" not in cache_coordinator._worker_cache_states
        assert "key1" not in cache_coordinator._cache_key_locations
        assert "key2" not in cache_coordinator._cache_key_locations
    
    def test_update_cache_state(self, cache_coordinator):
        """Test updating worker cache state."""
        cache_coordinator.register_worker("worker-1")
        
        # Update with cached keys
        cached_keys = {"key1", "key2", "key3"}
        cache_coordinator.update_worker_cache_state(
            "worker-1",
            cached_keys=cached_keys
        )
        
        state = cache_coordinator._worker_cache_states["worker-1"]
        assert state.cached_keys == cached_keys
        assert cache_coordinator._cache_key_locations["key1"] == {"worker-1"}
        assert cache_coordinator._cache_key_locations["key2"] == {"worker-1"}
    
    def test_update_cache_stats(self, cache_coordinator):
        """Test updating cache statistics."""
        cache_coordinator.register_worker("worker-1")
        
        stats = {
            "total_size_bytes": 1024 * 1024 * 100,  # 100MB
            "hit_rate": 0.75
        }
        
        cache_coordinator.update_worker_cache_state(
            "worker-1",
            cache_stats=stats
        )
        
        state = cache_coordinator._worker_cache_states["worker-1"]
        assert state.cache_size_bytes == 1024 * 1024 * 100
        assert state.cache_hit_rate == 0.75
    
    def test_report_cache_access_hit(self, cache_coordinator):
        """Test reporting cache hit."""
        cache_coordinator.register_worker("worker-1")
        
        cache_coordinator.report_cache_access("worker-1", "key1", was_hit=True)
        
        assert "key1" in cache_coordinator._worker_cache_states["worker-1"].cached_keys
        assert cache_coordinator.stats["cache_hits_from_routing"] == 1
    
    def test_report_cache_access_miss(self, cache_coordinator):
        """Test reporting cache miss."""
        cache_coordinator.register_worker("worker-1")
        
        cache_coordinator.report_cache_access("worker-1", "key1", was_hit=False)
        
        assert cache_coordinator.stats["cache_misses_from_routing"] == 1
    
    def test_cache_aware_routing_with_cache(self, cache_coordinator, sample_workers):
        """Test cache-aware routing when cache exists."""
        # Setup: worker-0 has the cache
        cache_coordinator.register_worker("worker-0")
        cache_coordinator.update_worker_cache_state(
            "worker-0",
            cached_keys={"test-key"}
        )
        
        # Register other workers
        for worker in sample_workers[1:]:
            cache_coordinator.register_worker(worker.worker_id)
        
        # Route request
        decision = cache_coordinator.select_worker_with_cache_affinity(
            "test-key",
            sample_workers
        )
        
        assert decision is not None
        assert decision.worker_id == "worker-0"
        assert decision.has_cache is True
        assert cache_coordinator.stats["cache_aware_routes"] == 1
    
    def test_cache_aware_routing_without_cache(self, cache_coordinator, sample_workers):
        """Test cache-aware routing when no cache exists."""
        # Register workers without cache
        for worker in sample_workers:
            cache_coordinator.register_worker(worker.worker_id)
        
        # Route request
        decision = cache_coordinator.select_worker_with_cache_affinity(
            "test-key",
            sample_workers
        )
        
        # Should return None (no cache available)
        assert decision is None
    
    def test_cache_aware_routing_with_base_scores(self, cache_coordinator, sample_workers):
        """Test cache-aware routing with base scores."""
        # Setup: worker-1 has cache, worker-0 has better base score
        cache_coordinator.register_worker("worker-0")
        cache_coordinator.register_worker("worker-1")
        cache_coordinator.update_worker_cache_state(
            "worker-1",
            cached_keys={"test-key"}
        )
        
        # Base scores (lower is better)
        # With cache_affinity_weight=0.3:
        # worker-0: 0.5 (no cache) = 0.5
        # worker-1: 0.6 - 0.3 (cache bonus) = 0.3 (wins!)
        base_scores = {
            "worker-0": 0.5,
            "worker-1": 0.6,  # Worse base score but cache bonus makes it better
            "worker-2": 0.8
        }
        
        decision = cache_coordinator.select_worker_with_cache_affinity(
            "test-key",
            sample_workers,
            base_scores=base_scores
        )
        
        # Should prefer worker-1 due to cache affinity
        assert decision is not None
        assert decision.worker_id == "worker-1"
        assert decision.has_cache is True
    
    def test_get_workers_with_cache(self, cache_coordinator):
        """Test getting workers with specific cache key."""
        # Setup multiple workers with same cache key
        cache_coordinator.register_worker("worker-0")
        cache_coordinator.register_worker("worker-1")
        cache_coordinator.update_worker_cache_state("worker-0", cached_keys={"key1"})
        cache_coordinator.update_worker_cache_state("worker-1", cached_keys={"key1"})
        
        workers = cache_coordinator.get_workers_with_cache("key1")
        
        assert len(workers) == 2
        assert "worker-0" in workers
        assert "worker-1" in workers
    
    def test_invalidate_cache_key(self, cache_coordinator):
        """Test cache key invalidation."""
        # Setup workers with cache
        cache_coordinator.register_worker("worker-0")
        cache_coordinator.register_worker("worker-1")
        cache_coordinator.update_worker_cache_state("worker-0", cached_keys={"key1"})
        cache_coordinator.update_worker_cache_state("worker-1", cached_keys={"key1"})
        
        # Invalidate
        affected = cache_coordinator.invalidate_cache_key("key1")
        
        assert affected == 2
        assert "key1" not in cache_coordinator._cache_key_locations
        assert "key1" not in cache_coordinator._worker_cache_states["worker-0"].cached_keys
        assert "key1" not in cache_coordinator._worker_cache_states["worker-1"].cached_keys
        assert cache_coordinator.stats["invalidations_broadcast"] == 1
    
    def test_suggest_cache_warming(self, cache_coordinator):
        """Test cache warming suggestions."""
        # Setup: only worker-0 has the cache
        cache_coordinator.register_worker("worker-0")
        cache_coordinator.register_worker("worker-1")
        cache_coordinator.register_worker("worker-2")
        
        cache_coordinator.update_worker_cache_state(
            "worker-0",
            cached_keys={"key1"},
            cache_stats={"total_size_bytes": 1000}
        )
        cache_coordinator.update_worker_cache_state(
            "worker-1",
            cache_stats={"total_size_bytes": 500}  # Less cache, preferred
        )
        cache_coordinator.update_worker_cache_state(
            "worker-2",
            cache_stats={"total_size_bytes": 2000}
        )
        
        # Suggest warming to achieve replication of 3
        suggested = cache_coordinator.suggest_cache_warming("key1", target_replication=3)
        
        assert len(suggested) == 2
        assert "worker-1" in suggested  # Should prefer worker with less cache
        assert cache_coordinator.stats["cache_warming_requests"] == 2
    
    def test_suggest_cache_warming_already_replicated(self, cache_coordinator):
        """Test cache warming when already at target replication."""
        # Setup: all workers have the cache
        for i in range(3):
            worker_id = f"worker-{i}"
            cache_coordinator.register_worker(worker_id)
            cache_coordinator.update_worker_cache_state(worker_id, cached_keys={"key1"})
        
        # Suggest warming
        suggested = cache_coordinator.suggest_cache_warming("key1", target_replication=2)
        
        assert len(suggested) == 0  # Already replicated enough
    
    def test_get_cache_distribution(self, cache_coordinator):
        """Test getting cache distribution statistics."""
        # Setup workers with different cache states
        cache_coordinator.register_worker("worker-0")
        cache_coordinator.register_worker("worker-1")
        
        cache_coordinator.update_worker_cache_state(
            "worker-0",
            cached_keys={"key1", "key2"},
            cache_stats={"total_size_bytes": 1024 * 1024 * 50, "hit_rate": 0.8}
        )
        cache_coordinator.update_worker_cache_state(
            "worker-1",
            cached_keys={"key2", "key3"},
            cache_stats={"total_size_bytes": 1024 * 1024 * 30, "hit_rate": 0.6}
        )
        
        distribution = cache_coordinator.get_cache_distribution()
        
        assert distribution["total_unique_keys"] == 3
        assert distribution["worker_count"] == 2
        assert distribution["average_replication_factor"] == 4 / 3  # 4 total replicas / 3 keys
        assert "worker-0" in distribution["workers"]
        assert distribution["workers"]["worker-0"]["cached_keys"] == 2
        assert distribution["workers"]["worker-0"]["hit_rate"] == 0.8
    
    def test_get_stats(self, cache_coordinator):
        """Test getting coordinator statistics."""
        cache_coordinator.register_worker("worker-0")
        cache_coordinator.update_worker_cache_state("worker-0", cached_keys={"key1"})
        
        # Simulate some routing
        cache_coordinator.report_cache_access("worker-0", "key1", was_hit=True)
        cache_coordinator.report_cache_access("worker-0", "key2", was_hit=False)
        
        stats = cache_coordinator.get_stats()
        
        assert stats["cache_hits_from_routing"] == 1
        assert stats["cache_misses_from_routing"] == 1
        assert stats["cache_routing_hit_rate"] == 0.5
        assert "distribution" in stats
    
    def test_cleanup_stale_state(self, cache_coordinator):
        """Test cleanup of stale cache state."""
        # Register worker with old timestamp
        cache_coordinator.register_worker("worker-0")
        state = cache_coordinator._worker_cache_states["worker-0"]
        state.last_updated = time.time() - 120  # 2 minutes ago
        
        # Cleanup
        cache_coordinator.cleanup_stale_state()
        
        assert "worker-0" not in cache_coordinator._worker_cache_states
    
    def test_multiple_workers_same_cache(self, cache_coordinator, sample_workers):
        """Test routing with multiple workers having same cache."""
        # Setup: multiple workers have the same cache
        for i in range(2):
            worker_id = f"worker-{i}"
            cache_coordinator.register_worker(worker_id)
            cache_coordinator.update_worker_cache_state(worker_id, cached_keys={"key1"})
        
        cache_coordinator.register_worker("worker-2")
        
        # Route multiple times
        decisions = []
        for _ in range(10):
            decision = cache_coordinator.select_worker_with_cache_affinity(
                "key1",
                sample_workers
            )
            decisions.append(decision.worker_id)
        
        # Should only select from workers with cache
        assert all(d in ["worker-0", "worker-1"] for d in decisions)
    
    def test_disabled_cache_routing(self, sample_workers):
        """Test behavior when cache routing is disabled."""
        coordinator = CacheCoordinator(enable_cache_routing=False)
        
        coordinator.register_worker("worker-0")
        coordinator.update_worker_cache_state("worker-0", cached_keys={"key1"})
        
        decision = coordinator.select_worker_with_cache_affinity(
            "key1",
            sample_workers
        )
        
        assert decision is None  # Should not route based on cache


class TestCacheState:
    """Test CacheState dataclass."""
    
    def test_initialization(self):
        """Test cache state initialization."""
        state = CacheState(worker_id="worker-1")
        
        assert state.worker_id == "worker-1"
        assert len(state.cached_keys) == 0
        assert state.cache_size_bytes == 0
        assert state.cache_hit_rate == 0.0
    
    def test_update_from_stats(self):
        """Test updating state from statistics."""
        state = CacheState(worker_id="worker-1")
        
        stats = {
            "total_size_bytes": 1024 * 1024,
            "hit_rate": 0.85
        }
        
        state.update_from_stats(stats)
        
        assert state.cache_size_bytes == 1024 * 1024
        assert state.cache_hit_rate == 0.85


class TestCacheRoutingDecision:
    """Test CacheRoutingDecision dataclass."""
    
    def test_initialization(self):
        """Test routing decision initialization."""
        decision = CacheRoutingDecision(
            worker_id="worker-1",
            has_cache=True,
            cache_affinity_score=0.5,
            reason="Cache hit expected"
        )
        
        assert decision.worker_id == "worker-1"
        assert decision.has_cache is True
        assert decision.cache_affinity_score == 0.5
        assert decision.reason == "Cache hit expected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
