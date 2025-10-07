"""
Distributed Cache Coordination for MLX Distributed Inference

Coordinates caching across workers to improve cache hit rates and load distribution.
Implements cache-aware routing, cache state tracking, and cache invalidation.
"""

import time
import asyncio
import threading
import logging
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .models import WorkerInfo

logger = logging.getLogger(__name__)


@dataclass
class CacheState:
    """Cache state for a worker."""
    
    worker_id: str
    cached_keys: Set[str] = field(default_factory=set)
    cache_size_bytes: int = 0
    cache_hit_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    def update_from_stats(self, stats: Dict[str, Any]):
        """Update cache state from worker statistics."""
        self.cache_size_bytes = stats.get("total_size_bytes", 0)
        self.cache_hit_rate = stats.get("hit_rate", 0.0)
        self.last_updated = time.time()


@dataclass
class CacheRoutingDecision:
    """Decision for cache-aware routing."""
    
    worker_id: str
    has_cache: bool
    cache_affinity_score: float
    reason: str


class CacheCoordinator:
    """
    Coordinates caching across distributed workers.
    
    Features:
    - Track cache state across workers
    - Cache-aware request routing
    - Cache invalidation broadcasting
    - Cache warming coordination
    - Cache replication strategies
    """
    
    def __init__(
        self,
        enable_cache_routing: bool = True,
        cache_affinity_weight: float = 0.3,
        cache_state_ttl: float = 60.0
    ):
        """
        Initialize cache coordinator.
        
        Args:
            enable_cache_routing: Enable cache-aware routing
            cache_affinity_weight: Weight for cache affinity in routing (0.0-1.0)
            cache_state_ttl: TTL for cache state information in seconds
        """
        self.enable_cache_routing = enable_cache_routing
        self.cache_affinity_weight = cache_affinity_weight
        self.cache_state_ttl = cache_state_ttl
        
        # Cache state tracking
        self._worker_cache_states: Dict[str, CacheState] = {}
        self._cache_key_locations: Dict[str, Set[str]] = defaultdict(set)  # key -> worker_ids
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "cache_aware_routes": 0,
            "cache_hits_from_routing": 0,
            "cache_misses_from_routing": 0,
            "invalidations_broadcast": 0,
            "cache_warming_requests": 0
        }
        
        logger.info(
            f"Initialized CacheCoordinator: "
            f"cache_routing={'enabled' if enable_cache_routing else 'disabled'}, "
            f"affinity_weight={cache_affinity_weight}"
        )
    
    def register_worker(self, worker_id: str):
        """
        Register a worker for cache coordination.
        
        Args:
            worker_id: Worker identifier
        """
        with self._lock:
            if worker_id not in self._worker_cache_states:
                self._worker_cache_states[worker_id] = CacheState(worker_id=worker_id)
                logger.info(f"Registered worker for cache coordination: {worker_id}")
    
    def unregister_worker(self, worker_id: str):
        """
        Unregister a worker from cache coordination.
        
        Args:
            worker_id: Worker identifier
        """
        with self._lock:
            if worker_id in self._worker_cache_states:
                # Remove worker from cache key locations
                cache_state = self._worker_cache_states[worker_id]
                for cache_key in cache_state.cached_keys:
                    if cache_key in self._cache_key_locations:
                        self._cache_key_locations[cache_key].discard(worker_id)
                        if not self._cache_key_locations[cache_key]:
                            del self._cache_key_locations[cache_key]
                
                del self._worker_cache_states[worker_id]
                logger.info(f"Unregistered worker from cache coordination: {worker_id}")
    
    def update_worker_cache_state(
        self,
        worker_id: str,
        cached_keys: Optional[Set[str]] = None,
        cache_stats: Optional[Dict[str, Any]] = None
    ):
        """
        Update cache state for a worker.
        
        Args:
            worker_id: Worker identifier
            cached_keys: Set of cache keys present on worker
            cache_stats: Cache statistics from worker
        """
        with self._lock:
            if worker_id not in self._worker_cache_states:
                self.register_worker(worker_id)
            
            cache_state = self._worker_cache_states[worker_id]
            
            # Update cached keys if provided
            if cached_keys is not None:
                # Remove old key locations
                for old_key in cache_state.cached_keys:
                    if old_key in self._cache_key_locations:
                        self._cache_key_locations[old_key].discard(worker_id)
                
                # Add new key locations
                cache_state.cached_keys = cached_keys.copy()
                for key in cached_keys:
                    self._cache_key_locations[key].add(worker_id)
            
            # Update cache statistics
            if cache_stats is not None:
                cache_state.update_from_stats(cache_stats)
            
            logger.debug(
                f"Updated cache state for {worker_id}: "
                f"{len(cache_state.cached_keys)} keys, "
                f"{cache_state.cache_size_bytes / (1024**2):.1f}MB"
            )
    
    def report_cache_access(
        self,
        worker_id: str,
        cache_key: str,
        was_hit: bool
    ):
        """
        Report cache access result from a worker.
        
        Args:
            worker_id: Worker identifier
            cache_key: Cache key accessed
            was_hit: Whether the access was a cache hit
        """
        with self._lock:
            if was_hit:
                # Add key to worker's cached keys
                if worker_id in self._worker_cache_states:
                    self._worker_cache_states[worker_id].cached_keys.add(cache_key)
                    self._cache_key_locations[cache_key].add(worker_id)
                
                self.stats["cache_hits_from_routing"] += 1
            else:
                self.stats["cache_misses_from_routing"] += 1
    
    def select_worker_with_cache_affinity(
        self,
        cache_key: str,
        available_workers: List[WorkerInfo],
        base_scores: Optional[Dict[str, float]] = None
    ) -> Optional[CacheRoutingDecision]:
        """
        Select worker considering cache affinity.
        
        Args:
            cache_key: Cache key for the request
            available_workers: List of available workers
            base_scores: Optional base scores for workers (lower is better)
            
        Returns:
            CacheRoutingDecision with selected worker and reasoning
        """
        if not self.enable_cache_routing or not available_workers:
            return None
        
        with self._lock:
            # Find workers that have this cache key
            workers_with_cache = self._cache_key_locations.get(cache_key, set())
            
            # Filter to available workers
            available_worker_ids = {w.worker_id for w in available_workers}
            workers_with_cache = workers_with_cache.intersection(available_worker_ids)
            
            # If no workers have the cache, return None (use normal routing)
            if not workers_with_cache:
                logger.debug(f"No workers have cache for key: {cache_key}")
                return None
            
            # Calculate cache affinity scores
            scored_workers = []
            
            for worker in available_workers:
                has_cache = worker.worker_id in workers_with_cache
                
                # Base score (0.0 if not provided)
                base_score = base_scores.get(worker.worker_id, 0.0) if base_scores else 0.0
                
                # Cache affinity bonus (negative to prefer cached workers)
                cache_bonus = -self.cache_affinity_weight if has_cache else 0.0
                
                # Combined score
                final_score = base_score + cache_bonus
                
                scored_workers.append((worker, has_cache, final_score))
            
            # Sort by score (ascending - lower is better)
            scored_workers.sort(key=lambda x: x[2])
            
            # Select best worker
            best_worker, has_cache, score = scored_workers[0]
            
            self.stats["cache_aware_routes"] += 1
            
            reason = (
                f"Cache hit expected (affinity={self.cache_affinity_weight})"
                if has_cache
                else "Best worker without cache"
            )
            
            decision = CacheRoutingDecision(
                worker_id=best_worker.worker_id,
                has_cache=has_cache,
                cache_affinity_score=score,
                reason=reason
            )
            
            logger.debug(
                f"Cache-aware routing: selected {best_worker.worker_id} "
                f"(has_cache={has_cache}, score={score:.3f})"
            )
            
            return decision
    
    def get_workers_with_cache(self, cache_key: str) -> List[str]:
        """
        Get list of workers that have a specific cache key.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            List of worker IDs that have the cache key
        """
        with self._lock:
            return list(self._cache_key_locations.get(cache_key, set()))
    
    def invalidate_cache_key(self, cache_key: str) -> int:
        """
        Invalidate a cache key across all workers.
        
        Note: This only updates the coordinator's tracking.
        Actual invalidation must be sent to workers separately.
        
        Args:
            cache_key: Cache key to invalidate
            
        Returns:
            Number of workers affected
        """
        with self._lock:
            workers_affected = self._cache_key_locations.get(cache_key, set()).copy()
            
            # Remove from worker cache states
            for worker_id in workers_affected:
                if worker_id in self._worker_cache_states:
                    self._worker_cache_states[worker_id].cached_keys.discard(cache_key)
            
            # Remove from cache key locations
            if cache_key in self._cache_key_locations:
                del self._cache_key_locations[cache_key]
            
            self.stats["invalidations_broadcast"] += 1
            
            logger.info(
                f"Invalidated cache key across {len(workers_affected)} workers: {cache_key}"
            )
            
            return len(workers_affected)
    
    def suggest_cache_warming(
        self,
        cache_key: str,
        target_replication: int = 2
    ) -> List[str]:
        """
        Suggest workers for cache warming to achieve target replication.
        
        Args:
            cache_key: Cache key to warm
            target_replication: Target number of workers with this cache
            
        Returns:
            List of worker IDs that should warm this cache
        """
        with self._lock:
            current_workers = self._cache_key_locations.get(cache_key, set())
            current_count = len(current_workers)
            
            if current_count >= target_replication:
                return []
            
            # Find workers that don't have this cache
            all_workers = set(self._worker_cache_states.keys())
            workers_without_cache = all_workers - current_workers
            
            # Select workers with lowest cache size (most available space)
            worker_sizes = [
                (worker_id, self._worker_cache_states[worker_id].cache_size_bytes)
                for worker_id in workers_without_cache
            ]
            worker_sizes.sort(key=lambda x: x[1])
            
            # Suggest workers to warm
            needed = target_replication - current_count
            suggested = [worker_id for worker_id, _ in worker_sizes[:needed]]
            
            self.stats["cache_warming_requests"] += len(suggested)
            
            logger.info(
                f"Suggested {len(suggested)} workers for cache warming: {cache_key}"
            )
            
            return suggested
    
    def get_cache_distribution(self) -> Dict[str, Any]:
        """
        Get cache distribution statistics across workers.
        
        Returns:
            Dictionary with cache distribution information
        """
        with self._lock:
            total_keys = len(self._cache_key_locations)
            total_cache_size = sum(
                state.cache_size_bytes
                for state in self._worker_cache_states.values()
            )
            
            # Calculate replication factor
            total_replicas = sum(
                len(workers) for workers in self._cache_key_locations.values()
            )
            avg_replication = total_replicas / max(total_keys, 1)
            
            # Worker-level statistics
            worker_stats = {}
            for worker_id, state in self._worker_cache_states.items():
                worker_stats[worker_id] = {
                    "cached_keys": len(state.cached_keys),
                    "cache_size_mb": state.cache_size_bytes / (1024**2),
                    "hit_rate": state.cache_hit_rate,
                    "age_seconds": time.time() - state.last_updated
                }
            
            return {
                "total_unique_keys": total_keys,
                "total_cache_size_mb": total_cache_size / (1024**2),
                "average_replication_factor": avg_replication,
                "worker_count": len(self._worker_cache_states),
                "workers": worker_stats
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache coordination statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            stats = self.stats.copy()
            
            # Calculate cache-aware routing effectiveness
            total_cache_routes = (
                stats["cache_hits_from_routing"] + stats["cache_misses_from_routing"]
            )
            stats["cache_routing_hit_rate"] = (
                stats["cache_hits_from_routing"] / max(total_cache_routes, 1)
            )
            
            # Add distribution stats
            stats["distribution"] = self.get_cache_distribution()
            
            return stats
    
    def cleanup_stale_state(self):
        """Clean up stale cache state information."""
        with self._lock:
            now = time.time()
            stale_workers = []
            
            for worker_id, state in self._worker_cache_states.items():
                if now - state.last_updated > self.cache_state_ttl:
                    stale_workers.append(worker_id)
            
            for worker_id in stale_workers:
                logger.warning(f"Removing stale cache state for worker: {worker_id}")
                self.unregister_worker(worker_id)
            
            if stale_workers:
                logger.info(f"Cleaned up {len(stale_workers)} stale cache states")
