"""
Activation Cache System for MLX Distributed Inference

Implements LRU caching for intermediate activations to reduce redundant computation
and improve inference latency. Supports distributed cache coordination and monitoring.
"""

import time
import hashlib
import threading
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    hit_count: int = 0
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics and metrics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_size_bytes": self.total_size_bytes,
            "entry_count": self.entry_count,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate
        }


class LRUActivationCache:
    """
    LRU cache for intermediate activations with memory management.
    
    Features:
    - Thread-safe LRU eviction policy
    - Memory-based size limits
    - Cache key generation from request parameters
    - Hit/miss metrics and monitoring
    - Automatic eviction on memory pressure
    """
    
    def __init__(
        self,
        max_size_bytes: int = 1024 * 1024 * 1024,  # 1GB default
        max_entries: int = 1000,
        ttl_seconds: Optional[float] = 3600.0  # 1 hour default
    ):
        """
        Initialize activation cache.
        
        Args:
            max_size_bytes: Maximum cache size in bytes
            max_entries: Maximum number of cache entries
            ttl_seconds: Time-to-live for cache entries (None for no expiration)
        """
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        
        # Cache storage (OrderedDict maintains insertion order for LRU)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats()
        
        logger.info(
            f"Initialized LRU activation cache: "
            f"max_size={max_size_bytes / (1024**3):.2f}GB, "
            f"max_entries={max_entries}, ttl={ttl_seconds}s"
        )
    
    def generate_key(
        self,
        model_name: str,
        prompt: str,
        layer_idx: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate cache key from request parameters.
        
        Args:
            model_name: Model identifier
            prompt: Input prompt text
            layer_idx: Optional layer index for layer-specific caching
            **kwargs: Additional parameters to include in key
            
        Returns:
            Cache key string
        """
        # Create deterministic key from parameters
        key_parts = [
            f"model:{model_name}",
            f"prompt:{prompt}",
        ]
        
        if layer_idx is not None:
            key_parts.append(f"layer:{layer_idx}")
        
        # Add sorted kwargs for deterministic key
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        
        key_string = "|".join(key_parts)
        
        # Hash for fixed-length key
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"act_{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self.stats.misses += 1
                logger.debug(f"Cache miss: {key}")
                return None
            
            # Check TTL expiration
            if self.ttl_seconds is not None:
                age = time.time() - entry.created_at
                if age > self.ttl_seconds:
                    logger.debug(f"Cache entry expired: {key} (age={age:.1f}s)")
                    self._remove_entry(key)
                    self.stats.misses += 1
                    return None
            
            # Update access statistics
            entry.update_access()
            entry.hit_count += 1
            self.stats.hits += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            logger.debug(f"Cache hit: {key} (hits={entry.hit_count})")
            return entry.value
    
    def put(self, key: str, value: Any, size_bytes: Optional[int] = None):
        """
        Put value into cache.
        
        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Size of value in bytes (estimated if not provided)
        """
        with self._lock:
            # Estimate size if not provided
            if size_bytes is None:
                size_bytes = self._estimate_size(value)
            
            # Check if adding this entry would exceed size limit
            if size_bytes > self.max_size_bytes:
                logger.warning(
                    f"Entry too large for cache: {size_bytes} bytes "
                    f"(max={self.max_size_bytes})"
                )
                return
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Evict entries until we have space
            while (
                self.stats.total_size_bytes + size_bytes > self.max_size_bytes
                or len(self._cache) >= self.max_entries
            ):
                if not self._cache:
                    break
                self._evict_lru()
            
            # Create and add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_at=time.time(),
                last_accessed=time.time()
            )
            
            self._cache[key] = entry
            self.stats.total_size_bytes += size_bytes
            self.stats.entry_count = len(self._cache)
            
            logger.debug(
                f"Cache put: {key} ({size_bytes} bytes, "
                f"total={self.stats.total_size_bytes / (1024**2):.1f}MB)"
            )
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                logger.debug(f"Cache invalidated: {key}")
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.stats.total_size_bytes = 0
            self.stats.entry_count = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            return self.stats.to_dict()
    
    def get_entries_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all cache entries.
        
        Returns:
            List of entry information dictionaries
        """
        with self._lock:
            entries = []
            for key, entry in self._cache.items():
                entries.append({
                    "key": key,
                    "size_bytes": entry.size_bytes,
                    "age_seconds": time.time() - entry.created_at,
                    "access_count": entry.access_count,
                    "hit_count": entry.hit_count,
                    "last_accessed": entry.last_accessed
                })
            return entries
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Get first (least recently used) entry
        key, entry = self._cache.popitem(last=False)
        self.stats.total_size_bytes -= entry.size_bytes
        self.stats.evictions += 1
        self.stats.entry_count = len(self._cache)
        
        logger.debug(
            f"Cache eviction: {key} ({entry.size_bytes} bytes, "
            f"age={time.time() - entry.created_at:.1f}s, "
            f"hits={entry.hit_count})"
        )
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        entry = self._cache.pop(key, None)
        if entry:
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.entry_count = len(self._cache)
    
    def _estimate_size(self, value: Any) -> int:
        """
        Estimate size of value in bytes.
        
        Args:
            value: Value to estimate size of
            
        Returns:
            Estimated size in bytes
        """
        # Handle MLX arrays
        if isinstance(value, mx.array):
            return value.nbytes
        
        # Handle tuples/lists of MLX arrays
        if isinstance(value, (tuple, list)):
            total_size = 0
            for item in value:
                if isinstance(item, mx.array):
                    total_size += item.nbytes
                else:
                    # Rough estimate for other types
                    total_size += 1024
            return total_size
        
        # Rough estimate for other types
        return 1024



class DistributedCacheCoordinator:
    """
    Coordinator for distributed cache management across workers.
    
    Features:
    - Cache invalidation broadcasting
    - Cache statistics aggregation
    - Cache warming coordination
    - Cache consistency management
    """
    
    def __init__(self):
        """Initialize distributed cache coordinator."""
        self.worker_caches: Dict[str, LRUActivationCache] = {}
        self._lock = threading.Lock()
        
        logger.info("Initialized distributed cache coordinator")
    
    def register_worker_cache(self, worker_id: str, cache: LRUActivationCache):
        """
        Register worker cache for coordination.
        
        Args:
            worker_id: Worker identifier
            cache: Worker's activation cache
        """
        with self._lock:
            self.worker_caches[worker_id] = cache
            logger.info(f"Registered cache for worker: {worker_id}")
    
    def unregister_worker_cache(self, worker_id: str):
        """
        Unregister worker cache.
        
        Args:
            worker_id: Worker identifier
        """
        with self._lock:
            if worker_id in self.worker_caches:
                del self.worker_caches[worker_id]
                logger.info(f"Unregistered cache for worker: {worker_id}")
    
    def invalidate_all(self, key: str):
        """
        Invalidate cache entry across all workers.
        
        Args:
            key: Cache key to invalidate
        """
        with self._lock:
            invalidated_count = 0
            for worker_id, cache in self.worker_caches.items():
                if cache.invalidate(key):
                    invalidated_count += 1
            
            logger.info(
                f"Invalidated cache key across {invalidated_count} workers: {key}"
            )
    
    def clear_all(self):
        """Clear all worker caches."""
        with self._lock:
            for worker_id, cache in self.worker_caches.items():
                cache.clear()
            logger.info(f"Cleared all worker caches ({len(self.worker_caches)} workers)")
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """
        Get aggregated cache statistics across all workers.
        
        Returns:
            Aggregated statistics dictionary
        """
        with self._lock:
            aggregate = {
                "total_hits": 0,
                "total_misses": 0,
                "total_evictions": 0,
                "total_size_bytes": 0,
                "total_entries": 0,
                "worker_count": len(self.worker_caches),
                "workers": {}
            }
            
            for worker_id, cache in self.worker_caches.items():
                stats = cache.get_stats()
                aggregate["total_hits"] += stats["hits"]
                aggregate["total_misses"] += stats["misses"]
                aggregate["total_evictions"] += stats["evictions"]
                aggregate["total_size_bytes"] += stats["total_size_bytes"]
                aggregate["total_entries"] += stats["entry_count"]
                aggregate["workers"][worker_id] = stats
            
            # Calculate aggregate hit rate
            total_requests = aggregate["total_hits"] + aggregate["total_misses"]
            aggregate["aggregate_hit_rate"] = (
                aggregate["total_hits"] / total_requests if total_requests > 0 else 0.0
            )
            
            return aggregate


# Global cache instance (can be configured per worker)
_global_cache: Optional[LRUActivationCache] = None
_cache_lock = threading.Lock()


def get_global_cache() -> LRUActivationCache:
    """
    Get or create global activation cache instance.
    
    Returns:
        Global activation cache
    """
    global _global_cache
    
    with _cache_lock:
        if _global_cache is None:
            _global_cache = LRUActivationCache()
        return _global_cache


def configure_global_cache(
    max_size_bytes: int = 1024 * 1024 * 1024,
    max_entries: int = 1000,
    ttl_seconds: Optional[float] = 3600.0
):
    """
    Configure global activation cache.
    
    Args:
        max_size_bytes: Maximum cache size in bytes
        max_entries: Maximum number of cache entries
        ttl_seconds: Time-to-live for cache entries
    """
    global _global_cache
    
    with _cache_lock:
        _global_cache = LRUActivationCache(
            max_size_bytes=max_size_bytes,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds
        )
        logger.info("Configured global activation cache")
