"""
KV Cache Optimization for MLX Distributed Inference

Implements optimized key-value caching for attention layers with cache sharing,
compression, and warming strategies to improve throughput and reduce memory usage.
"""

import time
import threading
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import logging
import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KVCacheEntry:
    """KV cache entry for attention layers."""
    
    key_cache: mx.array
    value_cache: mx.array
    sequence_length: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    shared_count: int = 0  # Number of requests sharing this cache
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    @property
    def size_bytes(self) -> int:
        """Calculate total size in bytes."""
        return self.key_cache.nbytes + self.value_cache.nbytes
    
    @property
    def age_seconds(self) -> float:
        """Calculate age in seconds."""
        return time.time() - self.created_at


class KVCache:
    """
    Optimized KV cache for attention layers.
    
    Features:
    - Efficient storage of key-value pairs
    - Cache sharing across similar requests
    - Automatic cache extension for continued generation
    - Memory-efficient storage with compression
    - Cache warming and preloading
    """
    
    def __init__(
        self,
        max_cache_size_bytes: int = 2 * 1024 * 1024 * 1024,  # 2GB default
        max_sequence_length: int = 4096,
        enable_compression: bool = False,
        compression_threshold: int = 1024  # Compress sequences longer than this
    ):
        """
        Initialize KV cache.
        
        Args:
            max_cache_size_bytes: Maximum cache size in bytes
            max_sequence_length: Maximum sequence length to cache
            enable_compression: Enable cache compression
            compression_threshold: Minimum sequence length for compression
        """
        self.max_cache_size_bytes = max_cache_size_bytes
        self.max_sequence_length = max_sequence_length
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        
        # Cache storage
        self._cache: Dict[str, KVCacheEntry] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "extensions": 0,
            "evictions": 0,
            "total_size_bytes": 0,
            "entry_count": 0,
            "shared_accesses": 0
        }
        
        logger.info(
            f"Initialized KV cache: "
            f"max_size={max_cache_size_bytes / (1024**3):.2f}GB, "
            f"max_seq_len={max_sequence_length}, "
            f"compression={'enabled' if enable_compression else 'disabled'}"
        )
    
    def get(
        self,
        cache_key: str,
        required_length: Optional[int] = None
    ) -> Optional[Tuple[mx.array, mx.array]]:
        """
        Get KV cache entry.
        
        Args:
            cache_key: Cache key
            required_length: Required sequence length (None for any length)
            
        Returns:
            Tuple of (key_cache, value_cache) or None if not found
        """
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                self.stats["misses"] += 1
                logger.debug(f"KV cache miss: {cache_key}")
                return None
            
            # Check if cached sequence is long enough
            if required_length is not None and entry.sequence_length < required_length:
                self.stats["misses"] += 1
                logger.debug(
                    f"KV cache miss (insufficient length): {cache_key} "
                    f"(cached={entry.sequence_length}, required={required_length})"
                )
                return None
            
            # Update access statistics
            entry.update_access()
            self.stats["hits"] += 1
            
            logger.debug(
                f"KV cache hit: {cache_key} "
                f"(seq_len={entry.sequence_length}, hits={entry.access_count})"
            )
            
            return (entry.key_cache, entry.value_cache)
    
    def put(
        self,
        cache_key: str,
        key_cache: mx.array,
        value_cache: mx.array,
        sequence_length: int
    ):
        """
        Put KV cache entry.
        
        Args:
            cache_key: Cache key
            key_cache: Key cache tensor
            value_cache: Value cache tensor
            sequence_length: Sequence length
        """
        with self._lock:
            # Check sequence length limit
            if sequence_length > self.max_sequence_length:
                logger.warning(
                    f"Sequence too long for KV cache: {sequence_length} "
                    f"(max={self.max_sequence_length})"
                )
                return
            
            # Calculate entry size
            entry_size = key_cache.nbytes + value_cache.nbytes
            
            # Check if entry would exceed cache size
            if entry_size > self.max_cache_size_bytes:
                logger.warning(
                    f"KV cache entry too large: {entry_size / (1024**2):.1f}MB "
                    f"(max={self.max_cache_size_bytes / (1024**2):.1f}MB)"
                )
                return
            
            # Remove existing entry if present
            if cache_key in self._cache:
                old_entry = self._cache[cache_key]
                self.stats["total_size_bytes"] -= old_entry.size_bytes
            
            # Evict entries if needed
            while self.stats["total_size_bytes"] + entry_size > self.max_cache_size_bytes:
                if not self._cache:
                    break
                self._evict_lru()
            
            # Apply compression if enabled
            if self.enable_compression and sequence_length >= self.compression_threshold:
                key_cache, value_cache = self._compress_cache(key_cache, value_cache)
            
            # Create and store entry
            entry = KVCacheEntry(
                key_cache=key_cache,
                value_cache=value_cache,
                sequence_length=sequence_length,
                created_at=time.time(),
                last_accessed=time.time()
            )
            
            self._cache[cache_key] = entry
            self.stats["total_size_bytes"] += entry.size_bytes
            self.stats["entry_count"] = len(self._cache)
            
            logger.debug(
                f"KV cache put: {cache_key} "
                f"(seq_len={sequence_length}, size={entry.size_bytes / (1024**2):.1f}MB)"
            )
    
    def extend(
        self,
        cache_key: str,
        new_key_cache: mx.array,
        new_value_cache: mx.array,
        new_sequence_length: int
    ) -> bool:
        """
        Extend existing KV cache entry with new tokens.
        
        Args:
            cache_key: Cache key
            new_key_cache: New key cache to append
            new_value_cache: New value cache to append
            new_sequence_length: New total sequence length
            
        Returns:
            True if extension was successful, False otherwise
        """
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                logger.debug(f"Cannot extend non-existent cache: {cache_key}")
                return False
            
            # Check sequence length limit
            if new_sequence_length > self.max_sequence_length:
                logger.warning(
                    f"Extended sequence too long: {new_sequence_length} "
                    f"(max={self.max_sequence_length})"
                )
                return False
            
            # Concatenate caches
            try:
                extended_key = mx.concatenate([entry.key_cache, new_key_cache], axis=-2)
                extended_value = mx.concatenate([entry.value_cache, new_value_cache], axis=-2)
            except Exception as e:
                logger.error(f"Failed to extend KV cache: {e}")
                return False
            
            # Update entry
            old_size = entry.size_bytes
            entry.key_cache = extended_key
            entry.value_cache = extended_value
            entry.sequence_length = new_sequence_length
            entry.update_access()
            
            # Update statistics
            new_size = entry.size_bytes
            self.stats["total_size_bytes"] += (new_size - old_size)
            self.stats["extensions"] += 1
            
            logger.debug(
                f"KV cache extended: {cache_key} "
                f"(new_seq_len={new_sequence_length}, size_delta={(new_size - old_size) / (1024**2):.1f}MB)"
            )
            
            return True
    
    def share(self, cache_key: str) -> bool:
        """
        Mark cache entry as shared by another request.
        
        Args:
            cache_key: Cache key
            
        Returns:
            True if entry exists and was marked as shared
        """
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                return False
            
            entry.shared_count += 1
            self.stats["shared_accesses"] += 1
            
            logger.debug(
                f"KV cache shared: {cache_key} (shared_count={entry.shared_count})"
            )
            
            return True
    
    def unshare(self, cache_key: str) -> bool:
        """
        Unmark cache entry as shared.
        
        Args:
            cache_key: Cache key
            
        Returns:
            True if entry exists and was unmarked
        """
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                return False
            
            entry.shared_count = max(0, entry.shared_count - 1)
            
            return True
    
    def invalidate(self, cache_key: str) -> bool:
        """
        Invalidate cache entry.
        
        Args:
            cache_key: Cache key
            
        Returns:
            True if entry was removed, False if not found
        """
        with self._lock:
            entry = self._cache.pop(cache_key, None)
            
            if entry is None:
                return False
            
            self.stats["total_size_bytes"] -= entry.size_bytes
            self.stats["entry_count"] = len(self._cache)
            
            logger.debug(f"KV cache invalidated: {cache_key}")
            
            return True
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.stats["total_size_bytes"] = 0
            self.stats["entry_count"] = 0
            logger.info("KV cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            stats = self.stats.copy()
            
            # Calculate hit rate
            total_requests = stats["hits"] + stats["misses"]
            stats["hit_rate"] = stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            # Calculate average entry size
            stats["avg_entry_size_mb"] = (
                (stats["total_size_bytes"] / stats["entry_count"] / (1024**2))
                if stats["entry_count"] > 0 else 0.0
            )
            
            return stats
    
    def get_entries_info(self) -> List[Dict[str, Any]]:
        """
        Get information about cache entries.
        
        Returns:
            List of entry information dictionaries
        """
        with self._lock:
            entries = []
            for key, entry in self._cache.items():
                entries.append({
                    "key": key,
                    "sequence_length": entry.sequence_length,
                    "size_bytes": entry.size_bytes,
                    "age_seconds": entry.age_seconds,
                    "access_count": entry.access_count,
                    "shared_count": entry.shared_count,
                    "last_accessed": entry.last_accessed
                })
            return entries
    
    def warm_cache(
        self,
        cache_key: str,
        key_cache: mx.array,
        value_cache: mx.array,
        sequence_length: int
    ):
        """
        Warm cache with precomputed KV pairs.
        
        Args:
            cache_key: Cache key
            key_cache: Precomputed key cache
            value_cache: Precomputed value cache
            sequence_length: Sequence length
        """
        logger.info(f"Warming KV cache: {cache_key} (seq_len={sequence_length})")
        self.put(cache_key, key_cache, value_cache, sequence_length)
    
    def _evict_lru(self):
        """Evict least recently used entry that is not shared."""
        if not self._cache:
            return
        
        # Find LRU entry that is not currently shared
        lru_key = None
        lru_time = float('inf')
        
        for key, entry in self._cache.items():
            if entry.shared_count == 0 and entry.last_accessed < lru_time:
                lru_key = key
                lru_time = entry.last_accessed
        
        # If all entries are shared, evict oldest shared entry
        if lru_key is None:
            lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        
        # Remove entry
        entry = self._cache.pop(lru_key)
        self.stats["total_size_bytes"] -= entry.size_bytes
        self.stats["evictions"] += 1
        self.stats["entry_count"] = len(self._cache)
        
        logger.debug(
            f"KV cache eviction: {lru_key} "
            f"(seq_len={entry.sequence_length}, age={entry.age_seconds:.1f}s, "
            f"shared={entry.shared_count})"
        )
    
    def _compress_cache(
        self,
        key_cache: mx.array,
        value_cache: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Compress KV cache using quantization.
        
        Args:
            key_cache: Key cache to compress
            value_cache: Value cache to compress
            
        Returns:
            Tuple of compressed (key_cache, value_cache)
        """
        # Simple quantization to float16 for compression
        # More sophisticated compression could be implemented
        try:
            compressed_key = key_cache.astype(mx.float16)
            compressed_value = value_cache.astype(mx.float16)
            
            logger.debug(
                f"KV cache compressed: "
                f"original={key_cache.nbytes + value_cache.nbytes}, "
                f"compressed={compressed_key.nbytes + compressed_value.nbytes}"
            )
            
            return compressed_key, compressed_value
        except Exception as e:
            logger.warning(f"KV cache compression failed: {e}")
            return key_cache, value_cache


class SharedKVCacheManager:
    """
    Manager for sharing KV caches across multiple requests.
    
    Features:
    - Automatic cache sharing for similar prompts
    - Reference counting for shared caches
    - Cache lifecycle management
    """
    
    def __init__(self, kv_cache: KVCache):
        """
        Initialize shared KV cache manager.
        
        Args:
            kv_cache: Underlying KV cache instance
        """
        self.kv_cache = kv_cache
        self._shared_refs: Dict[str, int] = {}
        self._lock = threading.Lock()
        
        logger.info("Initialized shared KV cache manager")
    
    def acquire(self, cache_key: str) -> Optional[Tuple[mx.array, mx.array]]:
        """
        Acquire shared KV cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Tuple of (key_cache, value_cache) or None if not found
        """
        with self._lock:
            result = self.kv_cache.get(cache_key)
            
            if result is not None:
                # Increment reference count
                self._shared_refs[cache_key] = self._shared_refs.get(cache_key, 0) + 1
                self.kv_cache.share(cache_key)
                
                logger.debug(
                    f"Acquired shared KV cache: {cache_key} "
                    f"(refs={self._shared_refs[cache_key]})"
                )
            
            return result
    
    def release(self, cache_key: str):
        """
        Release shared KV cache.
        
        Args:
            cache_key: Cache key
        """
        with self._lock:
            if cache_key in self._shared_refs:
                self._shared_refs[cache_key] -= 1
                self.kv_cache.unshare(cache_key)
                
                # Remove from tracking if no more references
                if self._shared_refs[cache_key] <= 0:
                    del self._shared_refs[cache_key]
                
                logger.debug(
                    f"Released shared KV cache: {cache_key} "
                    f"(refs={self._shared_refs.get(cache_key, 0)})"
                )
    
    def get_shared_count(self, cache_key: str) -> int:
        """
        Get number of active references to cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Number of active references
        """
        with self._lock:
            return self._shared_refs.get(cache_key, 0)


# Global KV cache instance
_global_kv_cache: Optional[KVCache] = None
_kv_cache_lock = threading.Lock()


def get_global_kv_cache() -> KVCache:
    """
    Get or create global KV cache instance.
    
    Returns:
        Global KV cache
    """
    global _global_kv_cache
    
    with _kv_cache_lock:
        if _global_kv_cache is None:
            _global_kv_cache = KVCache()
        return _global_kv_cache


def configure_global_kv_cache(
    max_cache_size_bytes: int = 2 * 1024 * 1024 * 1024,
    max_sequence_length: int = 4096,
    enable_compression: bool = False
):
    """
    Configure global KV cache.
    
    Args:
        max_cache_size_bytes: Maximum cache size in bytes
        max_sequence_length: Maximum sequence length
        enable_compression: Enable cache compression
    """
    global _global_kv_cache
    
    with _kv_cache_lock:
        _global_kv_cache = KVCache(
            max_cache_size_bytes=max_cache_size_bytes,
            max_sequence_length=max_sequence_length,
            enable_compression=enable_compression
        )
        logger.info("Configured global KV cache")
