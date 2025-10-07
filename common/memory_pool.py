"""
Memory Pooling for MLX Distributed Inference

Implements tensor memory pools for efficient reuse, reducing allocation overhead
and memory fragmentation. Provides memory pressure detection and response.
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import mlx.core as mx

logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PooledTensor:
    """Pooled tensor with metadata."""
    
    tensor: mx.array
    shape: Tuple[int, ...]
    dtype: mx.Dtype
    size_bytes: int
    pool_key: str
    allocated_at: float
    last_used: float
    use_count: int = 0
    in_use: bool = False
    
    def mark_used(self):
        """Mark tensor as in use."""
        self.in_use = True
        self.last_used = time.time()
        self.use_count += 1
    
    def mark_free(self):
        """Mark tensor as free."""
        self.in_use = False
        self.last_used = time.time()
    
    @property
    def age_seconds(self) -> float:
        """Get age since allocation."""
        return time.time() - self.allocated_at
    
    @property
    def idle_seconds(self) -> float:
        """Get time since last use."""
        return time.time() - self.last_used


@dataclass
class MemoryPoolStats:
    """Memory pool statistics."""
    
    total_allocated_bytes: int = 0
    total_in_use_bytes: int = 0
    total_free_bytes: int = 0
    total_tensors: int = 0
    in_use_tensors: int = 0
    free_tensors: int = 0
    allocations: int = 0
    deallocations: int = 0
    reuses: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def utilization_rate(self) -> float:
        """Calculate memory utilization rate."""
        if self.total_allocated_bytes == 0:
            return 0.0
        return self.total_in_use_bytes / self.total_allocated_bytes
    
    @property
    def reuse_rate(self) -> float:
        """Calculate tensor reuse rate."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_allocated_mb": self.total_allocated_bytes / (1024**2),
            "total_in_use_mb": self.total_in_use_bytes / (1024**2),
            "total_free_mb": self.total_free_bytes / (1024**2),
            "total_tensors": self.total_tensors,
            "in_use_tensors": self.in_use_tensors,
            "free_tensors": self.free_tensors,
            "allocations": self.allocations,
            "deallocations": self.deallocations,
            "reuses": self.reuses,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "utilization_rate": self.utilization_rate,
            "reuse_rate": self.reuse_rate
        }


class TensorMemoryPool:
    """
    Memory pool for tensor reuse.
    
    Features:
    - Shape and dtype-based pooling
    - Automatic tensor reuse
    - Memory pressure detection
    - Fragmentation monitoring
    - Idle tensor cleanup
    """
    
    def __init__(
        self,
        max_pool_size_bytes: int = 2 * 1024 * 1024 * 1024,  # 2GB default
        max_idle_seconds: float = 300.0,  # 5 minutes
        cleanup_interval_seconds: float = 60.0,  # 1 minute
        enable_auto_cleanup: bool = True
    ):
        """
        Initialize tensor memory pool.
        
        Args:
            max_pool_size_bytes: Maximum pool size in bytes
            max_idle_seconds: Maximum idle time before cleanup
            cleanup_interval_seconds: Interval for automatic cleanup
            enable_auto_cleanup: Enable automatic cleanup of idle tensors
        """
        self.max_pool_size_bytes = max_pool_size_bytes
        self.max_idle_seconds = max_idle_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.enable_auto_cleanup = enable_auto_cleanup
        
        # Pool storage: pool_key -> list of PooledTensor
        self._pools: Dict[str, List[PooledTensor]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = MemoryPoolStats()
        
        # Cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        if enable_auto_cleanup:
            self._start_cleanup_thread()
        
        logger.info(
            f"Initialized tensor memory pool: "
            f"max_size={max_pool_size_bytes / (1024**3):.2f}GB, "
            f"max_idle={max_idle_seconds}s, "
            f"auto_cleanup={'enabled' if enable_auto_cleanup else 'disabled'}"
        )
    
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: mx.Dtype = mx.float32
    ) -> mx.array:
        """
        Allocate tensor from pool or create new one.
        
        Args:
            shape: Tensor shape
            dtype: Tensor dtype
            
        Returns:
            Allocated tensor
        """
        pool_key = self._generate_pool_key(shape, dtype)
        
        with self._lock:
            # Try to reuse from pool
            pool = self._pools.get(pool_key, [])
            
            for pooled in pool:
                if not pooled.in_use:
                    # Reuse tensor
                    pooled.mark_used()
                    self.stats.cache_hits += 1
                    self.stats.reuses += 1
                    self.stats.in_use_tensors += 1
                    self.stats.free_tensors -= 1
                    self.stats.total_in_use_bytes += pooled.size_bytes
                    self.stats.total_free_bytes -= pooled.size_bytes
                    
                    logger.debug(
                        f"Reused tensor from pool: {shape} {dtype} "
                        f"(use_count={pooled.use_count})"
                    )
                    
                    return pooled.tensor
            
            # No free tensor available, allocate new one
            self.stats.cache_misses += 1
            
            # Check if we have space
            tensor_size = self._estimate_tensor_size(shape, dtype)
            if self.stats.total_allocated_bytes + tensor_size > self.max_pool_size_bytes:
                # Try to free up space
                self._cleanup_idle_tensors()
                
                # Check again
                if self.stats.total_allocated_bytes + tensor_size > self.max_pool_size_bytes:
                    logger.warning(
                        f"Memory pool full, allocating outside pool: "
                        f"{tensor_size / (1024**2):.1f}MB"
                    )
                    # Allocate outside pool
                    return mx.zeros(shape, dtype=dtype)
            
            # Allocate new tensor
            tensor = mx.zeros(shape, dtype=dtype)
            
            pooled = PooledTensor(
                tensor=tensor,
                shape=shape,
                dtype=dtype,
                size_bytes=tensor_size,
                pool_key=pool_key,
                allocated_at=time.time(),
                last_used=time.time()
            )
            pooled.mark_used()
            
            self._pools[pool_key].append(pooled)
            
            # Update statistics
            self.stats.allocations += 1
            self.stats.total_tensors += 1
            self.stats.in_use_tensors += 1
            self.stats.total_allocated_bytes += tensor_size
            self.stats.total_in_use_bytes += tensor_size
            
            logger.debug(
                f"Allocated new tensor: {shape} {dtype} "
                f"({tensor_size / (1024**2):.1f}MB)"
            )
            
            return tensor
    
    def release(self, tensor: mx.array):
        """
        Release tensor back to pool.
        
        Args:
            tensor: Tensor to release
        """
        with self._lock:
            # Find tensor in pools
            for pool_key, pool in self._pools.items():
                for pooled in pool:
                    if pooled.tensor is tensor and pooled.in_use:
                        pooled.mark_free()
                        self.stats.in_use_tensors -= 1
                        self.stats.free_tensors += 1
                        self.stats.total_in_use_bytes -= pooled.size_bytes
                        self.stats.total_free_bytes += pooled.size_bytes
                        
                        logger.debug(
                            f"Released tensor to pool: {pooled.shape} {pooled.dtype}"
                        )
                        return
            
            logger.debug("Tensor not found in pool (allocated outside pool)")
    
    def clear(self):
        """Clear all tensors from pool."""
        with self._lock:
            self._pools.clear()
            self.stats = MemoryPoolStats()
            logger.info("Memory pool cleared")
    
    def get_memory_pressure(self) -> MemoryPressureLevel:
        """
        Get current memory pressure level.
        
        Returns:
            Memory pressure level
        """
        with self._lock:
            utilization = self.stats.utilization_rate
            
            if utilization >= 0.95:
                return MemoryPressureLevel.CRITICAL
            elif utilization >= 0.85:
                return MemoryPressureLevel.HIGH
            elif utilization >= 0.70:
                return MemoryPressureLevel.MEDIUM
            else:
                return MemoryPressureLevel.LOW
    
    def get_fragmentation_ratio(self) -> float:
        """
        Calculate memory fragmentation ratio.
        
        Returns:
            Fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
        """
        with self._lock:
            if not self._pools:
                return 0.0
            
            # Count unique pool keys vs total tensors
            unique_pools = len(self._pools)
            total_tensors = self.stats.total_tensors
            
            if total_tensors == 0:
                return 0.0
            
            # More pools relative to tensors = more fragmentation
            fragmentation = min(unique_pools / max(total_tensors, 1), 1.0)
            
            return fragmentation
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory pool statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            stats = self.stats.to_dict()
            stats["memory_pressure"] = self.get_memory_pressure().value
            stats["fragmentation_ratio"] = self.get_fragmentation_ratio()
            stats["pool_count"] = len(self._pools)
            return stats
    
    def get_pool_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all pools.
        
        Returns:
            List of pool information dictionaries
        """
        with self._lock:
            pools_info = []
            
            for pool_key, pool in self._pools.items():
                in_use = sum(1 for p in pool if p.in_use)
                free = len(pool) - in_use
                total_size = sum(p.size_bytes for p in pool)
                
                pools_info.append({
                    "pool_key": pool_key,
                    "total_tensors": len(pool),
                    "in_use": in_use,
                    "free": free,
                    "total_size_mb": total_size / (1024**2),
                    "avg_use_count": sum(p.use_count for p in pool) / len(pool) if pool else 0
                })
            
            return pools_info
    
    def cleanup_idle_tensors(self, max_idle_seconds: Optional[float] = None):
        """
        Clean up idle tensors.
        
        Args:
            max_idle_seconds: Maximum idle time (uses default if None)
        """
        if max_idle_seconds is None:
            max_idle_seconds = self.max_idle_seconds
        
        self._cleanup_idle_tensors(max_idle_seconds)
    
    def _cleanup_idle_tensors(self, max_idle_seconds: Optional[float] = None):
        """Internal cleanup implementation."""
        if max_idle_seconds is None:
            max_idle_seconds = self.max_idle_seconds
        
        with self._lock:
            removed_count = 0
            removed_bytes = 0
            
            for pool_key, pool in list(self._pools.items()):
                # Remove idle tensors
                new_pool = []
                for pooled in pool:
                    if not pooled.in_use and pooled.idle_seconds > max_idle_seconds:
                        # Remove tensor
                        removed_count += 1
                        removed_bytes += pooled.size_bytes
                        self.stats.deallocations += 1
                        self.stats.total_tensors -= 1
                        self.stats.free_tensors -= 1
                        self.stats.total_allocated_bytes -= pooled.size_bytes
                        self.stats.total_free_bytes -= pooled.size_bytes
                    else:
                        new_pool.append(pooled)
                
                if new_pool:
                    self._pools[pool_key] = new_pool
                else:
                    del self._pools[pool_key]
            
            if removed_count > 0:
                logger.info(
                    f"Cleaned up {removed_count} idle tensors "
                    f"({removed_bytes / (1024**2):.1f}MB)"
                )
    
    def _start_cleanup_thread(self):
        """Start automatic cleanup thread."""
        def cleanup_loop():
            while not self._stop_cleanup.is_set():
                time.sleep(self.cleanup_interval_seconds)
                if not self._stop_cleanup.is_set():
                    self._cleanup_idle_tensors()
        
        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name="MemoryPoolCleanup"
        )
        self._cleanup_thread.start()
        logger.info("Started memory pool cleanup thread")
    
    def stop(self):
        """Stop memory pool and cleanup thread."""
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5.0)
            logger.info("Stopped memory pool cleanup thread")
    
    def _generate_pool_key(self, shape: Tuple[int, ...], dtype: mx.Dtype) -> str:
        """Generate pool key from shape and dtype."""
        return f"{shape}_{dtype}"
    
    def _estimate_tensor_size(self, shape: Tuple[int, ...], dtype: mx.Dtype) -> int:
        """Estimate tensor size in bytes."""
        # Calculate number of elements
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        
        # Get dtype size
        dtype_sizes = {
            mx.float32: 4,
            mx.float16: 2,
            mx.int32: 4,
            mx.int16: 2,
            mx.int8: 1,
            mx.uint8: 1,
        }
        
        dtype_size = dtype_sizes.get(dtype, 4)  # Default to 4 bytes
        
        return num_elements * dtype_size


# Global memory pool instance
_global_memory_pool: Optional[TensorMemoryPool] = None
_pool_lock = threading.Lock()


def get_global_memory_pool() -> TensorMemoryPool:
    """
    Get or create global memory pool instance.
    
    Returns:
        Global memory pool
    """
    global _global_memory_pool
    
    with _pool_lock:
        if _global_memory_pool is None:
            _global_memory_pool = TensorMemoryPool()
        return _global_memory_pool


def configure_global_memory_pool(
    max_pool_size_bytes: int = 2 * 1024 * 1024 * 1024,
    max_idle_seconds: float = 300.0,
    enable_auto_cleanup: bool = True
):
    """
    Configure global memory pool.
    
    Args:
        max_pool_size_bytes: Maximum pool size in bytes
        max_idle_seconds: Maximum idle time before cleanup
        enable_auto_cleanup: Enable automatic cleanup
    """
    global _global_memory_pool
    
    with _pool_lock:
        # Stop existing pool if any
        if _global_memory_pool is not None:
            _global_memory_pool.stop()
        
        _global_memory_pool = TensorMemoryPool(
            max_pool_size_bytes=max_pool_size_bytes,
            max_idle_seconds=max_idle_seconds,
            enable_auto_cleanup=enable_auto_cleanup
        )
        logger.info("Configured global memory pool")
