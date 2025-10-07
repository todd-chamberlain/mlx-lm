"""
Tensor Lifecycle Management for MLX Distributed Inference.

This module provides tensor lifecycle tracking and eager deallocation to optimize
memory usage and prevent memory leaks in long-running inference workloads.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, List, Tuple
from collections import defaultdict
import mlx.core as mx


@dataclass
class TensorInfo:
    """Information about a tracked tensor."""
    tensor_id: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    created_at: float
    last_accessed: float
    reference_count: int
    context: str  # Context where tensor was created (e.g., "inference", "cache")
    
    def update_access(self):
        """Update last access time."""
        self.last_accessed = time.time()


@dataclass
class MemoryLeakReport:
    """Report of potential memory leaks."""
    leaked_tensors: List[TensorInfo]
    total_leaked_bytes: int
    leak_contexts: Dict[str, int]  # Context -> count
    oldest_leak_age_seconds: float


@dataclass
class TensorLifecycleStats:
    """Statistics for tensor lifecycle management."""
    total_allocations: int = 0
    total_deallocations: int = 0
    current_tracked: int = 0
    total_bytes_allocated: int = 0
    total_bytes_deallocated: int = 0
    current_bytes_tracked: int = 0
    eager_deallocations: int = 0
    leaked_tensors_detected: int = 0
    
    def get_allocation_rate(self) -> float:
        """Get allocation rate (allocations per deallocation)."""
        if self.total_deallocations == 0:
            return float('inf')
        return self.total_allocations / self.total_deallocations
    
    def get_memory_efficiency(self) -> float:
        """Get memory efficiency (deallocated / allocated)."""
        if self.total_bytes_allocated == 0:
            return 1.0
        return self.total_bytes_deallocated / self.total_bytes_allocated


class TensorLifecycleManager:
    """
    Manage tensor lifecycle for efficient memory usage.
    
    Tracks tensor references and enables eager deallocation when tensors
    are no longer needed. Provides memory leak detection and profiling.
    """
    
    def __init__(
        self,
        leak_detection_threshold_seconds: float = 3600.0,
        enable_profiling: bool = True
    ):
        """
        Initialize tensor lifecycle manager.
        
        Args:
            leak_detection_threshold_seconds: Time threshold for leak detection
            enable_profiling: Enable detailed memory profiling
        """
        self.tracked_tensors: Dict[str, TensorInfo] = {}
        self.reference_counts: Dict[str, int] = {}
        self.tensor_contexts: Dict[str, str] = {}
        
        self.leak_threshold = leak_detection_threshold_seconds
        self.enable_profiling = enable_profiling
        
        self.stats = TensorLifecycleStats()
        self._lock = threading.RLock()
        
        # Profiling data
        self.allocation_history: List[Tuple[float, int]] = []  # (timestamp, bytes)
        self.context_allocations: Dict[str, int] = defaultdict(int)
        self.context_bytes: Dict[str, int] = defaultdict(int)
    
    def track_tensor(
        self,
        tensor_id: str,
        tensor: mx.array,
        context: str = "unknown"
    ) -> None:
        """
        Track a tensor for lifecycle management.
        
        Args:
            tensor_id: Unique identifier for the tensor
            tensor: MLX array to track
            context: Context where tensor was created
        """
        with self._lock:
            if tensor_id in self.tracked_tensors:
                # Already tracked, just increment reference
                self.increment_reference(tensor_id)
                return
            
            # Calculate tensor size
            size_bytes = self._calculate_tensor_size(tensor)
            
            # Create tensor info
            now = time.time()
            info = TensorInfo(
                tensor_id=tensor_id,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                size_bytes=size_bytes,
                created_at=now,
                last_accessed=now,
                reference_count=1,
                context=context
            )
            
            self.tracked_tensors[tensor_id] = info
            self.reference_counts[tensor_id] = 1
            self.tensor_contexts[tensor_id] = context
            
            # Update stats
            self.stats.total_allocations += 1
            self.stats.current_tracked += 1
            self.stats.total_bytes_allocated += size_bytes
            self.stats.current_bytes_tracked += size_bytes
            
            # Update profiling data
            if self.enable_profiling:
                self.allocation_history.append((now, size_bytes))
                self.context_allocations[context] += 1
                self.context_bytes[context] += size_bytes
    
    def increment_reference(self, tensor_id: str) -> None:
        """
        Increment reference count for a tensor.
        
        Args:
            tensor_id: Tensor identifier
        """
        with self._lock:
            if tensor_id not in self.reference_counts:
                return
            
            self.reference_counts[tensor_id] += 1
            
            if tensor_id in self.tracked_tensors:
                self.tracked_tensors[tensor_id].reference_count += 1
                self.tracked_tensors[tensor_id].update_access()
    
    def release_tensor(self, tensor_id: str) -> bool:
        """
        Release a tensor reference and eagerly deallocate if no longer needed.
        
        Args:
            tensor_id: Tensor identifier
            
        Returns:
            True if tensor was deallocated, False otherwise
        """
        with self._lock:
            if tensor_id not in self.reference_counts:
                return False
            
            # Decrement reference count
            self.reference_counts[tensor_id] -= 1
            
            if tensor_id in self.tracked_tensors:
                self.tracked_tensors[tensor_id].reference_count -= 1
                self.tracked_tensors[tensor_id].update_access()
            
            # Eagerly deallocate if no more references
            if self.reference_counts[tensor_id] <= 0:
                return self._deallocate_tensor(tensor_id)
            
            return False
    
    def _deallocate_tensor(self, tensor_id: str) -> bool:
        """
        Deallocate a tensor and clean up tracking.
        
        Args:
            tensor_id: Tensor identifier
            
        Returns:
            True if deallocated successfully
        """
        if tensor_id not in self.tracked_tensors:
            return False
        
        info = self.tracked_tensors[tensor_id]
        
        # Update stats
        self.stats.total_deallocations += 1
        self.stats.current_tracked -= 1
        self.stats.total_bytes_deallocated += info.size_bytes
        self.stats.current_bytes_tracked -= info.size_bytes
        self.stats.eager_deallocations += 1
        
        # Remove from tracking
        del self.tracked_tensors[tensor_id]
        del self.reference_counts[tensor_id]
        if tensor_id in self.tensor_contexts:
            del self.tensor_contexts[tensor_id]
        
        # Clear MLX cache to free memory
        mx.metal.clear_cache()
        
        return True
    
    def get_tensor_info(self, tensor_id: str) -> Optional[TensorInfo]:
        """
        Get information about a tracked tensor.
        
        Args:
            tensor_id: Tensor identifier
            
        Returns:
            TensorInfo if tracked, None otherwise
        """
        with self._lock:
            return self.tracked_tensors.get(tensor_id)
    
    def get_reference_count(self, tensor_id: str) -> int:
        """
        Get reference count for a tensor.
        
        Args:
            tensor_id: Tensor identifier
            
        Returns:
            Reference count, or 0 if not tracked
        """
        with self._lock:
            return self.reference_counts.get(tensor_id, 0)
    
    def detect_leaks(self) -> MemoryLeakReport:
        """
        Detect potential memory leaks.
        
        Returns:
            MemoryLeakReport with leak information
        """
        with self._lock:
            now = time.time()
            leaked_tensors = []
            total_leaked_bytes = 0
            leak_contexts = defaultdict(int)
            oldest_leak_age = 0.0
            
            for tensor_id, info in self.tracked_tensors.items():
                age = now - info.created_at
                
                # Consider a tensor leaked if it's been around too long
                # and hasn't been accessed recently
                if age > self.leak_threshold:
                    time_since_access = now - info.last_accessed
                    
                    if time_since_access > self.leak_threshold / 2:
                        leaked_tensors.append(info)
                        total_leaked_bytes += info.size_bytes
                        leak_contexts[info.context] += 1
                        oldest_leak_age = max(oldest_leak_age, age)
            
            # Update stats
            self.stats.leaked_tensors_detected = len(leaked_tensors)
            
            return MemoryLeakReport(
                leaked_tensors=leaked_tensors,
                total_leaked_bytes=total_leaked_bytes,
                leak_contexts=dict(leak_contexts),
                oldest_leak_age_seconds=oldest_leak_age
            )
    
    def cleanup_leaked_tensors(self) -> int:
        """
        Clean up detected leaked tensors.
        
        Returns:
            Number of tensors cleaned up
        """
        leak_report = self.detect_leaks()
        
        with self._lock:
            cleaned = 0
            for info in leak_report.leaked_tensors:
                if self._deallocate_tensor(info.tensor_id):
                    cleaned += 1
            
            return cleaned
    
    def get_stats(self) -> TensorLifecycleStats:
        """
        Get lifecycle statistics.
        
        Returns:
            TensorLifecycleStats
        """
        with self._lock:
            return TensorLifecycleStats(
                total_allocations=self.stats.total_allocations,
                total_deallocations=self.stats.total_deallocations,
                current_tracked=self.stats.current_tracked,
                total_bytes_allocated=self.stats.total_bytes_allocated,
                total_bytes_deallocated=self.stats.total_bytes_deallocated,
                current_bytes_tracked=self.stats.current_bytes_tracked,
                eager_deallocations=self.stats.eager_deallocations,
                leaked_tensors_detected=self.stats.leaked_tensors_detected
            )
    
    def get_memory_profile(self) -> Dict[str, any]:
        """
        Get detailed memory profiling information.
        
        Returns:
            Dictionary with profiling data
        """
        with self._lock:
            stats = self.get_stats()
            
            # Calculate memory by context
            context_memory = defaultdict(int)
            context_counts = defaultdict(int)
            for info in self.tracked_tensors.values():
                context_memory[info.context] += info.size_bytes
                context_counts[info.context] += 1
            
            # Calculate allocation rate over time
            recent_allocations = [
                (ts, size) for ts, size in self.allocation_history
                if time.time() - ts < 300  # Last 5 minutes
            ]
            
            return {
                'stats': stats,
                'current_memory_by_context': dict(context_memory),
                'current_count_by_context': dict(context_counts),
                'total_allocations_by_context': dict(self.context_allocations),
                'total_bytes_by_context': dict(self.context_bytes),
                'recent_allocation_count': len(recent_allocations),
                'recent_allocation_bytes': sum(size for _, size in recent_allocations),
                'allocation_rate': stats.get_allocation_rate(),
                'memory_efficiency': stats.get_memory_efficiency()
            }
    
    def reset_stats(self) -> None:
        """Reset statistics (for testing)."""
        with self._lock:
            self.stats = TensorLifecycleStats()
            self.allocation_history.clear()
            self.context_allocations.clear()
            self.context_bytes.clear()
    
    def clear_all(self) -> int:
        """
        Clear all tracked tensors (for cleanup).
        
        Returns:
            Number of tensors cleared
        """
        with self._lock:
            count = len(self.tracked_tensors)
            
            for tensor_id in list(self.tracked_tensors.keys()):
                self._deallocate_tensor(tensor_id)
            
            return count
    
    @staticmethod
    def _calculate_tensor_size(tensor: mx.array) -> int:
        """
        Calculate tensor size in bytes.
        
        Args:
            tensor: MLX array
            
        Returns:
            Size in bytes
        """
        # Calculate number of elements
        num_elements = 1
        for dim in tensor.shape:
            num_elements *= dim
        
        # Get dtype size - handle both 'float32' and 'mlx.core.float32' formats
        dtype_sizes = {
            'float32': 4,
            'mlx.core.float32': 4,
            'float16': 2,
            'mlx.core.float16': 2,
            'int32': 4,
            'mlx.core.int32': 4,
            'int16': 2,
            'mlx.core.int16': 2,
            'int8': 1,
            'mlx.core.int8': 1,
            'uint8': 1,
            'mlx.core.uint8': 1,
            'bool': 1,
            'mlx.core.bool_': 1
        }
        
        dtype_str = str(tensor.dtype)
        dtype_size = dtype_sizes.get(dtype_str, 4)  # Default to 4 bytes
        
        return num_elements * dtype_size


# Global lifecycle manager instance
_global_lifecycle_manager: Optional[TensorLifecycleManager] = None
_manager_lock = threading.Lock()


def get_lifecycle_manager() -> TensorLifecycleManager:
    """
    Get the global tensor lifecycle manager instance.
    
    Returns:
        TensorLifecycleManager instance
    """
    global _global_lifecycle_manager
    
    with _manager_lock:
        if _global_lifecycle_manager is None:
            _global_lifecycle_manager = TensorLifecycleManager()
        
        return _global_lifecycle_manager


def reset_lifecycle_manager() -> None:
    """Reset the global lifecycle manager (for testing)."""
    global _global_lifecycle_manager
    
    with _manager_lock:
        _global_lifecycle_manager = None
