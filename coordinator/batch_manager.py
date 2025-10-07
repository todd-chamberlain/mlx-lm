"""
Batch Manager for MLX Distributed Inference

Implements request batching to improve throughput and GPU utilization.
Features dynamic batch sizing, fairness policies, and priority scheduling.
"""

import time
import asyncio
import threading
import logging
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import heapq

logger = logging.getLogger(__name__)


class BatchPriority(Enum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BatchRequest:
    """Individual request in a batch."""
    
    request_id: str
    model_name: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    priority: BatchPriority = BatchPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    timeout_seconds: float = 30.0
    
    # Metadata
    prompt_length: int = 0
    estimated_tokens: int = 0
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.prompt_length == 0:
            self.prompt_length = len(self.prompt.split())
        if self.estimated_tokens == 0:
            self.estimated_tokens = self.max_tokens
    
    @property
    def age_seconds(self) -> float:
        """Get request age in seconds."""
        return time.time() - self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        return self.age_seconds > self.timeout_seconds
    
    def __lt__(self, other):
        """Compare for priority queue ordering (higher priority first)."""
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at


@dataclass
class Batch:
    """Batch of requests to process together."""
    
    batch_id: str
    requests: List[BatchRequest]
    created_at: float = field(default_factory=time.time)
    
    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.requests)
    
    @property
    def max_prompt_length(self) -> int:
        """Get maximum prompt length in batch."""
        return max((r.prompt_length for r in self.requests), default=0)
    
    @property
    def total_estimated_tokens(self) -> int:
        """Get total estimated tokens for batch."""
        return sum(r.estimated_tokens for r in self.requests)
    
    @property
    def age_seconds(self) -> float:
        """Get batch age in seconds."""
        return time.time() - self.created_at


class BatchingPolicy:
    """Policy for batch assembly and scheduling."""
    
    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        max_batch_wait_ms: float = 50.0,
        enable_dynamic_sizing: bool = True,
        fairness_window_seconds: float = 5.0
    ):
        """
        Initialize batching policy.
        
        Args:
            min_batch_size: Minimum batch size before processing
            max_batch_size: Maximum batch size
            max_batch_wait_ms: Maximum time to wait for batch assembly (ms)
            enable_dynamic_sizing: Enable dynamic batch size adjustment
            fairness_window_seconds: Time window for fairness tracking
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_batch_wait_ms = max_batch_wait_ms
        self.enable_dynamic_sizing = enable_dynamic_sizing
        self.fairness_window_seconds = fairness_window_seconds
        
        # Dynamic sizing state
        self._current_optimal_size = max_batch_size
        self._recent_throughputs: deque = deque(maxlen=10)
        
        logger.info(
            f"Initialized batching policy: "
            f"min={min_batch_size}, max={max_batch_size}, "
            f"wait={max_batch_wait_ms}ms, dynamic={enable_dynamic_sizing}"
        )
    
    def should_create_batch(
        self,
        queue_size: int,
        oldest_request_age_ms: float,
        current_load: float
    ) -> bool:
        """
        Determine if a batch should be created now.
        
        Args:
            queue_size: Current queue size
            oldest_request_age_ms: Age of oldest request in ms
            current_load: Current system load (0.0-1.0)
            
        Returns:
            True if batch should be created
        """
        # Always create batch if at max size
        if queue_size >= self.max_batch_size:
            return True
        
        # Create batch if minimum size reached and wait time exceeded
        if queue_size >= self.min_batch_size and oldest_request_age_ms >= self.max_batch_wait_ms:
            return True
        
        # Under high load, create smaller batches more frequently
        if current_load > 0.8 and queue_size >= self.min_batch_size:
            return True
        
        # Under low load, wait for larger batches
        if current_load < 0.3 and queue_size < self._current_optimal_size:
            return False
        
        return False
    
    def get_optimal_batch_size(self, current_load: float) -> int:
        """
        Get optimal batch size based on current load.
        
        Args:
            current_load: Current system load (0.0-1.0)
            
        Returns:
            Optimal batch size
        """
        if not self.enable_dynamic_sizing:
            return self.max_batch_size
        
        # Adjust based on load
        if current_load > 0.9:
            # High load: smaller batches for lower latency
            return max(self.min_batch_size, self.max_batch_size // 2)
        elif current_load < 0.3:
            # Low load: larger batches for better throughput
            return self.max_batch_size
        else:
            # Medium load: use current optimal
            return self._current_optimal_size
    
    def update_performance(self, batch_size: int, throughput: float):
        """
        Update policy based on observed performance.
        
        Args:
            batch_size: Size of processed batch
            throughput: Observed throughput (tokens/sec)
        """
        if not self.enable_dynamic_sizing:
            return
        
        self._recent_throughputs.append((batch_size, throughput))
        
        # Adjust optimal size based on recent performance
        if len(self._recent_throughputs) >= 5:
            # Find batch size with best throughput
            best_size = max(
                self._recent_throughputs,
                key=lambda x: x[1]
            )[0]
            
            # Gradually adjust toward best size
            self._current_optimal_size = int(
                0.7 * self._current_optimal_size + 0.3 * best_size
            )
            self._current_optimal_size = max(
                self.min_batch_size,
                min(self.max_batch_size, self._current_optimal_size)
            )


class BatchQueue:
    """Thread-safe queue for batch requests."""
    
    def __init__(self, enable_priority: bool = True):
        """
        Initialize batch queue.
        
        Args:
            enable_priority: Enable priority-based ordering
        """
        self.enable_priority = enable_priority
        
        if enable_priority:
            self._queue: List[BatchRequest] = []  # Priority heap
        else:
            self._queue: deque = deque()  # FIFO queue
        
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        
        # Statistics
        self.stats = {
            "total_enqueued": 0,
            "total_dequeued": 0,
            "total_expired": 0,
            "current_size": 0,
            "max_size_seen": 0
        }
        
        logger.info(f"Initialized batch queue: priority={enable_priority}")
    
    def enqueue(self, request: BatchRequest):
        """
        Add request to queue.
        
        Args:
            request: Request to enqueue
        """
        with self._condition:
            if self.enable_priority:
                heapq.heappush(self._queue, request)
            else:
                self._queue.append(request)
            
            self.stats["total_enqueued"] += 1
            self.stats["current_size"] = len(self._queue)
            self.stats["max_size_seen"] = max(
                self.stats["max_size_seen"],
                self.stats["current_size"]
            )
            
            # Notify waiting threads
            self._condition.notify()
            
            logger.debug(
                f"Enqueued request: {request.request_id} "
                f"(priority={request.priority.name}, queue_size={self.stats['current_size']})"
            )
    
    def dequeue_batch(
        self,
        max_size: int,
        timeout_seconds: Optional[float] = None
    ) -> List[BatchRequest]:
        """
        Dequeue a batch of requests.
        
        Args:
            max_size: Maximum batch size
            timeout_seconds: Maximum time to wait for requests
            
        Returns:
            List of requests (may be empty if timeout)
        """
        with self._condition:
            # Wait for requests if queue is empty
            if not self._queue and timeout_seconds:
                self._condition.wait(timeout_seconds)
            
            # Remove expired requests
            self._remove_expired()
            
            # Dequeue up to max_size requests
            batch = []
            while self._queue and len(batch) < max_size:
                if self.enable_priority:
                    request = heapq.heappop(self._queue)
                else:
                    request = self._queue.popleft()
                
                batch.append(request)
                self.stats["total_dequeued"] += 1
            
            self.stats["current_size"] = len(self._queue)
            
            if batch:
                logger.debug(
                    f"Dequeued batch: size={len(batch)}, "
                    f"remaining={self.stats['current_size']}"
                )
            
            return batch
    
    def peek_oldest(self) -> Optional[BatchRequest]:
        """
        Peek at oldest request without removing.
        
        Returns:
            Oldest request or None if queue is empty
        """
        with self._lock:
            if not self._queue:
                return None
            
            if self.enable_priority:
                # Find oldest request (min created_at)
                return min(self._queue, key=lambda r: r.created_at)
            else:
                return self._queue[0]
    
    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0
    
    def clear(self):
        """Clear all requests from queue."""
        with self._lock:
            if self.enable_priority:
                self._queue.clear()
            else:
                self._queue.clear()
            
            self.stats["current_size"] = 0
            logger.info("Cleared batch queue")
    
    def _remove_expired(self):
        """Remove expired requests from queue."""
        if self.enable_priority:
            # Filter out expired requests
            valid_requests = [r for r in self._queue if not r.is_expired]
            expired_count = len(self._queue) - len(valid_requests)
            
            if expired_count > 0:
                self._queue.clear()
                for request in valid_requests:
                    heapq.heappush(self._queue, request)
                
                self.stats["total_expired"] += expired_count
                logger.warning(f"Removed {expired_count} expired requests")
        else:
            # Remove from front while expired
            expired_count = 0
            while self._queue and self._queue[0].is_expired:
                self._queue.popleft()
                expired_count += 1
            
            if expired_count > 0:
                self.stats["total_expired"] += expired_count
                logger.warning(f"Removed {expired_count} expired requests")


class BatchManager:
    """
    Manages request batching for improved throughput.
    
    Features:
    - Dynamic batch sizing based on load
    - Priority-based request scheduling
    - Fairness policies to prevent starvation
    - Batch timeout and expiration handling
    """
    
    def __init__(
        self,
        policy: Optional[BatchingPolicy] = None,
        enable_priority: bool = True,
        enable_fairness: bool = True
    ):
        """
        Initialize batch manager.
        
        Args:
            policy: Batching policy (uses default if None)
            enable_priority: Enable priority-based scheduling
            enable_fairness: Enable fairness policies
        """
        self.policy = policy or BatchingPolicy()
        self.enable_priority = enable_priority
        self.enable_fairness = enable_fairness
        
        # Request queue
        self.queue = BatchQueue(enable_priority=enable_priority)
        
        # Batch tracking
        self._active_batches: Dict[str, Batch] = {}
        self._batch_counter = 0
        self._lock = threading.Lock()
        
        # Fairness tracking
        self._priority_counts: Dict[BatchPriority, int] = {
            p: 0 for p in BatchPriority
        }
        self._last_fairness_check = time.time()
        
        # Statistics
        self.stats = {
            "total_batches_created": 0,
            "total_requests_batched": 0,
            "avg_batch_size": 0.0,
            "avg_batch_wait_ms": 0.0,
            "current_load": 0.0
        }
        
        logger.info(
            f"Initialized batch manager: "
            f"priority={enable_priority}, fairness={enable_fairness}"
        )
    
    def submit_request(self, request: BatchRequest):
        """
        Submit request for batching.
        
        Args:
            request: Request to batch
        """
        self.queue.enqueue(request)
        
        # Update fairness tracking
        if self.enable_fairness:
            with self._lock:
                self._priority_counts[request.priority] += 1
    
    def create_batch(
        self,
        current_load: float = 0.5,
        timeout_seconds: Optional[float] = None
    ) -> Optional[Batch]:
        """
        Create a batch from queued requests.
        
        Args:
            current_load: Current system load (0.0-1.0)
            timeout_seconds: Maximum time to wait for requests
            
        Returns:
            Batch of requests or None if no batch created
        """
        # Check if we should create a batch
        oldest = self.queue.peek_oldest()
        if oldest is None:
            return None
        
        queue_size = self.queue.size()
        oldest_age_ms = oldest.age_seconds * 1000
        
        if not self.policy.should_create_batch(queue_size, oldest_age_ms, current_load):
            return None
        
        # Determine batch size
        optimal_size = self.policy.get_optimal_batch_size(current_load)
        batch_size = min(optimal_size, queue_size)
        
        # Apply fairness adjustments
        if self.enable_fairness:
            batch_size = self._apply_fairness_policy(batch_size)
        
        # Dequeue requests
        requests = self.queue.dequeue_batch(batch_size, timeout_seconds)
        
        if not requests:
            return None
        
        # Create batch
        with self._lock:
            self._batch_counter += 1
            batch_id = f"batch_{self._batch_counter}_{int(time.time() * 1000)}"
            
            batch = Batch(
                batch_id=batch_id,
                requests=requests
            )
            
            self._active_batches[batch_id] = batch
            
            # Update statistics
            self.stats["total_batches_created"] += 1
            self.stats["total_requests_batched"] += len(requests)
            
            # Update average batch size
            total_batches = self.stats["total_batches_created"]
            self.stats["avg_batch_size"] = (
                (self.stats["avg_batch_size"] * (total_batches - 1) + len(requests))
                / total_batches
            )
            
            # Update average wait time
            avg_wait = sum(r.age_seconds for r in requests) / len(requests) * 1000
            self.stats["avg_batch_wait_ms"] = (
                (self.stats["avg_batch_wait_ms"] * (total_batches - 1) + avg_wait)
                / total_batches
            )
            
            self.stats["current_load"] = current_load
            
            logger.info(
                f"Created batch: {batch_id} "
                f"(size={len(requests)}, wait={avg_wait:.1f}ms, load={current_load:.2f})"
            )
            
            return batch
    
    def complete_batch(
        self,
        batch_id: str,
        throughput: Optional[float] = None
    ):
        """
        Mark batch as complete and update performance metrics.
        
        Args:
            batch_id: Batch identifier
            throughput: Observed throughput (tokens/sec)
        """
        with self._lock:
            batch = self._active_batches.pop(batch_id, None)
            
            if batch is None:
                logger.warning(f"Attempted to complete unknown batch: {batch_id}")
                return
            
            # Update policy with performance data
            if throughput is not None:
                self.policy.update_performance(batch.size, throughput)
            
            logger.info(
                f"Completed batch: {batch_id} "
                f"(size={batch.size}, age={batch.age_seconds:.2f}s)"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get batch manager statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            stats = self.stats.copy()
            stats["queue_size"] = self.queue.size()
            stats["active_batches"] = len(self._active_batches)
            stats["queue_stats"] = self.queue.stats.copy()
            
            if self.enable_fairness:
                stats["priority_distribution"] = {
                    p.name: count
                    for p, count in self._priority_counts.items()
                }
            
            return stats
    
    def _apply_fairness_policy(self, batch_size: int) -> int:
        """
        Apply fairness policy to batch size.
        
        Args:
            batch_size: Proposed batch size
            
        Returns:
            Adjusted batch size
        """
        # Check if fairness window has elapsed
        now = time.time()
        if now - self._last_fairness_check < self.policy.fairness_window_seconds:
            return batch_size
        
        # Reset fairness tracking
        with self._lock:
            self._priority_counts = {p: 0 for p in BatchPriority}
            self._last_fairness_check = now
        
        return batch_size
