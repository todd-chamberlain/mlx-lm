"""
Batch Scheduling Optimization for MLX Distributed Inference

Implements adaptive batch scheduling with load-based optimization, batch preemption,
and performance monitoring for optimal throughput and latency balance.
"""

import time
import asyncio
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from .batch_manager import BatchManager, Batch, BatchRequest, BatchPriority
from .batch_processor import BatchProcessor, BatchResult

logger = logging.getLogger(__name__)


@dataclass
class BatchPerformanceMetrics:
    """Performance metrics for a processed batch."""
    
    batch_size: int
    processing_time_ms: float
    throughput_tokens_per_sec: float
    padding_ratio: float
    success_rate: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def efficiency_score(self) -> float:
        """
        Calculate batch efficiency score.
        
        Higher is better. Considers throughput, success rate, and padding overhead.
        """
        # Penalize high padding ratios
        padding_penalty = 1.0 - (self.padding_ratio * 0.5)
        
        # Combine metrics
        score = (
            self.throughput_tokens_per_sec * 
            self.success_rate * 
            padding_penalty
        )
        
        return score


@dataclass
class LoadMetrics:
    """System load metrics."""
    
    queue_depth: int
    active_batches: int
    worker_utilization: float
    avg_latency_ms: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def load_level(self) -> float:
        """
        Calculate overall load level (0.0 to 1.0).
        
        Combines queue depth, active batches, and worker utilization.
        """
        # Normalize queue depth (assume 100 is high)
        queue_factor = min(self.queue_depth / 100.0, 1.0)
        
        # Normalize active batches (assume 10 is high)
        batch_factor = min(self.active_batches / 10.0, 1.0)
        
        # Worker utilization is already 0-1
        
        # Weighted average
        load = (
            queue_factor * 0.4 +
            batch_factor * 0.3 +
            self.worker_utilization * 0.3
        )
        
        return load


class AdaptiveBatchSizer:
    """
    Adaptive batch sizing based on observed performance.
    
    Features:
    - Track performance metrics per batch size
    - Use exponential moving average for stability
    - Adjust batch size to maximize efficiency
    - Respect min/max constraints
    """
    
    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.1
    ):
        """
        Initialize adaptive batch sizer.
        
        Args:
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            learning_rate: Learning rate for EMA (0.0-1.0)
            exploration_rate: Rate of exploration vs exploitation
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Performance tracking per batch size
        self._performance_history: Dict[int, deque] = {}
        self._avg_efficiency: Dict[int, float] = {}
        
        # Current optimal size
        self._optimal_size = max_batch_size
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        logger.info(
            f"Initialized adaptive batch sizer: "
            f"min={min_batch_size}, max={max_batch_size}, "
            f"lr={learning_rate}, exploration={exploration_rate}"
        )
    
    def update_performance(self, metrics: BatchPerformanceMetrics):
        """
        Update performance metrics for a batch size.
        
        Args:
            metrics: Performance metrics from processed batch
        """
        with self._lock:
            batch_size = metrics.batch_size
            
            # Initialize history if needed
            if batch_size not in self._performance_history:
                self._performance_history[batch_size] = deque(maxlen=20)
                self._avg_efficiency[batch_size] = metrics.efficiency_score
            
            # Add to history
            self._performance_history[batch_size].append(metrics)
            
            # Update exponential moving average
            current_avg = self._avg_efficiency[batch_size]
            new_avg = (
                self.learning_rate * metrics.efficiency_score +
                (1 - self.learning_rate) * current_avg
            )
            self._avg_efficiency[batch_size] = new_avg
            
            # Update optimal size
            self._update_optimal_size()
            
            logger.debug(
                f"Updated batch size {batch_size} performance: "
                f"efficiency={metrics.efficiency_score:.2f}, "
                f"avg={new_avg:.2f}"
            )
    
    def get_optimal_size(self, current_load: float) -> int:
        """
        Get optimal batch size for current load.
        
        Args:
            current_load: Current system load (0.0-1.0)
            
        Returns:
            Optimal batch size
        """
        with self._lock:
            # Exploration: occasionally try different sizes
            import random
            if random.random() < self.exploration_rate:
                # Try a random size
                size = random.randint(self.min_batch_size, self.max_batch_size)
                logger.debug(f"Exploration: trying batch size {size}")
                return size
            
            # Exploitation: use optimal size
            # Adjust based on load
            if current_load > 0.8:
                # High load: prefer smaller batches for lower latency
                adjusted_size = max(
                    self.min_batch_size,
                    int(self._optimal_size * 0.7)
                )
            elif current_load < 0.3:
                # Low load: prefer larger batches for throughput
                adjusted_size = self._optimal_size
            else:
                # Medium load: use optimal size
                adjusted_size = self._optimal_size
            
            return adjusted_size
    
    def _update_optimal_size(self):
        """Update optimal batch size based on performance history."""
        if not self._avg_efficiency:
            return
        
        # Find size with best average efficiency
        best_size = max(
            self._avg_efficiency.keys(),
            key=lambda s: self._avg_efficiency[s]
        )
        
        self._optimal_size = best_size
        
        logger.debug(
            f"Updated optimal batch size: {best_size} "
            f"(efficiency={self._avg_efficiency[best_size]:.2f})"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get batch sizer statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                "optimal_size": self._optimal_size,
                "tracked_sizes": list(self._avg_efficiency.keys()),
                "avg_efficiency_by_size": self._avg_efficiency.copy(),
                "best_efficiency": max(self._avg_efficiency.values()) if self._avg_efficiency else 0.0
            }


class BatchScheduler:
    """
    Optimized batch scheduler with adaptive sizing and preemption.
    
    Features:
    - Adaptive batch sizing based on performance
    - Load-based scheduling decisions
    - Batch preemption for high-priority requests
    - Performance monitoring and metrics
    """
    
    def __init__(
        self,
        batch_manager: BatchManager,
        batch_processor: BatchProcessor,
        enable_adaptive_sizing: bool = True,
        enable_preemption: bool = True,
        high_load_threshold: float = 0.8,
        low_load_threshold: float = 0.3
    ):
        """
        Initialize batch scheduler.
        
        Args:
            batch_manager: Batch manager instance
            batch_processor: Batch processor instance
            enable_adaptive_sizing: Enable adaptive batch sizing
            enable_preemption: Enable batch preemption
            high_load_threshold: Threshold for high load (0.0-1.0)
            low_load_threshold: Threshold for low load (0.0-1.0)
        """
        self.batch_manager = batch_manager
        self.batch_processor = batch_processor
        self.enable_adaptive_sizing = enable_adaptive_sizing
        self.enable_preemption = enable_preemption
        self.high_load_threshold = high_load_threshold
        self.low_load_threshold = low_load_threshold
        
        # Adaptive batch sizer
        self.adaptive_sizer = AdaptiveBatchSizer(
            min_batch_size=batch_manager.policy.min_batch_size,
            max_batch_size=batch_manager.policy.max_batch_size
        ) if enable_adaptive_sizing else None
        
        # Load tracking
        self._load_history: deque = deque(maxlen=100)
        self._current_load = LoadMetrics(
            queue_depth=0,
            active_batches=0,
            worker_utilization=0.0,
            avg_latency_ms=0.0
        )
        
        # Statistics
        self.stats = {
            "batches_scheduled": 0,
            "preemptions": 0,
            "adaptive_adjustments": 0,
            "total_scheduling_time_ms": 0.0
        }
        
        self._lock = threading.Lock()
        
        logger.info(
            f"Initialized batch scheduler: "
            f"adaptive={'enabled' if enable_adaptive_sizing else 'disabled'}, "
            f"preemption={'enabled' if enable_preemption else 'disabled'}"
        )
    
    def update_load_metrics(
        self,
        queue_depth: int,
        active_batches: int,
        worker_utilization: float,
        avg_latency_ms: float
    ):
        """
        Update system load metrics.
        
        Args:
            queue_depth: Current queue depth
            active_batches: Number of active batches
            worker_utilization: Worker utilization (0.0-1.0)
            avg_latency_ms: Average latency in milliseconds
        """
        with self._lock:
            self._current_load = LoadMetrics(
                queue_depth=queue_depth,
                active_batches=active_batches,
                worker_utilization=worker_utilization,
                avg_latency_ms=avg_latency_ms
            )
            
            self._load_history.append(self._current_load)
    
    def should_create_batch(self) -> bool:
        """
        Determine if a batch should be created now.
        
        Returns:
            True if batch should be created
        """
        load_level = self._current_load.load_level
        queue_depth = self._current_load.queue_depth
        
        # Get oldest request age
        oldest = self.batch_manager.queue.peek_oldest()
        if oldest is None:
            return False
        
        oldest_age_ms = oldest.age_seconds * 1000
        
        # Use batch manager's policy
        should_create = self.batch_manager.policy.should_create_batch(
            queue_size=queue_depth,
            oldest_request_age_ms=oldest_age_ms,
            current_load=load_level
        )
        
        # Additional scheduling logic
        if not should_create:
            # High load: create batches more aggressively
            if load_level > self.high_load_threshold and queue_depth >= self.batch_manager.policy.min_batch_size:
                should_create = True
                logger.debug(f"High load batch creation: load={load_level:.2f}")
        
        return should_create
    
    def get_batch_size(self) -> int:
        """
        Get optimal batch size for current conditions.
        
        Returns:
            Optimal batch size
        """
        load_level = self._current_load.load_level
        
        if self.enable_adaptive_sizing and self.adaptive_sizer:
            # Use adaptive sizing
            size = self.adaptive_sizer.get_optimal_size(load_level)
            self.stats["adaptive_adjustments"] += 1
        else:
            # Use policy-based sizing
            size = self.batch_manager.policy.get_optimal_batch_size(load_level)
        
        return size
    
    async def schedule_batch(
        self,
        inference_fn: callable
    ) -> Optional[BatchResult]:
        """
        Schedule and process a batch.
        
        Args:
            inference_fn: Async inference function
            
        Returns:
            BatchResult or None if no batch created
        """
        start_time = time.time()
        
        try:
            # Check if we should create a batch
            if not self.should_create_batch():
                return None
            
            # Check for preemption
            if self.enable_preemption:
                if self._should_preempt():
                    logger.info("Batch preemption triggered for high-priority request")
                    self.stats["preemptions"] += 1
            
            # Get optimal batch size
            batch_size = self.get_batch_size()
            
            # Create batch
            load_level = self._current_load.load_level
            batch = self.batch_manager.create_batch(
                current_load=load_level,
                timeout_seconds=None
            )
            
            if batch is None:
                return None
            
            # Limit batch size if needed
            if len(batch.requests) > batch_size:
                # Split batch - return excess to queue
                excess_requests = batch.requests[batch_size:]
                batch.requests = batch.requests[:batch_size]
                
                # Re-queue excess requests
                for req in excess_requests:
                    self.batch_manager.submit_request(req)
                
                logger.debug(
                    f"Limited batch size: {len(batch.requests)} "
                    f"(returned {len(excess_requests)} to queue)"
                )
            
            # Process batch
            batch_result = await self.batch_processor.process_batch(batch, inference_fn)
            
            # Update performance metrics
            if self.enable_adaptive_sizing and self.adaptive_sizer:
                metrics = self._create_performance_metrics(batch, batch_result)
                self.adaptive_sizer.update_performance(metrics)
            
            # Update statistics
            self.stats["batches_scheduled"] += 1
            scheduling_time = (time.time() - start_time) * 1000
            self.stats["total_scheduling_time_ms"] += scheduling_time
            
            # Complete batch in manager
            throughput = (
                batch_result.tokens_generated / (batch_result.processing_time_ms / 1000)
                if batch_result.processing_time_ms > 0 else 0.0
            )
            self.batch_manager.complete_batch(batch.batch_id, throughput=throughput)
            
            logger.info(
                f"Scheduled batch: {batch.batch_id} "
                f"(size={batch.size}, time={scheduling_time:.1f}ms, "
                f"success_rate={batch_result.success_rate:.2%})"
            )
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Batch scheduling failed: {e}", exc_info=True)
            return None
    
    def _should_preempt(self) -> bool:
        """
        Check if batch should be preempted for high-priority request.
        
        Returns:
            True if preemption should occur
        """
        # Check if there are critical priority requests
        oldest = self.batch_manager.queue.peek_oldest()
        if oldest is None:
            return False
        
        # Preempt if critical priority and waiting too long
        if oldest.priority == BatchPriority.CRITICAL:
            if oldest.age_seconds > 0.1:  # 100ms threshold
                return True
        
        return False
    
    def _create_performance_metrics(
        self,
        batch: Batch,
        result: BatchResult
    ) -> BatchPerformanceMetrics:
        """
        Create performance metrics from batch result.
        
        Args:
            batch: Original batch
            result: Batch processing result
            
        Returns:
            Performance metrics
        """
        # Calculate padding ratio
        if batch.requests:
            prompt_lengths = [req.prompt_length for req in batch.requests]
            max_length = max(prompt_lengths)
            min_length = min(prompt_lengths)
            padding_ratio = (max_length - min_length) / max_length if max_length > 0 else 0.0
        else:
            padding_ratio = 0.0
        
        # Calculate throughput
        throughput = (
            result.tokens_generated / (result.processing_time_ms / 1000)
            if result.processing_time_ms > 0 else 0.0
        )
        
        return BatchPerformanceMetrics(
            batch_size=batch.size,
            processing_time_ms=result.processing_time_ms,
            throughput_tokens_per_sec=throughput,
            padding_ratio=padding_ratio,
            success_rate=result.success_rate
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            stats = self.stats.copy()
            
            # Add current load
            stats["current_load"] = {
                "queue_depth": self._current_load.queue_depth,
                "active_batches": self._current_load.active_batches,
                "worker_utilization": self._current_load.worker_utilization,
                "avg_latency_ms": self._current_load.avg_latency_ms,
                "load_level": self._current_load.load_level
            }
            
            # Add adaptive sizer stats
            if self.adaptive_sizer:
                stats["adaptive_sizer"] = self.adaptive_sizer.get_stats()
            
            # Calculate average scheduling time
            if stats["batches_scheduled"] > 0:
                stats["avg_scheduling_time_ms"] = (
                    stats["total_scheduling_time_ms"] / stats["batches_scheduled"]
                )
            else:
                stats["avg_scheduling_time_ms"] = 0.0
            
            return stats
    
    def get_load_history(self, window_seconds: float = 60.0) -> List[LoadMetrics]:
        """
        Get load history for a time window.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            List of load metrics within window
        """
        with self._lock:
            cutoff_time = time.time() - window_seconds
            return [
                load for load in self._load_history
                if load.timestamp >= cutoff_time
            ]
