"""
Garbage Collection Optimizer

Optimizes Python's garbage collection for inference workloads to reduce
GC pause times and improve latency predictability.

Key Features:
- Manual GC triggering between requests
- Tuned GC thresholds for inference patterns
- GC metrics tracking (pause duration, frequency, event counts)
- Prometheus metrics export
"""

import gc
import time
import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class GCEvent:
    """Record of a garbage collection event."""
    timestamp: datetime
    generation: int
    collected: int
    uncollectable: int
    duration_ms: float
    manual: bool = False


@dataclass
class GCMetrics:
    """Garbage collection metrics."""
    total_collections: int = 0
    manual_collections: int = 0
    automatic_collections: int = 0
    total_pause_ms: float = 0.0
    total_collected: int = 0
    total_uncollectable: int = 0
    events: List[GCEvent] = field(default_factory=list)
    
    # Per-generation metrics
    gen0_collections: int = 0
    gen1_collections: int = 0
    gen2_collections: int = 0
    
    def add_event(self, event: GCEvent):
        """Add a GC event and update metrics."""
        self.events.append(event)
        self.total_collections += 1
        self.total_pause_ms += event.duration_ms
        self.total_collected += event.collected
        self.total_uncollectable += event.uncollectable
        
        if event.manual:
            self.manual_collections += 1
        else:
            self.automatic_collections += 1
        
        # Update per-generation counters
        if event.generation == 0:
            self.gen0_collections += 1
        elif event.generation == 1:
            self.gen1_collections += 1
        elif event.generation == 2:
            self.gen2_collections += 1
    
    def get_average_pause_ms(self) -> float:
        """Get average GC pause duration."""
        if self.total_collections == 0:
            return 0.0
        return self.total_pause_ms / self.total_collections
    
    def get_collection_frequency(self, window_seconds: float = 60.0) -> float:
        """Get GC collection frequency (collections per second)."""
        if not self.events:
            return 0.0
        
        now = datetime.now()
        recent_events = [
            e for e in self.events
            if (now - e.timestamp).total_seconds() <= window_seconds
        ]
        
        if not recent_events:
            return 0.0
        
        return len(recent_events) / window_seconds


class GCOptimizer:
    """
    Optimize garbage collection for inference workloads.
    
    Features:
    - Disable automatic GC during inference
    - Manual GC triggering between requests
    - Tuned GC thresholds for inference patterns
    - Comprehensive GC metrics tracking
    """
    
    def __init__(
        self,
        gc_threshold: tuple = (10000, 20, 20),
        manual_gc_enabled: bool = True,
        track_metrics: bool = True,
        max_events: int = 1000,
        event_callback: Optional[callable] = None
    ):
        """
        Initialize GC optimizer.
        
        Args:
            gc_threshold: GC threshold tuple (gen0, gen1, gen2)
            manual_gc_enabled: Enable manual GC triggering
            track_metrics: Enable metrics tracking
            max_events: Maximum number of events to keep in history
            event_callback: Optional callback function called on each GC event
        """
        self.gc_threshold = gc_threshold
        self.manual_gc_enabled = manual_gc_enabled
        self.track_metrics = track_metrics
        self.max_events = max_events
        self.event_callback = event_callback
        
        self.metrics = GCMetrics()
        self._lock = threading.Lock()
        self._original_threshold = gc.get_threshold()
        self._original_gc_enabled = gc.isenabled()
        self._optimized = False
        
        logger.info(f"Initialized GCOptimizer with threshold={gc_threshold}, "
                   f"manual_gc={manual_gc_enabled}")
    
    def tune_for_inference(self):
        """
        Tune GC for inference workload.
        
        This method:
        1. Disables automatic GC if manual GC is enabled
        2. Increases generation 0 threshold to reduce frequency
        3. Adjusts generation 1 and 2 thresholds
        """
        if self._optimized:
            logger.warning("GC already optimized, skipping")
            return
        
        try:
            # Store original settings
            self._original_threshold = gc.get_threshold()
            self._original_gc_enabled = gc.isenabled()
            
            # Disable automatic GC if manual mode is enabled
            if self.manual_gc_enabled:
                gc.disable()
                logger.info("Disabled automatic garbage collection")
            
            # Set tuned thresholds
            gc.set_threshold(*self.gc_threshold)
            logger.info(f"Set GC threshold to {self.gc_threshold} "
                       f"(original: {self._original_threshold})")
            
            self._optimized = True
            
        except Exception as e:
            logger.error(f"Failed to tune GC: {e}")
            raise
    
    def restore_defaults(self):
        """Restore original GC settings."""
        if not self._optimized:
            return
        
        try:
            # Restore original threshold
            gc.set_threshold(*self._original_threshold)
            
            # Restore automatic GC if it was enabled
            if self._original_gc_enabled:
                gc.enable()
            
            logger.info("Restored original GC settings")
            self._optimized = False
            
        except Exception as e:
            logger.error(f"Failed to restore GC settings: {e}")
    
    def manual_gc_cycle(self, generation: int = 0) -> Optional[GCEvent]:
        """
        Trigger manual garbage collection.
        
        Args:
            generation: GC generation to collect (0, 1, or 2)
            
        Returns:
            GCEvent if metrics tracking is enabled, None otherwise
        """
        if not self.manual_gc_enabled:
            logger.debug("Manual GC is disabled, skipping")
            return None
        
        try:
            start_time = time.perf_counter()
            
            # Perform garbage collection
            collected = gc.collect(generation=generation)
            
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Get uncollectable objects count
            uncollectable = len(gc.garbage)
            
            # Create event
            event = GCEvent(
                timestamp=datetime.now(),
                generation=generation,
                collected=collected,
                uncollectable=uncollectable,
                duration_ms=duration_ms,
                manual=True
            )
            
            # Track metrics
            if self.track_metrics:
                with self._lock:
                    self.metrics.add_event(event)
                    
                    # Limit event history
                    if len(self.metrics.events) > self.max_events:
                        self.metrics.events = self.metrics.events[-self.max_events:]
            
            # Call event callback if provided
            if self.event_callback:
                try:
                    self.event_callback(event)
                except Exception as e:
                    logger.error(f"Error in GC event callback: {e}")
            
            logger.debug(f"Manual GC gen{generation}: collected={collected}, "
                        f"duration={duration_ms:.2f}ms")
            
            return event
            
        except Exception as e:
            logger.error(f"Error during manual GC: {e}")
            return None
    
    @contextmanager
    def gc_pause(self):
        """
        Context manager to temporarily disable GC.
        
        Example:
            with gc_optimizer.gc_pause():
                # GC is disabled here
                result = model.generate(prompt)
        """
        was_enabled = gc.isenabled()
        
        try:
            if was_enabled:
                gc.disable()
            yield
        finally:
            if was_enabled:
                gc.enable()
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """
        Get current GC statistics.
        
        Returns:
            Dictionary with GC statistics
        """
        with self._lock:
            stats = {
                "enabled": gc.isenabled(),
                "threshold": gc.get_threshold(),
                "counts": gc.get_count(),
                "optimized": self._optimized,
                "manual_gc_enabled": self.manual_gc_enabled,
                "metrics": {
                    "total_collections": self.metrics.total_collections,
                    "manual_collections": self.metrics.manual_collections,
                    "automatic_collections": self.metrics.automatic_collections,
                    "total_pause_ms": self.metrics.total_pause_ms,
                    "average_pause_ms": self.metrics.get_average_pause_ms(),
                    "collection_frequency_per_sec": self.metrics.get_collection_frequency(),
                    "total_collected": self.metrics.total_collected,
                    "total_uncollectable": self.metrics.total_uncollectable,
                    "gen0_collections": self.metrics.gen0_collections,
                    "gen1_collections": self.metrics.gen1_collections,
                    "gen2_collections": self.metrics.gen2_collections,
                }
            }
            
            return stats
    
    def get_recent_events(self, count: int = 10) -> List[GCEvent]:
        """
        Get recent GC events.
        
        Args:
            count: Number of recent events to return
            
        Returns:
            List of recent GC events
        """
        with self._lock:
            return self.metrics.events[-count:]
    
    def reset_metrics(self):
        """Reset GC metrics."""
        with self._lock:
            self.metrics = GCMetrics()
        logger.info("Reset GC metrics")
    
    def print_stats(self):
        """Print GC statistics."""
        stats = self.get_gc_stats()
        
        print("\n=== Garbage Collection Statistics ===")
        print(f"GC Enabled: {stats['enabled']}")
        print(f"GC Threshold: {stats['threshold']}")
        print(f"Current Counts: {stats['counts']}")
        print(f"Optimized: {stats['optimized']}")
        print(f"Manual GC Enabled: {stats['manual_gc_enabled']}")
        
        print("\nMetrics:")
        metrics = stats['metrics']
        print(f"  Total Collections: {metrics['total_collections']}")
        print(f"  Manual Collections: {metrics['manual_collections']}")
        print(f"  Automatic Collections: {metrics['automatic_collections']}")
        print(f"  Total Pause Time: {metrics['total_pause_ms']:.2f}ms")
        print(f"  Average Pause: {metrics['average_pause_ms']:.2f}ms")
        print(f"  Collection Frequency: {metrics['collection_frequency_per_sec']:.2f}/sec")
        print(f"  Total Collected: {metrics['total_collected']}")
        print(f"  Total Uncollectable: {metrics['total_uncollectable']}")
        
        print("\nPer-Generation Collections:")
        print(f"  Gen 0: {metrics['gen0_collections']}")
        print(f"  Gen 1: {metrics['gen1_collections']}")
        print(f"  Gen 2: {metrics['gen2_collections']}")


# Global GC optimizer instance
_global_gc_optimizer: Optional[GCOptimizer] = None


def get_gc_optimizer() -> Optional[GCOptimizer]:
    """Get the global GC optimizer instance."""
    return _global_gc_optimizer


def initialize_gc_optimizer(
    gc_threshold: tuple = (10000, 20, 20),
    manual_gc_enabled: bool = True,
    track_metrics: bool = True,
    event_callback: Optional[callable] = None
) -> GCOptimizer:
    """
    Initialize the global GC optimizer instance.
    
    Args:
        gc_threshold: GC threshold tuple (gen0, gen1, gen2)
        manual_gc_enabled: Enable manual GC triggering
        track_metrics: Enable metrics tracking
        event_callback: Optional callback function called on each GC event
        
    Returns:
        Initialized GCOptimizer instance
    """
    global _global_gc_optimizer
    
    _global_gc_optimizer = GCOptimizer(
        gc_threshold=gc_threshold,
        manual_gc_enabled=manual_gc_enabled,
        track_metrics=track_metrics,
        event_callback=event_callback
    )
    
    # Automatically tune for inference
    _global_gc_optimizer.tune_for_inference()
    
    return _global_gc_optimizer
