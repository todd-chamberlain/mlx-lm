"""
Tests for Garbage Collection Optimizer

Tests GC optimization functionality including:
- GC tuning for inference workloads
- Manual GC triggering
- Metrics tracking
- Prometheus integration
"""

import gc
import time
import pytest
from unittest.mock import Mock, patch
from common.gc_optimizer import (
    GCOptimizer,
    GCEvent,
    GCMetrics,
    initialize_gc_optimizer,
    get_gc_optimizer
)


class TestGCMetrics:
    """Test GC metrics tracking."""
    
    def test_metrics_initialization(self):
        """Test metrics are initialized correctly."""
        metrics = GCMetrics()
        
        assert metrics.total_collections == 0
        assert metrics.manual_collections == 0
        assert metrics.automatic_collections == 0
        assert metrics.total_pause_ms == 0.0
        assert metrics.total_collected == 0
        assert metrics.total_uncollectable == 0
        assert len(metrics.events) == 0
    
    def test_add_manual_event(self):
        """Test adding a manual GC event."""
        metrics = GCMetrics()
        
        event = GCEvent(
            timestamp=time.time(),
            generation=0,
            collected=100,
            uncollectable=0,
            duration_ms=5.0,
            manual=True
        )
        
        metrics.add_event(event)
        
        assert metrics.total_collections == 1
        assert metrics.manual_collections == 1
        assert metrics.automatic_collections == 0
        assert metrics.total_pause_ms == 5.0
        assert metrics.total_collected == 100
        assert metrics.gen0_collections == 1
    
    def test_add_automatic_event(self):
        """Test adding an automatic GC event."""
        metrics = GCMetrics()
        
        event = GCEvent(
            timestamp=time.time(),
            generation=1,
            collected=50,
            uncollectable=2,
            duration_ms=10.0,
            manual=False
        )
        
        metrics.add_event(event)
        
        assert metrics.total_collections == 1
        assert metrics.manual_collections == 0
        assert metrics.automatic_collections == 1
        assert metrics.total_uncollectable == 2
        assert metrics.gen1_collections == 1
    
    def test_average_pause_calculation(self):
        """Test average pause duration calculation."""
        metrics = GCMetrics()
        
        # Add multiple events
        for i in range(5):
            event = GCEvent(
                timestamp=time.time(),
                generation=0,
                collected=100,
                uncollectable=0,
                duration_ms=float(i + 1),
                manual=True
            )
            metrics.add_event(event)
        
        # Average should be (1+2+3+4+5)/5 = 3.0
        assert metrics.get_average_pause_ms() == 3.0
    
    def test_collection_frequency(self):
        """Test collection frequency calculation."""
        from datetime import datetime, timedelta
        
        metrics = GCMetrics()
        
        # Add events over a 10-second window
        base_time = datetime.now()
        for i in range(10):
            event = GCEvent(
                timestamp=base_time + timedelta(seconds=i),
                generation=0,
                collected=100,
                uncollectable=0,
                duration_ms=1.0,
                manual=True
            )
            metrics.add_event(event)
        
        # Should be approximately 10 collections / 60 seconds = 0.167/sec
        # But since all events are within the window, it should be higher
        freq = metrics.get_collection_frequency(window_seconds=60.0)
        assert freq > 0


class TestGCOptimizer:
    """Test GC optimizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Store original GC settings
        self.original_threshold = gc.get_threshold()
        self.original_enabled = gc.isenabled()
    
    def teardown_method(self):
        """Restore original GC settings."""
        gc.set_threshold(*self.original_threshold)
        if self.original_enabled:
            gc.enable()
        else:
            gc.disable()
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = GCOptimizer(
            gc_threshold=(10000, 20, 20),
            manual_gc_enabled=True,
            track_metrics=True
        )
        
        assert optimizer.gc_threshold == (10000, 20, 20)
        assert optimizer.manual_gc_enabled is True
        assert optimizer.track_metrics is True
        assert optimizer._optimized is False
    
    def test_tune_for_inference(self):
        """Test GC tuning for inference workload."""
        optimizer = GCOptimizer(
            gc_threshold=(10000, 20, 20),
            manual_gc_enabled=True
        )
        
        optimizer.tune_for_inference()
        
        assert optimizer._optimized is True
        assert gc.get_threshold() == (10000, 20, 20)
        assert not gc.isenabled()  # Should be disabled for manual mode
    
    def test_restore_defaults(self):
        """Test restoring original GC settings."""
        original_threshold = gc.get_threshold()
        original_enabled = gc.isenabled()
        
        optimizer = GCOptimizer(
            gc_threshold=(10000, 20, 20),
            manual_gc_enabled=True
        )
        optimizer.tune_for_inference()
        
        # Settings should be changed
        assert gc.get_threshold() != original_threshold
        
        # Restore
        optimizer.restore_defaults()
        
        assert gc.get_threshold() == original_threshold
        assert gc.isenabled() == original_enabled
        assert optimizer._optimized is False
    
    def test_manual_gc_cycle(self):
        """Test manual GC triggering."""
        optimizer = GCOptimizer(
            gc_threshold=(10000, 20, 20),
            manual_gc_enabled=True,
            track_metrics=True
        )
        optimizer.tune_for_inference()
        
        # Create some garbage
        garbage = [[] for _ in range(1000)]
        del garbage
        
        # Trigger manual GC
        event = optimizer.manual_gc_cycle(generation=0)
        
        assert event is not None
        assert event.generation == 0
        assert event.manual is True
        assert event.duration_ms >= 0
        assert optimizer.metrics.total_collections == 1
        assert optimizer.metrics.manual_collections == 1
    
    def test_manual_gc_disabled(self):
        """Test that manual GC doesn't run when disabled."""
        optimizer = GCOptimizer(
            gc_threshold=(10000, 20, 20),
            manual_gc_enabled=False
        )
        
        event = optimizer.manual_gc_cycle(generation=0)
        
        assert event is None
        assert optimizer.metrics.total_collections == 0
    
    def test_gc_pause_context_manager(self):
        """Test GC pause context manager."""
        # Enable GC first
        gc.enable()
        
        optimizer = GCOptimizer()
        
        assert gc.isenabled()
        
        with optimizer.gc_pause():
            # GC should be disabled inside context
            assert not gc.isenabled()
        
        # GC should be re-enabled after context
        assert gc.isenabled()
    
    def test_event_callback(self):
        """Test event callback is called."""
        callback_mock = Mock()
        
        optimizer = GCOptimizer(
            manual_gc_enabled=True,
            track_metrics=True,
            event_callback=callback_mock
        )
        optimizer.tune_for_inference()
        
        # Trigger manual GC
        optimizer.manual_gc_cycle(generation=0)
        
        # Callback should have been called
        callback_mock.assert_called_once()
        
        # Check the event passed to callback
        event = callback_mock.call_args[0][0]
        assert isinstance(event, GCEvent)
        assert event.manual is True
    
    def test_get_gc_stats(self):
        """Test getting GC statistics."""
        optimizer = GCOptimizer(
            gc_threshold=(10000, 20, 20),
            manual_gc_enabled=True,
            track_metrics=True
        )
        optimizer.tune_for_inference()
        
        # Trigger some GC cycles
        optimizer.manual_gc_cycle(generation=0)
        optimizer.manual_gc_cycle(generation=0)
        
        stats = optimizer.get_gc_stats()
        
        assert 'enabled' in stats
        assert 'threshold' in stats
        assert 'counts' in stats
        assert 'optimized' in stats
        assert 'manual_gc_enabled' in stats
        assert 'metrics' in stats
        
        assert stats['optimized'] is True
        assert stats['threshold'] == (10000, 20, 20)
        assert stats['metrics']['total_collections'] == 2
        assert stats['metrics']['manual_collections'] == 2
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        optimizer = GCOptimizer(
            manual_gc_enabled=True,
            track_metrics=True
        )
        optimizer.tune_for_inference()
        
        # Trigger some GC cycles
        optimizer.manual_gc_cycle(generation=0)
        optimizer.manual_gc_cycle(generation=0)
        
        assert optimizer.metrics.total_collections == 2
        
        # Reset
        optimizer.reset_metrics()
        
        assert optimizer.metrics.total_collections == 0
        assert len(optimizer.metrics.events) == 0
    
    def test_get_recent_events(self):
        """Test getting recent GC events."""
        optimizer = GCOptimizer(
            manual_gc_enabled=True,
            track_metrics=True
        )
        optimizer.tune_for_inference()
        
        # Trigger multiple GC cycles
        for _ in range(5):
            optimizer.manual_gc_cycle(generation=0)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Get recent events
        recent = optimizer.get_recent_events(count=3)
        
        assert len(recent) == 3
        assert all(isinstance(e, GCEvent) for e in recent)
        
        # Should be in chronological order (oldest to newest)
        for i in range(len(recent) - 1):
            assert recent[i].timestamp <= recent[i + 1].timestamp
    
    def test_max_events_limit(self):
        """Test that event history is limited."""
        optimizer = GCOptimizer(
            manual_gc_enabled=True,
            track_metrics=True,
            max_events=10
        )
        optimizer.tune_for_inference()
        
        # Trigger more GC cycles than max_events
        for _ in range(20):
            optimizer.manual_gc_cycle(generation=0)
        
        # Should only keep last 10 events
        assert len(optimizer.metrics.events) == 10
        assert optimizer.metrics.total_collections == 20


class TestGlobalGCOptimizer:
    """Test global GC optimizer functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Store original GC settings
        self.original_threshold = gc.get_threshold()
        self.original_enabled = gc.isenabled()
    
    def teardown_method(self):
        """Restore original GC settings."""
        gc.set_threshold(*self.original_threshold)
        if self.original_enabled:
            gc.enable()
        else:
            gc.disable()
    
    def test_initialize_global_optimizer(self):
        """Test initializing global optimizer."""
        optimizer = initialize_gc_optimizer(
            gc_threshold=(10000, 20, 20),
            manual_gc_enabled=True,
            track_metrics=True
        )
        
        assert optimizer is not None
        assert optimizer._optimized is True
        
        # Should be accessible via get_gc_optimizer
        global_optimizer = get_gc_optimizer()
        assert global_optimizer is optimizer
    
    def test_global_optimizer_with_callback(self):
        """Test global optimizer with event callback."""
        callback_mock = Mock()
        
        optimizer = initialize_gc_optimizer(
            gc_threshold=(10000, 20, 20),
            manual_gc_enabled=True,
            track_metrics=True,
            event_callback=callback_mock
        )
        
        # Trigger manual GC
        optimizer.manual_gc_cycle(generation=0)
        
        # Callback should have been called
        callback_mock.assert_called_once()


class TestGCOptimizerPerformance:
    """Test GC optimizer performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.original_threshold = gc.get_threshold()
        self.original_enabled = gc.isenabled()
    
    def teardown_method(self):
        """Restore original GC settings."""
        gc.set_threshold(*self.original_threshold)
        if self.original_enabled:
            gc.enable()
        else:
            gc.disable()
    
    def test_gc_pause_reduction(self):
        """Test that manual GC reduces pause times during workload."""
        # This is a basic test - real performance testing would be more comprehensive
        
        optimizer = GCOptimizer(
            gc_threshold=(10000, 20, 20),
            manual_gc_enabled=True,
            track_metrics=True
        )
        optimizer.tune_for_inference()
        
        # Simulate workload with manual GC
        for i in range(10):
            # Create some garbage
            temp = [[] for _ in range(100)]
            del temp
            
            # Manual GC between iterations
            optimizer.manual_gc_cycle(generation=0)
        
        stats = optimizer.get_gc_stats()
        
        # Should have triggered manual collections
        assert stats['metrics']['manual_collections'] == 10
        assert stats['metrics']['total_collections'] == 10
        
        # Average pause should be reasonable (< 100ms for this small workload)
        assert stats['metrics']['average_pause_ms'] < 100.0
    
    def test_metrics_overhead(self):
        """Test that metrics tracking has minimal overhead."""
        optimizer_with_metrics = GCOptimizer(
            manual_gc_enabled=True,
            track_metrics=True
        )
        optimizer_with_metrics.tune_for_inference()
        
        optimizer_without_metrics = GCOptimizer(
            manual_gc_enabled=True,
            track_metrics=False
        )
        optimizer_without_metrics.tune_for_inference()
        
        # Time with metrics
        start = time.perf_counter()
        for _ in range(100):
            optimizer_with_metrics.manual_gc_cycle(generation=0)
        time_with_metrics = time.perf_counter() - start
        
        # Time without metrics
        start = time.perf_counter()
        for _ in range(100):
            optimizer_without_metrics.manual_gc_cycle(generation=0)
        time_without_metrics = time.perf_counter() - start
        
        # Overhead should be minimal (< 100% increase)
        overhead_ratio = time_with_metrics / time_without_metrics
        assert overhead_ratio < 2.0, f"Metrics overhead too high: {overhead_ratio:.2f}x"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
