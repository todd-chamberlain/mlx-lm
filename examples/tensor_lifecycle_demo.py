#!/usr/bin/env python3
"""
Demonstration of Tensor Lifecycle Management

This script demonstrates the tensor lifecycle management capabilities
including tracking, eager deallocation, leak detection, and profiling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from common.tensor_lifecycle import TensorLifecycleManager


def demo_basic_tracking():
    """Demonstrate basic tensor tracking."""
    print("\n=== Basic Tensor Tracking ===")
    
    manager = TensorLifecycleManager()
    
    # Track some tensors
    for i in range(5):
        tensor_id = f"tensor_{i}"
        tensor = mx.zeros((100, 100), dtype=mx.float32)
        manager.track_tensor(tensor_id, tensor, context="demo")
        print(f"Tracked {tensor_id}")
    
    # Get stats
    stats = manager.get_stats()
    print(f"\nStats after tracking:")
    print(f"  Total allocations: {stats.total_allocations}")
    print(f"  Current tracked: {stats.current_tracked}")
    print(f"  Total bytes: {stats.current_bytes_tracked:,} bytes")
    
    # Release tensors
    for i in range(5):
        tensor_id = f"tensor_{i}"
        deallocated = manager.release_tensor(tensor_id)
        print(f"Released {tensor_id} (deallocated: {deallocated})")
    
    # Get final stats
    stats = manager.get_stats()
    print(f"\nStats after release:")
    print(f"  Current tracked: {stats.current_tracked}")
    print(f"  Eager deallocations: {stats.eager_deallocations}")
    print(f"  Memory efficiency: {stats.get_memory_efficiency():.2%}")


def demo_reference_counting():
    """Demonstrate reference counting."""
    print("\n=== Reference Counting ===")
    
    manager = TensorLifecycleManager()
    
    # Track a shared tensor
    tensor_id = "shared_tensor"
    tensor = mx.zeros((200, 200), dtype=mx.float32)
    manager.track_tensor(tensor_id, tensor, context="shared")
    print(f"Tracked {tensor_id} with reference count: {manager.get_reference_count(tensor_id)}")
    
    # Increment references
    for i in range(3):
        manager.increment_reference(tensor_id)
        print(f"Incremented reference (count: {manager.get_reference_count(tensor_id)})")
    
    # Release references
    for i in range(4):
        ref_count = manager.get_reference_count(tensor_id)
        deallocated = manager.release_tensor(tensor_id)
        print(f"Released reference (count was: {ref_count}, deallocated: {deallocated})")


def demo_leak_detection():
    """Demonstrate memory leak detection."""
    print("\n=== Memory Leak Detection ===")
    
    import time
    
    manager = TensorLifecycleManager(
        leak_detection_threshold_seconds=1.0  # Short threshold for demo
    )
    
    # Track some normal tensors and release them
    for i in range(3):
        tensor_id = f"normal_{i}"
        tensor = mx.zeros((50, 50), dtype=mx.float32)
        manager.track_tensor(tensor_id, tensor, context="normal")
        manager.release_tensor(tensor_id)
    
    print("Tracked and released 3 normal tensors")
    
    # Track some tensors but don't release them (simulate leak)
    for i in range(2):
        tensor_id = f"leaked_{i}"
        tensor = mx.zeros((100, 100), dtype=mx.float32)
        manager.track_tensor(tensor_id, tensor, context="leaked")
        
        # Make them appear old
        info = manager.tracked_tensors[tensor_id]
        info.created_at = time.time() - 7200  # 2 hours ago
        info.last_accessed = time.time() - 3600  # 1 hour ago
    
    print("Created 2 leaked tensors (simulated old age)")
    
    # Detect leaks
    leak_report = manager.detect_leaks()
    print(f"\nLeak Detection Results:")
    print(f"  Leaked tensors: {len(leak_report.leaked_tensors)}")
    print(f"  Total leaked bytes: {leak_report.total_leaked_bytes:,} bytes")
    print(f"  Leak contexts: {leak_report.leak_contexts}")
    print(f"  Oldest leak age: {leak_report.oldest_leak_age_seconds:.0f} seconds")
    
    # Cleanup leaks
    cleaned = manager.cleanup_leaked_tensors()
    print(f"\nCleaned up {cleaned} leaked tensors")
    
    # Verify cleanup
    stats = manager.get_stats()
    print(f"Current tracked after cleanup: {stats.current_tracked}")


def demo_memory_profiling():
    """Demonstrate memory profiling."""
    print("\n=== Memory Profiling ===")
    
    manager = TensorLifecycleManager(enable_profiling=True)
    
    # Track tensors in different contexts
    contexts = {
        "inference_input": 5,
        "inference_output": 5,
        "cache": 3,
        "activation": 4
    }
    
    for context, count in contexts.items():
        for i in range(count):
            tensor_id = f"{context}_{i}"
            size = 50 if context == "cache" else 100
            tensor = mx.zeros((size, size), dtype=mx.float32)
            manager.track_tensor(tensor_id, tensor, context=context)
    
    print(f"Tracked tensors across {len(contexts)} contexts")
    
    # Get memory profile
    profile = manager.get_memory_profile()
    
    print(f"\nMemory Profile:")
    print(f"  Total allocations: {profile['stats'].total_allocations}")
    print(f"  Current tracked: {profile['stats'].current_tracked}")
    print(f"  Total bytes: {profile['stats'].current_bytes_tracked:,} bytes")
    
    print(f"\nMemory by Context:")
    for context, bytes_used in profile['current_memory_by_context'].items():
        count = profile['current_count_by_context'][context]
        print(f"  {context}: {count} tensors, {bytes_used:,} bytes")
    
    print(f"\nAllocation Rate: {profile['allocation_rate']:.2f}")
    print(f"Memory Efficiency: {profile['memory_efficiency']:.2%}")


def demo_inference_integration():
    """Demonstrate integration with inference engine."""
    print("\n=== Inference Engine Integration ===")
    
    from common.tensor_lifecycle import get_lifecycle_manager, reset_lifecycle_manager
    
    # Reset to get fresh manager
    reset_lifecycle_manager()
    
    # Get global manager
    manager = get_lifecycle_manager()
    
    # Simulate inference requests
    print("Simulating 10 inference requests...")
    
    for request_num in range(10):
        # Track input tensor
        input_id = f"request_{request_num}_input"
        input_tensor = mx.zeros((50, 50), dtype=mx.float32)
        manager.track_tensor(input_id, input_tensor, context="inference_input")
        
        # Track output tensor
        output_id = f"request_{request_num}_output"
        output_tensor = mx.zeros((50, 100), dtype=mx.float32)
        manager.track_tensor(output_id, output_tensor, context="inference_output")
        
        # Release tensors (simulating eager deallocation)
        manager.release_tensor(input_id)
        manager.release_tensor(output_id)
    
    # Get final stats
    stats = manager.get_stats()
    print(f"\nInference Stats:")
    print(f"  Total requests: 10")
    print(f"  Total allocations: {stats.total_allocations}")
    print(f"  Total deallocations: {stats.total_deallocations}")
    print(f"  Current tracked: {stats.current_tracked}")
    print(f"  Eager deallocations: {stats.eager_deallocations}")
    print(f"  Memory efficiency: {stats.get_memory_efficiency():.2%}")
    
    # Verify no leaks
    leak_report = manager.detect_leaks()
    print(f"\nLeak Check:")
    print(f"  Leaked tensors: {len(leak_report.leaked_tensors)}")
    print(f"  Status: {'✅ No leaks detected' if len(leak_report.leaked_tensors) == 0 else '❌ Leaks detected'}")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Tensor Lifecycle Management Demonstration")
    print("=" * 60)
    
    demo_basic_tracking()
    demo_reference_counting()
    demo_leak_detection()
    demo_memory_profiling()
    demo_inference_integration()
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
