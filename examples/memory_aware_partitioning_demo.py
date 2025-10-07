"""
Memory-Aware Partitioning Demonstration

This script demonstrates the benefits of memory-aware partitioning
for heterogeneous clusters with different memory capacities.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from worker.model_partitioner import ModelPartitioner, PartitionStrategy


def print_partition_summary(config, title):
    """Print a summary of partition assignments"""
    print(f"\n{title}")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Total Layers: {config.total_layers}")
    print(f"Strategy: {config.strategy.value}")
    print(f"Nodes: {len(config.assignments)}")
    print("\nAssignments:")
    
    for i, assignment in enumerate(config.assignments, 1):
        print(f"  {i}. {assignment.node_id:20s} | "
              f"Layers {assignment.start_layer:2d}-{assignment.end_layer:2d} | "
              f"Count: {assignment.num_layers:2d}")
    
    print("-" * 80)


def demo_homogeneous_cluster():
    """Demonstrate partitioning on homogeneous cluster"""
    print("\n" + "=" * 80)
    print("DEMO 1: Homogeneous Cluster (All nodes have equal memory)")
    print("=" * 80)
    
    partitioner = ModelPartitioner()
    
    nodes = [
        {'node_id': 'mac-mini-1', 'address': '192.168.1.10:8080', 'available_memory': 32.0},
        {'node_id': 'mac-mini-2', 'address': '192.168.1.11:8080', 'available_memory': 32.0},
        {'node_id': 'mac-mini-3', 'address': '192.168.1.12:8080', 'available_memory': 32.0}
    ]
    
    print("\nCluster Configuration:")
    for node in nodes:
        print(f"  - {node['node_id']:20s}: {node['available_memory']:.0f}GB")
    
    # Balanced partitioning
    balanced = partitioner.partition_model(
        model_name='llama-3.2-3b',
        total_layers=32,
        nodes=nodes,
        strategy=PartitionStrategy.BALANCED
    )
    print_partition_summary(balanced, "Balanced Partitioning")
    
    # Memory-aware partitioning
    memory_aware = partitioner.partition_model(
        model_name='llama-3.2-3b',
        total_layers=32,
        nodes=nodes,
        strategy=PartitionStrategy.MEMORY_AWARE
    )
    print_partition_summary(memory_aware, "Memory-Aware Partitioning")
    
    print("\n📊 Analysis:")
    print("  For homogeneous clusters, both strategies produce similar results.")
    print("  This is expected since all nodes have equal capacity.")


def demo_heterogeneous_cluster():
    """Demonstrate partitioning on heterogeneous cluster"""
    print("\n" + "=" * 80)
    print("DEMO 2: Heterogeneous Cluster (Nodes have different memory)")
    print("=" * 80)
    
    partitioner = ModelPartitioner()
    
    nodes = [
        {'node_id': 'mac-studio-ultra', 'address': '192.168.1.10:8080', 'available_memory': 192.0},
        {'node_id': 'mac-studio-max', 'address': '192.168.1.11:8080', 'available_memory': 96.0},
        {'node_id': 'mac-mini', 'address': '192.168.1.12:8080', 'available_memory': 16.0}
    ]
    
    print("\nCluster Configuration:")
    for node in nodes:
        print(f"  - {node['node_id']:20s}: {node['available_memory']:.0f}GB")
    
    total_memory = sum(n['available_memory'] for n in nodes)
    print(f"\nTotal Memory: {total_memory:.0f}GB")
    print("Memory Proportions:")
    for node in nodes:
        proportion = node['available_memory'] / total_memory * 100
        print(f"  - {node['node_id']:20s}: {proportion:.1f}%")
    
    # Balanced partitioning
    balanced = partitioner.partition_model(
        model_name='llama-70b',
        total_layers=80,
        nodes=nodes,
        strategy=PartitionStrategy.BALANCED
    )
    print_partition_summary(balanced, "Balanced Partitioning")
    
    # Memory-aware partitioning
    memory_aware = partitioner.partition_model(
        model_name='llama-70b',
        total_layers=80,
        nodes=nodes,
        strategy=PartitionStrategy.MEMORY_AWARE
    )
    print_partition_summary(memory_aware, "Memory-Aware Partitioning")
    
    print("\n📊 Analysis:")
    print("  Balanced: Each node gets ~27 layers regardless of memory capacity")
    print("  Memory-Aware: Layers distributed proportionally to memory:")
    
    for i, node in enumerate(nodes):
        balanced_layers = balanced.assignments[i].num_layers
        memory_aware_layers = memory_aware.assignments[i].num_layers
        diff = memory_aware_layers - balanced_layers
        sign = "+" if diff > 0 else ""
        print(f"    - {node['node_id']:20s}: {sign}{diff:2d} layers ({memory_aware_layers} vs {balanced_layers})")
    
    print("\n  ✅ Benefits:")
    print("    - Mac Studio Ultra handles more layers (better utilization)")
    print("    - Mac Mini protected from overload (prevents OOM)")
    print("    - Overall better resource distribution")


def demo_extreme_heterogeneity():
    """Demonstrate partitioning with extreme memory differences"""
    print("\n" + "=" * 80)
    print("DEMO 3: Extreme Heterogeneity (10x memory difference)")
    print("=" * 80)
    
    partitioner = ModelPartitioner()
    
    nodes = [
        {'node_id': 'powerful-workstation', 'address': '192.168.1.10:8080', 'available_memory': 256.0},
        {'node_id': 'basic-laptop', 'address': '192.168.1.11:8080', 'available_memory': 16.0}
    ]
    
    print("\nCluster Configuration:")
    for node in nodes:
        print(f"  - {node['node_id']:25s}: {node['available_memory']:.0f}GB")
    
    ratio = nodes[0]['available_memory'] / nodes[1]['available_memory']
    print(f"\nMemory Ratio: {ratio:.1f}x")
    
    # Balanced partitioning
    balanced = partitioner.partition_model(
        model_name='llama-70b',
        total_layers=80,
        nodes=nodes,
        strategy=PartitionStrategy.BALANCED
    )
    print_partition_summary(balanced, "Balanced Partitioning")
    
    # Memory-aware partitioning
    memory_aware = partitioner.partition_model(
        model_name='llama-70b',
        total_layers=80,
        nodes=nodes,
        strategy=PartitionStrategy.MEMORY_AWARE
    )
    print_partition_summary(memory_aware, "Memory-Aware Partitioning")
    
    print("\n📊 Analysis:")
    print("  Balanced: 40 layers each (likely to cause OOM on laptop)")
    print("  Memory-Aware: ~75 layers on workstation, ~5 layers on laptop")
    print("\n  ✅ Critical Benefits:")
    print("    - Prevents OOM errors on low-memory node")
    print("    - Maximizes utilization of high-memory node")
    print("    - Enables running large models on mixed hardware")


def demo_real_world_mac_studios():
    """Demonstrate real-world Mac Studio scenario"""
    print("\n" + "=" * 80)
    print("DEMO 4: Real-World Mac Studio Deployment")
    print("=" * 80)
    
    partitioner = ModelPartitioner()
    
    nodes = [
        {
            'node_id': 'mac-studio-1-m2-ultra',
            'address': '192.168.1.10:8080',
            'available_memory': 192.0
        },
        {
            'node_id': 'mac-studio-2-m2-max',
            'address': '192.168.1.11:8080',
            'available_memory': 96.0
        }
    ]
    
    print("\nProduction Cluster:")
    print("  - Mac Studio 1: M2 Ultra with 192GB unified memory")
    print("  - Mac Studio 2: M2 Max with 96GB unified memory")
    print("\nModel: Llama-70B (80 layers)")
    
    # Balanced partitioning
    balanced = partitioner.partition_model(
        model_name='llama-70b',
        total_layers=80,
        nodes=nodes,
        strategy=PartitionStrategy.BALANCED
    )
    print_partition_summary(balanced, "Balanced Partitioning")
    
    # Memory-aware partitioning
    memory_aware = partitioner.partition_model(
        model_name='llama-70b',
        total_layers=80,
        nodes=nodes,
        strategy=PartitionStrategy.MEMORY_AWARE
    )
    print_partition_summary(memory_aware, "Memory-Aware Partitioning")
    
    print("\n📊 Production Impact:")
    print("  Balanced Strategy:")
    print("    - Each Mac Studio handles 40 layers")
    print("    - M2 Ultra underutilized (192GB capacity, only using ~50%)")
    print("    - M2 Max potentially stressed (96GB capacity, using ~80%)")
    
    print("\n  Memory-Aware Strategy:")
    print("    - M2 Ultra handles ~53 layers (better utilization)")
    print("    - M2 Max handles ~27 layers (comfortable load)")
    print("    - Overall better performance and stability")
    
    print("\n  ✅ Production Benefits:")
    print("    - 33% more work on Ultra (53 vs 40 layers)")
    print("    - 33% less work on Max (27 vs 40 layers)")
    print("    - Reduced risk of OOM on Max")
    print("    - Better overall throughput")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 80)
    print("MEMORY-AWARE PARTITIONING DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo shows how memory-aware partitioning improves resource")
    print("utilization in heterogeneous clusters with different memory capacities.")
    
    demo_homogeneous_cluster()
    demo_heterogeneous_cluster()
    demo_extreme_heterogeneity()
    demo_real_world_mac_studios()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n✅ Memory-Aware Partitioning Benefits:")
    print("  1. Better resource utilization in heterogeneous clusters")
    print("  2. Prevents overloading low-memory nodes")
    print("  3. Maximizes utilization of high-memory nodes")
    print("  4. Reduces OOM errors and improves stability")
    print("  5. Enables running large models on mixed hardware")
    
    print("\n📝 When to Use:")
    print("  - Homogeneous clusters: Both strategies work equally well")
    print("  - Heterogeneous clusters: Memory-aware provides significant benefits")
    print("  - Production deployments: Memory-aware recommended for mixed hardware")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
