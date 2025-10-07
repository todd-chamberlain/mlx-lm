"""
Comparison tests for balanced vs memory-aware partitioning

Demonstrates the benefits of memory-aware partitioning for heterogeneous clusters.
"""

import pytest
from worker.model_partitioner import ModelPartitioner, PartitionStrategy


class TestPartitioningComparison:
    """Compare balanced vs memory-aware partitioning strategies"""
    
    def test_homogeneous_cluster_similar_results(self):
        """Test that both strategies give similar results for homogeneous clusters"""
        partitioner = ModelPartitioner()
        
        # Homogeneous cluster - all nodes have same memory
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080', 'available_memory': 32.0},
            {'node_id': 'node2', 'address': '192.168.1.11:8080', 'available_memory': 32.0},
            {'node_id': 'node3', 'address': '192.168.1.12:8080', 'available_memory': 32.0}
        ]
        
        balanced_config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        memory_aware_config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.MEMORY_AWARE
        )
        
        # For homogeneous clusters, results should be very similar
        balanced_layers = [a.num_layers for a in balanced_config.assignments]
        memory_aware_layers = [a.num_layers for a in memory_aware_config.assignments]
        
        # Total should be the same
        assert sum(balanced_layers) == sum(memory_aware_layers) == 32
        
        # Distribution should be similar (within 2 layers per node)
        for i in range(len(nodes)):
            assert abs(balanced_layers[i] - memory_aware_layers[i]) <= 2
    
    def test_heterogeneous_cluster_different_results(self):
        """Test that memory-aware gives better distribution for heterogeneous clusters"""
        partitioner = ModelPartitioner()
        
        # Heterogeneous cluster - nodes have very different memory
        nodes = [
            {'node_id': 'mac-studio', 'address': '192.168.1.10:8080', 'available_memory': 128.0},  # Large
            {'node_id': 'mac-mini', 'address': '192.168.1.11:8080', 'available_memory': 16.0},     # Small
            {'node_id': 'macbook', 'address': '192.168.1.12:8080', 'available_memory': 32.0}       # Medium
        ]
        
        balanced_config = partitioner.partition_model(
            model_name='test-model',
            total_layers=48,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        memory_aware_config = partitioner.partition_model(
            model_name='test-model',
            total_layers=48,
            nodes=nodes,
            strategy=PartitionStrategy.MEMORY_AWARE
        )
        
        # Balanced: Each node gets 16 layers (48/3)
        assert balanced_config.assignments[0].num_layers == 16  # Mac Studio
        assert balanced_config.assignments[1].num_layers == 16  # Mac Mini
        assert balanced_config.assignments[2].num_layers == 16  # MacBook
        
        # Memory-aware: Distribution based on memory capacity
        # 128GB / 176GB = 72.7%, 16GB / 176GB = 9.1%, 32GB / 176GB = 18.2%
        # So approximately: 35 layers, 4 layers, 9 layers
        
        mac_studio_layers = memory_aware_config.assignments[0].num_layers
        mac_mini_layers = memory_aware_config.assignments[1].num_layers
        macbook_layers = memory_aware_config.assignments[2].num_layers
        
        # Mac Studio should get significantly more layers
        assert mac_studio_layers > 30
        
        # Mac Mini should get fewer layers
        assert mac_mini_layers < 10
        
        # MacBook should be in between
        assert 8 <= macbook_layers <= 12
        
        # All layers should be assigned
        assert mac_studio_layers + mac_mini_layers + macbook_layers == 48
    
    def test_memory_aware_prevents_overload(self):
        """Test that memory-aware prevents overloading small-memory nodes"""
        partitioner = ModelPartitioner()
        
        # Cluster with one very small node
        nodes = [
            {'node_id': 'powerful', 'address': '192.168.1.10:8080', 'available_memory': 64.0},
            {'node_id': 'weak', 'address': '192.168.1.11:8080', 'available_memory': 8.0}
        ]
        
        balanced_config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        memory_aware_config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.MEMORY_AWARE
        )
        
        # Balanced gives equal layers (16 each)
        assert balanced_config.assignments[0].num_layers == 16
        assert balanced_config.assignments[1].num_layers == 16
        
        # Memory-aware gives proportional layers
        # 64GB / 72GB = 88.9%, 8GB / 72GB = 11.1%
        # So approximately: 28 layers, 4 layers
        
        powerful_layers = memory_aware_config.assignments[0].num_layers
        weak_layers = memory_aware_config.assignments[1].num_layers
        
        # Powerful node should get most layers
        assert powerful_layers >= 28
        
        # Weak node should get fewer layers
        assert weak_layers <= 4
        
        # This prevents overloading the weak node
        assert powerful_layers > balanced_config.assignments[0].num_layers
        assert weak_layers < balanced_config.assignments[1].num_layers
    
    def test_memory_aware_utilization_efficiency(self):
        """Test that memory-aware provides better memory utilization"""
        partitioner = ModelPartitioner()
        
        # Heterogeneous cluster
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080', 'available_memory': 96.0},
            {'node_id': 'node2', 'address': '192.168.1.11:8080', 'available_memory': 48.0},
            {'node_id': 'node3', 'address': '192.168.1.12:8080', 'available_memory': 24.0}
        ]
        
        total_memory = sum(n['available_memory'] for n in nodes)
        
        memory_aware_config = partitioner.partition_model(
            model_name='test-model',
            total_layers=48,
            nodes=nodes,
            strategy=PartitionStrategy.MEMORY_AWARE
        )
        
        # Calculate memory utilization for each node
        # Assuming each layer uses roughly equal memory
        layer_memory = 100  # MB per layer (simplified)
        
        for i, assignment in enumerate(memory_aware_config.assignments):
            node_memory_gb = nodes[i]['available_memory']
            node_memory_mb = node_memory_gb * 1024
            
            layers_assigned = assignment.num_layers
            memory_used_mb = layers_assigned * layer_memory
            
            utilization = memory_used_mb / node_memory_mb
            
            # Each node should have reasonable utilization
            # (not too high to avoid OOM, not too low to waste resources)
            assert 0.01 < utilization < 0.8
    
    def test_real_world_scenario_mac_studios(self):
        """Test realistic scenario with two Mac Studios"""
        partitioner = ModelPartitioner()
        
        # Two Mac Studios with different configurations
        nodes = [
            {
                'node_id': 'mac-studio-m2-ultra',
                'address': '192.168.1.10:8080',
                'available_memory': 192.0  # M2 Ultra with 192GB
            },
            {
                'node_id': 'mac-studio-m2-max',
                'address': '192.168.1.11:8080',
                'available_memory': 96.0   # M2 Max with 96GB
            }
        ]
        
        # Large model with 80 layers (e.g., Llama-70B)
        balanced_config = partitioner.partition_model(
            model_name='llama-70b',
            total_layers=80,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        memory_aware_config = partitioner.partition_model(
            model_name='llama-70b',
            total_layers=80,
            nodes=nodes,
            strategy=PartitionStrategy.MEMORY_AWARE
        )
        
        # Balanced: 40 layers each
        assert balanced_config.assignments[0].num_layers == 40
        assert balanced_config.assignments[1].num_layers == 40
        
        # Memory-aware: Proportional to memory
        # 192GB / 288GB = 66.7%, 96GB / 288GB = 33.3%
        # So approximately: 53 layers, 27 layers
        
        ultra_layers = memory_aware_config.assignments[0].num_layers
        max_layers = memory_aware_config.assignments[1].num_layers
        
        assert ultra_layers >= 50
        assert max_layers <= 30
        assert ultra_layers + max_layers == 80
        
        # Memory-aware should better utilize the Ultra's extra memory
        assert ultra_layers > balanced_config.assignments[0].num_layers
    
    def test_edge_case_single_node(self):
        """Test that both strategies work with single node"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080', 'available_memory': 64.0}
        ]
        
        balanced_config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        memory_aware_config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.MEMORY_AWARE
        )
        
        # Both should assign all layers to the single node
        assert len(balanced_config.assignments) == 1
        assert len(memory_aware_config.assignments) == 1
        assert balanced_config.assignments[0].num_layers == 32
        assert memory_aware_config.assignments[0].num_layers == 32
    
    def test_performance_comparison_summary(self):
        """Generate a summary comparing both strategies"""
        partitioner = ModelPartitioner()
        
        # Various cluster configurations
        test_cases = [
            {
                'name': 'Homogeneous (3x 32GB)',
                'nodes': [
                    {'node_id': 'n1', 'address': 'addr1', 'available_memory': 32.0},
                    {'node_id': 'n2', 'address': 'addr2', 'available_memory': 32.0},
                    {'node_id': 'n3', 'address': 'addr3', 'available_memory': 32.0}
                ]
            },
            {
                'name': 'Heterogeneous (128GB, 64GB, 32GB)',
                'nodes': [
                    {'node_id': 'n1', 'address': 'addr1', 'available_memory': 128.0},
                    {'node_id': 'n2', 'address': 'addr2', 'available_memory': 64.0},
                    {'node_id': 'n3', 'address': 'addr3', 'available_memory': 32.0}
                ]
            },
            {
                'name': 'Very Heterogeneous (192GB, 16GB)',
                'nodes': [
                    {'node_id': 'n1', 'address': 'addr1', 'available_memory': 192.0},
                    {'node_id': 'n2', 'address': 'addr2', 'available_memory': 16.0}
                ]
            }
        ]
        
        for test_case in test_cases:
            balanced = partitioner.partition_model(
                model_name='test',
                total_layers=48,
                nodes=test_case['nodes'],
                strategy=PartitionStrategy.BALANCED
            )
            
            memory_aware = partitioner.partition_model(
                model_name='test',
                total_layers=48,
                nodes=test_case['nodes'],
                strategy=PartitionStrategy.MEMORY_AWARE
            )
            
            # Verify both strategies assign all layers
            balanced_total = sum(a.num_layers for a in balanced.assignments)
            memory_aware_total = sum(a.num_layers for a in memory_aware.assignments)
            
            assert balanced_total == 48
            assert memory_aware_total == 48
            
            # For heterogeneous clusters, memory-aware should show different distribution
            if 'Heterogeneous' in test_case['name']:
                # Check that distributions are different
                balanced_dist = [a.num_layers for a in balanced.assignments]
                memory_aware_dist = [a.num_layers for a in memory_aware.assignments]
                
                # At least one node should have different layer count
                assert balanced_dist != memory_aware_dist
