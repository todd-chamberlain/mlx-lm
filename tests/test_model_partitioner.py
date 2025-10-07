"""
Tests for ModelPartitioner

Tests balanced and memory-aware partitioning strategies.
"""

import pytest
from worker.model_partitioner import (
    ModelPartitioner,
    PartitionStrategy,
    LayerAssignment,
    PipelineConfig,
    create_pipeline_for_llama
)


class TestModelPartitioner:
    """Test ModelPartitioner functionality"""
    
    def test_balanced_partitioning_equal_split(self):
        """Test balanced partitioning with evenly divisible layers"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080'},
            {'node_id': 'node2', 'address': '192.168.1.11:8080'}
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        assert len(config.assignments) == 2
        assert config.assignments[0].start_layer == 0
        assert config.assignments[0].end_layer == 16
        assert config.assignments[0].num_layers == 16
        
        assert config.assignments[1].start_layer == 16
        assert config.assignments[1].end_layer == 32
        assert config.assignments[1].num_layers == 16
    
    def test_balanced_partitioning_uneven_split(self):
        """Test balanced partitioning with remainder layers"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080'},
            {'node_id': 'node2', 'address': '192.168.1.11:8080'},
            {'node_id': 'node3', 'address': '192.168.1.12:8080'}
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        assert len(config.assignments) == 3
        # 32 / 3 = 10 remainder 2, so first two nodes get 11 layers
        assert config.assignments[0].num_layers == 11
        assert config.assignments[1].num_layers == 11
        assert config.assignments[2].num_layers == 10
        
        # Verify continuity
        assert config.assignments[0].start_layer == 0
        assert config.assignments[0].end_layer == 11
        assert config.assignments[1].start_layer == 11
        assert config.assignments[1].end_layer == 22
        assert config.assignments[2].start_layer == 22
        assert config.assignments[2].end_layer == 32
    
    def test_memory_aware_partitioning_equal_memory(self):
        """Test memory-aware partitioning with equal memory nodes"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080', 'available_memory': 16.0},
            {'node_id': 'node2', 'address': '192.168.1.11:8080', 'available_memory': 16.0}
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.MEMORY_AWARE
        )
        
        assert len(config.assignments) == 2
        # With equal memory, should be similar to balanced
        assert config.assignments[0].num_layers == 16
        assert config.assignments[1].num_layers == 16
    
    def test_memory_aware_partitioning_heterogeneous(self):
        """Test memory-aware partitioning with different memory capacities"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080', 'available_memory': 32.0},  # 2x memory
            {'node_id': 'node2', 'address': '192.168.1.11:8080', 'available_memory': 16.0}
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.MEMORY_AWARE
        )
        
        assert len(config.assignments) == 2
        
        # Node1 has 2x memory, so should get ~2x layers
        # 32GB / 48GB total = 66.7%, 16GB / 48GB = 33.3%
        # So node1 should get ~21 layers, node2 should get ~11 layers
        assert config.assignments[0].num_layers > config.assignments[1].num_layers
        assert config.assignments[0].num_layers >= 20  # At least 20 layers
        assert config.assignments[1].num_layers <= 12  # At most 12 layers
        
        # Verify all layers are assigned
        total_assigned = sum(a.num_layers for a in config.assignments)
        assert total_assigned == 32
    
    def test_memory_aware_partitioning_three_nodes(self):
        """Test memory-aware partitioning with three heterogeneous nodes"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080', 'available_memory': 64.0},  # Large
            {'node_id': 'node2', 'address': '192.168.1.11:8080', 'available_memory': 32.0},  # Medium
            {'node_id': 'node3', 'address': '192.168.1.12:8080', 'available_memory': 16.0}   # Small
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.MEMORY_AWARE
        )
        
        assert len(config.assignments) == 3
        
        # Verify proportional allocation
        # 64GB / 112GB = 57%, 32GB / 112GB = 29%, 16GB / 112GB = 14%
        assert config.assignments[0].num_layers > config.assignments[1].num_layers
        assert config.assignments[1].num_layers > config.assignments[2].num_layers
        
        # Verify all layers are assigned
        total_assigned = sum(a.num_layers for a in config.assignments)
        assert total_assigned == 32
        
        # Verify continuity
        assert config.assignments[0].start_layer == 0
        assert config.assignments[1].start_layer == config.assignments[0].end_layer
        assert config.assignments[2].start_layer == config.assignments[1].end_layer
        assert config.assignments[2].end_layer == 32
    
    def test_memory_aware_fallback_no_memory_info(self):
        """Test that memory-aware falls back to balanced when memory info missing"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080'},  # No memory info
            {'node_id': 'node2', 'address': '192.168.1.11:8080'}
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.MEMORY_AWARE
        )
        
        # Should fall back to balanced partitioning
        assert len(config.assignments) == 2
        assert config.assignments[0].num_layers == 16
        assert config.assignments[1].num_layers == 16
    
    def test_layer_memory_profiling(self):
        """Test layer memory profiling"""
        partitioner = ModelPartitioner()
        
        layer_memory = partitioner.profile_model_memory('test-model', 32)
        
        assert len(layer_memory) == 32
        assert all(mem > 0 for mem in layer_memory)
        
        # First layer (embedding) should be larger
        assert layer_memory[0] > layer_memory[10]
        
        # Total memory should be reasonable
        total_mb = sum(layer_memory) / (1024 * 1024)
        assert 1000 < total_mb < 10000  # Between 1GB and 10GB
    
    def test_layer_memory_profiling_caching(self):
        """Test that layer memory profiling results are cached"""
        partitioner = ModelPartitioner()
        
        # First call
        layer_memory1 = partitioner.profile_model_memory('test-model', 32)
        
        # Second call should use cache
        layer_memory2 = partitioner.profile_model_memory('test-model', 32)
        
        assert layer_memory1 == layer_memory2
        assert id(layer_memory1) == id(layer_memory2)  # Same object
    
    def test_memory_aware_with_profiling(self):
        """Test memory-aware partitioning with profiling"""
        partitioner = ModelPartitioner()
        
        # Use smaller memory to force distribution across nodes
        # Each layer is ~100MB, so 32 layers = ~3.2GB
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080', 'available_memory': 2.0},  # Can fit ~20 layers
            {'node_id': 'node2', 'address': '192.168.1.11:8080', 'available_memory': 2.0}   # Can fit ~20 layers
        ]
        
        # Use the profiling-based method
        assignments = partitioner._partition_memory_aware_with_profiling(
            model_name='test-model',
            total_layers=32,
            nodes=nodes
        )
        
        # Should use both nodes
        assert len(assignments) >= 1
        
        # Verify all layers are assigned
        total_assigned = sum(a.num_layers for a in assignments)
        assert total_assigned == 32
        
        # Verify continuity
        for i in range(len(assignments) - 1):
            assert assignments[i].end_layer == assignments[i + 1].start_layer
    
    def test_pipeline_config_get_assignment(self):
        """Test getting assignment for a specific node"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080'},
            {'node_id': 'node2', 'address': '192.168.1.11:8080'}
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        assignment = config.get_assignment('node1')
        assert assignment.node_id == 'node1'
        assert assignment.start_layer == 0
        assert assignment.end_layer == 16
        
        with pytest.raises(ValueError):
            config.get_assignment('nonexistent')
    
    def test_get_layer_range_for_node(self):
        """Test getting layer range for a node"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080'},
            {'node_id': 'node2', 'address': '192.168.1.11:8080'}
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        start, end = partitioner.get_layer_range_for_node(config, 'node1')
        assert start == 0
        assert end == 16
        
        start, end = partitioner.get_layer_range_for_node(config, 'node2')
        assert start == 16
        assert end == 32
    
    def test_is_first_node(self):
        """Test checking if node is first in pipeline"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080'},
            {'node_id': 'node2', 'address': '192.168.1.11:8080'}
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        assert partitioner.is_first_node(config, 'node1') is True
        assert partitioner.is_first_node(config, 'node2') is False
    
    def test_is_last_node(self):
        """Test checking if node is last in pipeline"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080'},
            {'node_id': 'node2', 'address': '192.168.1.11:8080'}
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        assert partitioner.is_last_node(config, 'node1') is False
        assert partitioner.is_last_node(config, 'node2') is True
    
    def test_get_next_node(self):
        """Test getting next node in pipeline"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080'},
            {'node_id': 'node2', 'address': '192.168.1.11:8080'}
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        next_id, next_addr = partitioner.get_next_node(config, 'node1')
        assert next_id == 'node2'
        assert next_addr == '192.168.1.11:8080'
        
        next_id, next_addr = partitioner.get_next_node(config, 'node2')
        assert next_id == ''
        assert next_addr == ''
    
    def test_create_pipeline_for_llama(self):
        """Test helper function for creating Llama pipeline"""
        config = create_pipeline_for_llama(
            model_name='llama-3.2-3b',
            node1_id='node1',
            node1_address='192.168.1.10:8080',
            node2_id='node2',
            node2_address='192.168.1.11:8080',
            total_layers=32
        )
        
        assert config.model_name == 'llama-3.2-3b'
        assert config.total_layers == 32
        assert len(config.assignments) == 2
        assert config.strategy == PartitionStrategy.BALANCED
    
    def test_next_node_linking(self):
        """Test that assignments are properly linked"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080'},
            {'node_id': 'node2', 'address': '192.168.1.11:8080'},
            {'node_id': 'node3', 'address': '192.168.1.12:8080'}
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        # Check linking
        assert config.assignments[0].next_node_id == 'node2'
        assert config.assignments[0].next_node_address == '192.168.1.11:8080'
        
        assert config.assignments[1].next_node_id == 'node3'
        assert config.assignments[1].next_node_address == '192.168.1.12:8080'
        
        assert config.assignments[2].next_node_id == ''
        assert config.assignments[2].next_node_address == ''
    
    def test_memory_aware_minimum_layers_per_node(self):
        """Test that each node gets at least one layer"""
        partitioner = ModelPartitioner()
        
        nodes = [
            {'node_id': 'node1', 'address': '192.168.1.10:8080', 'available_memory': 100.0},
            {'node_id': 'node2', 'address': '192.168.1.11:8080', 'available_memory': 1.0}
        ]
        
        config = partitioner.partition_model(
            model_name='test-model',
            total_layers=10,
            nodes=nodes,
            strategy=PartitionStrategy.MEMORY_AWARE
        )
        
        # Even with very different memory, each node should get at least 1 layer
        assert all(a.num_layers >= 1 for a in config.assignments)
        
        # Verify all layers are assigned
        total_assigned = sum(a.num_layers for a in config.assignments)
        assert total_assigned == 10


class TestLayerAssignment:
    """Test LayerAssignment dataclass"""
    
    def test_num_layers_property(self):
        """Test num_layers property calculation"""
        assignment = LayerAssignment(
            node_id='node1',
            start_layer=0,
            end_layer=16
        )
        
        assert assignment.num_layers == 16
    
    def test_repr(self):
        """Test string representation"""
        assignment = LayerAssignment(
            node_id='node1',
            start_layer=0,
            end_layer=16
        )
        
        repr_str = repr(assignment)
        assert 'node1' in repr_str
        assert '0' in repr_str
        assert '16' in repr_str


class TestPipelineConfig:
    """Test PipelineConfig dataclass"""
    
    def test_get_assignment_success(self):
        """Test successful assignment retrieval"""
        assignment1 = LayerAssignment(node_id='node1', start_layer=0, end_layer=16)
        assignment2 = LayerAssignment(node_id='node2', start_layer=16, end_layer=32)
        
        config = PipelineConfig(
            pipeline_id='test-pipeline',
            model_name='test-model',
            total_layers=32,
            assignments=[assignment1, assignment2],
            strategy=PartitionStrategy.BALANCED
        )
        
        assert config.get_assignment('node1') == assignment1
        assert config.get_assignment('node2') == assignment2
    
    def test_get_assignment_not_found(self):
        """Test assignment retrieval for non-existent node"""
        assignment = LayerAssignment(node_id='node1', start_layer=0, end_layer=16)
        
        config = PipelineConfig(
            pipeline_id='test-pipeline',
            model_name='test-model',
            total_layers=32,
            assignments=[assignment],
            strategy=PartitionStrategy.BALANCED
        )
        
        with pytest.raises(ValueError, match="No assignment found"):
            config.get_assignment('nonexistent')
