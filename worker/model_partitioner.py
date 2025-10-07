"""
Model Partitioner for Distributed Layer Inference

Splits model layers across multiple nodes for pipeline parallelism.
"""

import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PartitionStrategy(Enum):
    """Partitioning strategies"""
    BALANCED = "balanced"  # Equal layers per node
    MEMORY_AWARE = "memory_aware"  # Based on available memory
    CUSTOM = "custom"  # User-defined splits


@dataclass
class LayerAssignment:
    """Layer assignment for a node"""
    node_id: str
    start_layer: int
    end_layer: int  # Exclusive
    next_node_id: str = ""
    next_node_address: str = ""
    
    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer
    
    def __repr__(self) -> str:
        return f"LayerAssignment(node={self.node_id}, layers={self.start_layer}-{self.end_layer})"


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    pipeline_id: str
    model_name: str
    total_layers: int
    assignments: List[LayerAssignment]
    strategy: PartitionStrategy
    
    def get_assignment(self, node_id: str) -> LayerAssignment:
        """Get assignment for a specific node"""
        for assignment in self.assignments:
            if assignment.node_id == node_id:
                return assignment
        raise ValueError(f"No assignment found for node {node_id}")


class ModelPartitioner:
    """Partitions model layers across nodes"""
    
    def __init__(self):
        self._layer_memory_cache: Dict[str, List[int]] = {}
        logger.info("Initialized ModelPartitioner")
    
    def partition_model(
        self,
        model_name: str,
        total_layers: int,
        nodes: List[Dict[str, Any]],
        strategy: PartitionStrategy = PartitionStrategy.BALANCED
    ) -> PipelineConfig:
        """
        Partition model layers across nodes.
        
        Args:
            model_name: Name of the model
            total_layers: Total number of layers in the model
            nodes: List of node info dicts with 'node_id' and 'address'
            strategy: Partitioning strategy
            
        Returns:
            PipelineConfig with layer assignments
        """
        logger.info(f"Partitioning model {model_name} with {total_layers} layers across {len(nodes)} nodes")
        logger.info(f"Strategy: {strategy.value}")
        
        if strategy == PartitionStrategy.BALANCED:
            assignments = self._partition_balanced(total_layers, nodes)
        elif strategy == PartitionStrategy.MEMORY_AWARE:
            assignments = self._partition_memory_aware(total_layers, nodes)
        elif strategy == PartitionStrategy.CUSTOM:
            raise NotImplementedError("Custom partitioning not yet implemented")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Link assignments (set next_node info)
        for i, assignment in enumerate(assignments):
            if i < len(assignments) - 1:
                next_assignment = assignments[i + 1]
                assignment.next_node_id = next_assignment.node_id
                assignment.next_node_address = nodes[i + 1]['address']
        
        pipeline_config = PipelineConfig(
            pipeline_id=f"pipeline-{model_name}",
            model_name=model_name,
            total_layers=total_layers,
            assignments=assignments,
            strategy=strategy
        )
        
        logger.info(f"Created pipeline config: {pipeline_config.pipeline_id}")
        for assignment in assignments:
            logger.info(f"  {assignment}")
        
        return pipeline_config
    
    def _partition_balanced(
        self,
        total_layers: int,
        nodes: List[Dict[str, Any]]
    ) -> List[LayerAssignment]:
        """
        Partition layers evenly across nodes.
        
        Simple strategy: divide layers equally.
        """
        num_nodes = len(nodes)
        layers_per_node = total_layers // num_nodes
        remainder = total_layers % num_nodes
        
        assignments = []
        current_layer = 0
        
        for i, node in enumerate(nodes):
            # Distribute remainder layers to first nodes
            num_layers = layers_per_node + (1 if i < remainder else 0)
            
            assignment = LayerAssignment(
                node_id=node['node_id'],
                start_layer=current_layer,
                end_layer=current_layer + num_layers
            )
            assignments.append(assignment)
            
            current_layer += num_layers
        
        return assignments
    
    def _partition_memory_aware(
        self,
        total_layers: int,
        nodes: List[Dict[str, Any]]
    ) -> List[LayerAssignment]:
        """
        Partition layers based on available memory.
        
        Uses a bin-packing algorithm to distribute layers across nodes
        based on their available memory capacity. This is useful for
        heterogeneous clusters where nodes have different memory capacities.
        
        Args:
            total_layers: Total number of layers to partition
            nodes: List of node dicts with 'node_id', 'address', and 'available_memory' (in GB)
        
        Returns:
            List of LayerAssignment objects
        """
        # Validate that nodes have memory information
        for node in nodes:
            if 'available_memory' not in node:
                logger.warning(
                    f"Node {node['node_id']} missing 'available_memory', "
                    "falling back to balanced partitioning"
                )
                return self._partition_balanced(total_layers, nodes)
        
        # Get total available memory across all nodes
        total_memory = sum(node['available_memory'] for node in nodes)
        
        if total_memory <= 0:
            logger.error("Total available memory is zero, falling back to balanced partitioning")
            return self._partition_balanced(total_layers, nodes)
        
        # Calculate memory proportion for each node
        memory_proportions = [
            node['available_memory'] / total_memory
            for node in nodes
        ]
        
        # Allocate layers proportionally to memory
        assignments = []
        current_layer = 0
        
        for i, (node, proportion) in enumerate(zip(nodes, memory_proportions)):
            # Calculate number of layers for this node based on memory proportion
            if i == len(nodes) - 1:
                # Last node gets all remaining layers
                num_layers = total_layers - current_layer
            else:
                num_layers = max(1, int(total_layers * proportion))
                # Ensure we don't exceed total layers
                num_layers = min(num_layers, total_layers - current_layer - (len(nodes) - i - 1))
            
            assignment = LayerAssignment(
                node_id=node['node_id'],
                start_layer=current_layer,
                end_layer=current_layer + num_layers
            )
            assignments.append(assignment)
            
            logger.info(
                f"Assigned {num_layers} layers to {node['node_id']} "
                f"({node['available_memory']:.1f}GB memory, {proportion*100:.1f}% of total)"
            )
            
            current_layer += num_layers
        
        return assignments
    
    def get_layer_range_for_node(
        self,
        config: PipelineConfig,
        node_id: str
    ) -> Tuple[int, int]:
        """
        Get layer range for a specific node.
        
        Returns:
            Tuple of (start_layer, end_layer)
        """
        assignment = config.get_assignment(node_id)
        return (assignment.start_layer, assignment.end_layer)
    
    def is_first_node(self, config: PipelineConfig, node_id: str) -> bool:
        """Check if node is first in pipeline"""
        return config.assignments[0].node_id == node_id
    
    def is_last_node(self, config: PipelineConfig, node_id: str) -> bool:
        """Check if node is last in pipeline"""
        return config.assignments[-1].node_id == node_id
    
    def get_next_node(
        self,
        config: PipelineConfig,
        node_id: str
    ) -> Tuple[str, str]:
        """
        Get next node in pipeline.
        
        Returns:
            Tuple of (next_node_id, next_node_address) or ("", "") if last node
        """
        assignment = config.get_assignment(node_id)
        return (assignment.next_node_id, assignment.next_node_address)
    
    def _profile_layer_memory(self, model_name: str, layer_idx: int) -> int:
        """
        Profile memory usage of a single layer.
        
        This is a simplified estimation based on typical transformer layer sizes.
        In a production system, this would load and measure actual layer memory.
        
        Args:
            model_name: Name of the model
            layer_idx: Index of the layer to profile
            
        Returns:
            Estimated memory usage in bytes
        """
        # For now, use a simplified estimation
        # Typical transformer layer memory varies by position:
        # - Early layers (embeddings): larger
        # - Middle layers: consistent
        # - Final layers (output): larger
        
        # Base layer size (in MB)
        base_layer_mb = 100
        
        # Adjust for layer position
        if layer_idx == 0:
            # Embedding layer is typically larger
            layer_mb = base_layer_mb * 1.5
        elif layer_idx < 5:
            # Early layers slightly larger
            layer_mb = base_layer_mb * 1.2
        else:
            # Standard layer size
            layer_mb = base_layer_mb
        
        # Convert to bytes
        return int(layer_mb * 1024 * 1024)
    
    def profile_model_memory(self, model_name: str, total_layers: int) -> List[int]:
        """
        Profile memory requirements for all layers in a model.
        
        Args:
            model_name: Name of the model
            total_layers: Total number of layers
            
        Returns:
            List of memory requirements in bytes for each layer
        """
        cache_key = f"{model_name}_{total_layers}"
        
        # Check cache
        if cache_key in self._layer_memory_cache:
            logger.info(f"Using cached memory profile for {model_name}")
            return self._layer_memory_cache[cache_key]
        
        logger.info(f"Profiling memory for {model_name} with {total_layers} layers")
        
        # Profile each layer
        layer_memory = []
        for layer_idx in range(total_layers):
            mem_bytes = self._profile_layer_memory(model_name, layer_idx)
            layer_memory.append(mem_bytes)
        
        # Cache the results
        self._layer_memory_cache[cache_key] = layer_memory
        
        total_mb = sum(layer_memory) / (1024 * 1024)
        logger.info(f"Total estimated model memory: {total_mb:.1f}MB")
        
        return layer_memory
    
    def _partition_memory_aware_with_profiling(
        self,
        model_name: str,
        total_layers: int,
        nodes: List[Dict[str, Any]]
    ) -> List[LayerAssignment]:
        """
        Partition layers based on profiled memory requirements.
        
        Uses actual layer memory measurements to perform bin-packing
        across nodes with different memory capacities.
        
        Args:
            model_name: Name of the model
            total_layers: Total number of layers
            nodes: List of node dicts with 'node_id', 'address', and 'available_memory' (in GB)
            
        Returns:
            List of LayerAssignment objects
        """
        # Validate that nodes have memory information
        for node in nodes:
            if 'available_memory' not in node:
                logger.warning(
                    f"Node {node['node_id']} missing 'available_memory', "
                    "falling back to balanced partitioning"
                )
                return self._partition_balanced(total_layers, nodes)
        
        # Get layer memory requirements
        layer_memory = self.profile_model_memory(model_name, total_layers)
        
        # Convert node memory from GB to bytes
        node_memory_bytes = [int(node['available_memory'] * 1024 * 1024 * 1024) for node in nodes]
        
        # Bin packing algorithm
        assignments = []
        current_node = 0
        current_memory = 0
        current_layers = []
        
        for layer_idx, mem_required in enumerate(layer_memory):
            # Check if adding this layer would exceed current node's capacity
            if current_layers and current_memory + mem_required > node_memory_bytes[current_node]:
                # Create assignment for current node
                assignments.append(LayerAssignment(
                    node_id=nodes[current_node]['node_id'],
                    start_layer=current_layers[0],
                    end_layer=current_layers[-1] + 1
                ))
                
                logger.info(
                    f"Assigned layers {current_layers[0]}-{current_layers[-1]} to "
                    f"{nodes[current_node]['node_id']} "
                    f"({current_memory / (1024*1024):.1f}MB)"
                )
                
                # Move to next node
                current_node += 1
                if current_node >= len(nodes):
                    logger.error("Ran out of nodes during memory-aware partitioning")
                    # Fall back to balanced partitioning
                    return self._partition_balanced(total_layers, nodes)
                
                current_memory = 0
                current_layers = []
            
            # Add layer to current node
            current_memory += mem_required
            current_layers.append(layer_idx)
        
        # Add final assignment
        if current_layers:
            assignments.append(LayerAssignment(
                node_id=nodes[current_node]['node_id'],
                start_layer=current_layers[0],
                end_layer=current_layers[-1] + 1
            ))
            
            logger.info(
                f"Assigned layers {current_layers[0]}-{current_layers[-1]} to "
                f"{nodes[current_node]['node_id']} "
                f"({current_memory / (1024*1024):.1f}MB)"
            )
        
        return assignments


def create_pipeline_for_llama(
    model_name: str,
    node1_id: str,
    node1_address: str,
    node2_id: str,
    node2_address: str,
    total_layers: int = 32
) -> PipelineConfig:
    """
    Helper function to create a 2-node pipeline for Llama models.
    
    Args:
        model_name: Name of the Llama model
        node1_id: ID of first node
        node1_address: Address of first node
        node2_id: ID of second node
        node2_address: Address of second node
        total_layers: Total layers (default 32 for Llama-3.2-3B)
        
    Returns:
        PipelineConfig
    """
    partitioner = ModelPartitioner()
    
    nodes = [
        {'node_id': node1_id, 'address': node1_address},
        {'node_id': node2_id, 'address': node2_address}
    ]
    
    return partitioner.partition_model(
        model_name=model_name,
        total_layers=total_layers,
        nodes=nodes,
        strategy=PartitionStrategy.BALANCED
    )
