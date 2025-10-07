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
        
        For now, falls back to balanced partitioning.
        TODO: Implement actual memory-aware partitioning.
        """
        logger.warning("Memory-aware partitioning not fully implemented, using balanced")
        return self._partition_balanced(total_layers, nodes)
    
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
