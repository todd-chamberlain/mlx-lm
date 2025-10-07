"""
Pipeline Manager for Distributed Inference

Manages the complete lifecycle of distributed inference pipelines including
initialization, execution, and teardown.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from worker.model_partitioner import ModelPartitioner, PartitionStrategy, LayerAssignment
from .pipeline_coordinator import PipelineCoordinator, NodeInfo, PipelineConfig


logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline lifecycle states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class PipelineSetupResult:
    """Result of pipeline setup"""
    success: bool
    pipeline_id: str
    config: Optional[PipelineConfig] = None
    assignments: Optional[Dict[str, LayerAssignment]] = None
    error: Optional[str] = None
    setup_time: float = 0.0


class PipelineManager:
    """
    Manages distributed inference pipelines.
    
    Responsibilities:
    - Pipeline initialization and setup
    - Model partitioning coordination
    - Node assignment and verification
    - Pipeline health monitoring
    - Pipeline teardown and cleanup
    """
    
    def __init__(self, coordinator: PipelineCoordinator):
        """
        Initialize pipeline manager.
        
        Args:
            coordinator: Pipeline coordinator instance
        """
        self.coordinator = coordinator
        self.pipelines: Dict[str, PipelineState] = {}
        self.pipeline_configs: Dict[str, PipelineConfig] = {}
        self.pipeline_assignments: Dict[str, Dict[str, LayerAssignment]] = {}
        
        logger.info("PipelineManager initialized")
    
    async def setup_pipeline(
        self,
        pipeline_id: str,
        model_path: str,
        partition_strategy: PartitionStrategy = PartitionStrategy.BALANCED,
        node_capacities: Optional[Dict[str, float]] = None,
        verify_nodes: bool = True
    ) -> PipelineSetupResult:
        """
        Setup a complete distributed inference pipeline.
        
        This is the main entry point for pipeline initialization. It:
        1. Validates registered nodes
        2. Partitions the model across nodes
        3. Assigns layers to each node
        4. Verifies node readiness
        5. Establishes inter-node connections
        
        Args:
            pipeline_id: Unique identifier for this pipeline
            model_path: Path to the model to load
            partition_strategy: Strategy for partitioning model
            node_capacities: Optional node memory capacities (for memory-aware partitioning)
            verify_nodes: Whether to verify node readiness
            
        Returns:
            PipelineSetupResult with setup status and details
        """
        import time
        start_time = time.time()
        
        logger.info(f"Setting up pipeline '{pipeline_id}' for model: {model_path}")
        logger.info(f"Partition strategy: {partition_strategy.value}")
        
        try:
            # Mark pipeline as initializing
            self.pipelines[pipeline_id] = PipelineState.INITIALIZING
            
            # Step 1: Validate registered nodes
            logger.info("Step 1: Validating registered nodes...")
            nodes = self.coordinator.get_registered_nodes()
            
            if not nodes:
                error_msg = "No nodes registered with coordinator"
                logger.error(error_msg)
                self.pipelines[pipeline_id] = PipelineState.ERROR
                return PipelineSetupResult(
                    success=False,
                    pipeline_id=pipeline_id,
                    error=error_msg,
                    setup_time=time.time() - start_time
                )
            
            num_nodes = len(nodes)
            logger.info(f"Found {num_nodes} registered nodes")
            
            # Verify all nodes are healthy
            if verify_nodes:
                unhealthy_nodes = []
                for node_id, node_info in nodes.items():
                    if node_info.status != "healthy":
                        unhealthy_nodes.append(node_id)
                
                if unhealthy_nodes:
                    error_msg = f"Unhealthy nodes detected: {unhealthy_nodes}"
                    logger.error(error_msg)
                    self.pipelines[pipeline_id] = PipelineState.ERROR
                    return PipelineSetupResult(
                        success=False,
                        pipeline_id=pipeline_id,
                        error=error_msg,
                        setup_time=time.time() - start_time
                    )
            
            # Step 2: Partition model across nodes
            logger.info("Step 2: Partitioning model across nodes...")
            partitioner = ModelPartitioner(
                model_path=model_path,
                num_nodes=num_nodes
            )
            
            # Perform partitioning
            if partition_strategy == PartitionStrategy.MEMORY_AWARE:
                if node_capacities is None:
                    # Use node capacities from registration if available
                    node_capacities = {
                        node_id: node_info.memory_gb
                        for node_id, node_info in nodes.items()
                        if node_info.memory_gb > 0
                    }
                
                if not node_capacities:
                    logger.warning("No node capacities available, falling back to balanced partitioning")
                    partition_strategy = PartitionStrategy.BALANCED
            
            assignments = partitioner.partition(
                strategy=partition_strategy,
                node_capacities=node_capacities
            )
            
            logger.info(f"Model partitioned into {len(assignments)} assignments")
            for i, assignment in enumerate(assignments):
                logger.info(
                    f"  Assignment {i}: layers {assignment.start_layer}-{assignment.end_layer} "
                    f"({assignment.num_layers} layers, {assignment.estimated_memory_gb:.2f}GB)"
                )
            
            # Step 3: Assign layers to specific nodes
            logger.info("Step 3: Assigning layers to nodes...")
            node_assignments = {}
            node_list = list(nodes.keys())
            
            for i, assignment in enumerate(assignments):
                if i >= len(node_list):
                    error_msg = f"Not enough nodes for {len(assignments)} assignments"
                    logger.error(error_msg)
                    self.pipelines[pipeline_id] = PipelineState.ERROR
                    return PipelineSetupResult(
                        success=False,
                        pipeline_id=pipeline_id,
                        error=error_msg,
                        setup_time=time.time() - start_time
                    )
                
                node_id = node_list[i]
                assignment.node_id = node_id
                node_assignments[node_id] = assignment
                
                logger.info(f"  Assigned layers {assignment.start_layer}-{assignment.end_layer} to {node_id}")
            
            # Step 4: Create pipeline configuration
            logger.info("Step 4: Creating pipeline configuration...")
            pipeline_config = PipelineConfig(
                pipeline_id=pipeline_id,
                model_path=model_path,
                num_nodes=num_nodes,
                partition_strategy=partition_strategy.value,
                node_assignments=node_assignments
            )
            
            # Step 5: Send configuration to coordinator
            logger.info("Step 5: Registering pipeline with coordinator...")
            success = await self.coordinator.setup_pipeline(pipeline_config)
            
            if not success:
                error_msg = "Failed to register pipeline with coordinator"
                logger.error(error_msg)
                self.pipelines[pipeline_id] = PipelineState.ERROR
                return PipelineSetupResult(
                    success=False,
                    pipeline_id=pipeline_id,
                    error=error_msg,
                    setup_time=time.time() - start_time
                )
            
            # Step 6: Verify pipeline setup
            if verify_nodes:
                logger.info("Step 6: Verifying pipeline setup...")
                verification_result = await self._verify_pipeline_setup(
                    pipeline_id,
                    node_assignments
                )
                
                if not verification_result:
                    error_msg = "Pipeline verification failed"
                    logger.error(error_msg)
                    self.pipelines[pipeline_id] = PipelineState.ERROR
                    return PipelineSetupResult(
                        success=False,
                        pipeline_id=pipeline_id,
                        error=error_msg,
                        setup_time=time.time() - start_time
                    )
            
            # Step 7: Mark pipeline as ready
            self.pipelines[pipeline_id] = PipelineState.READY
            self.pipeline_configs[pipeline_id] = pipeline_config
            self.pipeline_assignments[pipeline_id] = node_assignments
            
            setup_time = time.time() - start_time
            logger.info(f"Pipeline '{pipeline_id}' setup complete in {setup_time:.2f}s")
            
            return PipelineSetupResult(
                success=True,
                pipeline_id=pipeline_id,
                config=pipeline_config,
                assignments=node_assignments,
                setup_time=setup_time
            )
            
        except Exception as e:
            error_msg = f"Pipeline setup failed: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            self.pipelines[pipeline_id] = PipelineState.ERROR
            
            return PipelineSetupResult(
                success=False,
                pipeline_id=pipeline_id,
                error=error_msg,
                setup_time=time.time() - start_time
            )
    
    async def _verify_pipeline_setup(
        self,
        pipeline_id: str,
        node_assignments: Dict[str, LayerAssignment]
    ) -> bool:
        """
        Verify that all nodes in the pipeline are ready.
        
        Args:
            pipeline_id: Pipeline identifier
            node_assignments: Node layer assignments
            
        Returns:
            True if all nodes are ready, False otherwise
        """
        logger.info(f"Verifying pipeline setup for '{pipeline_id}'...")
        
        # Check each node
        for node_id, assignment in node_assignments.items():
            # Get node info from coordinator
            nodes = self.coordinator.get_registered_nodes()
            
            if node_id not in nodes:
                logger.error(f"Node {node_id} not found in registered nodes")
                return False
            
            node_info = nodes[node_id]
            
            # Check node health
            if node_info.status != "healthy":
                logger.error(f"Node {node_id} is not healthy: {node_info.status}")
                return False
            
            # TODO: Verify node has loaded assigned layers
            # This would require adding a gRPC call to query node state
            
            logger.debug(f"Node {node_id} verified: layers {assignment.start_layer}-{assignment.end_layer}")
        
        logger.info("Pipeline verification complete - all nodes ready")
        return True
    
    async def teardown_pipeline(self, pipeline_id: str) -> bool:
        """
        Teardown a pipeline and cleanup resources.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            True if teardown successful, False otherwise
        """
        logger.info(f"Tearing down pipeline '{pipeline_id}'...")
        
        if pipeline_id not in self.pipelines:
            logger.warning(f"Pipeline '{pipeline_id}' not found")
            return False
        
        try:
            # Mark as shutting down
            self.pipelines[pipeline_id] = PipelineState.SHUTDOWN
            
            # Remove from coordinator
            # TODO: Add coordinator method to remove pipeline
            
            # Cleanup local state
            if pipeline_id in self.pipeline_configs:
                del self.pipeline_configs[pipeline_id]
            
            if pipeline_id in self.pipeline_assignments:
                del self.pipeline_assignments[pipeline_id]
            
            del self.pipelines[pipeline_id]
            
            logger.info(f"Pipeline '{pipeline_id}' teardown complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to teardown pipeline '{pipeline_id}': {e}")
            return False
    
    def get_pipeline_state(self, pipeline_id: str) -> Optional[PipelineState]:
        """Get current state of a pipeline"""
        return self.pipelines.get(pipeline_id)
    
    def get_pipeline_config(self, pipeline_id: str) -> Optional[PipelineConfig]:
        """Get configuration of a pipeline"""
        return self.pipeline_configs.get(pipeline_id)
    
    def get_pipeline_assignments(self, pipeline_id: str) -> Optional[Dict[str, LayerAssignment]]:
        """Get node assignments for a pipeline"""
        return self.pipeline_assignments.get(pipeline_id)
    
    def list_pipelines(self) -> Dict[str, PipelineState]:
        """List all pipelines and their states"""
        return self.pipelines.copy()
    
    def get_pipeline_info(self, pipeline_id: str) -> Optional[Dict]:
        """Get comprehensive information about a pipeline"""
        if pipeline_id not in self.pipelines:
            return None
        
        return {
            'pipeline_id': pipeline_id,
            'state': self.pipelines[pipeline_id].value,
            'config': self.pipeline_configs.get(pipeline_id),
            'assignments': self.pipeline_assignments.get(pipeline_id),
            'num_nodes': len(self.pipeline_assignments.get(pipeline_id, {}))
        }
