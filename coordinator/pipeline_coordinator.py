"""
Pipeline Coordinator for Distributed Layer-Level Inference

Manages pipeline setup, node registration, and health monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from proto import distributed_pb2
from worker.model_partitioner import ModelPartitioner, PartitionStrategy

logger = logging.getLogger(__name__)


@dataclass
class PipelineNode:
    """Node in the inference pipeline."""
    node_id: str
    address: str
    start_layer: int
    end_layer: int
    next_node_id: Optional[str] = None
    next_node_address: Optional[str] = None
    is_healthy: bool = True
    last_heartbeat: datetime = field(default_factory=datetime.now)
    memory_used_bytes: int = 0


@dataclass
class Pipeline:
    """Complete inference pipeline."""
    pipeline_id: str
    model_name: str
    total_layers: int
    nodes: Dict[str, PipelineNode] = field(default_factory=dict)
    strategy: PartitionStrategy = PartitionStrategy.BALANCED
    status: str = "INITIALIZING"
    created_at: datetime = field(default_factory=datetime.now)


class PipelineCoordinator:
    """Coordinates distributed inference pipelines."""
    
    def __init__(self, heartbeat_timeout: float = 30.0):
        self.pipelines: Dict[str, Pipeline] = {}
        self.partitioner = ModelPartitioner()
        self.heartbeat_timeout = heartbeat_timeout
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        logger.info("Initialized PipelineCoordinator")
    
    async def start(self):
        """Start the coordinator."""
        logger.info("Starting PipelineCoordinator")
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self):
        """Stop the coordinator."""
        logger.info("Stopping PipelineCoordinator")
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

    async def create_pipeline(
        self,
        pipeline_id: str,
        model_name: str,
        node_addresses: List[str],
        strategy: PartitionStrategy = PartitionStrategy.BALANCED,
        node_capabilities: Optional[Dict[str, Dict]] = None
    ) -> Pipeline:
        """Create a new inference pipeline."""
        async with self._lock:
            if pipeline_id in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_id} already exists")
            
            if not node_addresses:
                raise ValueError("At least one node address required")
            
            logger.info(f"Creating pipeline {pipeline_id} for model {model_name}")
            
            # Get partition configuration
            pipeline_config = self.partitioner.partition_model(
                model_name=model_name,
                num_nodes=len(node_addresses),
                strategy=strategy,
                node_capabilities=node_capabilities
            )
            
            # Create pipeline
            pipeline = Pipeline(
                pipeline_id=pipeline_id,
                model_name=model_name,
                total_layers=pipeline_config.total_layers,
                strategy=strategy,
                status="INITIALIZING"
            )
            
            # Create nodes from assignments
            for i, assignment in enumerate(pipeline_config.assignments):
                node_id = assignment.node_id
                address = node_addresses[i] if i < len(node_addresses) else f"node-{i}"
                
                # Determine next node
                next_node_id = None
                next_node_address = None
                if i < len(pipeline_config.assignments) - 1:
                    next_assignment = pipeline_config.assignments[i + 1]
                    next_node_id = next_assignment.node_id
                    next_node_address = node_addresses[i + 1] if i + 1 < len(node_addresses) else f"node-{i+1}"
                
                node = PipelineNode(
                    node_id=node_id,
                    address=address,
                    start_layer=assignment.start_layer,
                    end_layer=assignment.end_layer,
                    next_node_id=next_node_id,
                    next_node_address=next_node_address
                )
                
                pipeline.nodes[node_id] = node
            
            self.pipelines[pipeline_id] = pipeline
            logger.info(f"Created pipeline {pipeline_id} with {len(pipeline.nodes)} nodes")
            
            return pipeline
    
    async def register_node(self, pipeline_id: str, node_id: str, address: str) -> bool:
        """Register a node in a pipeline."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                logger.error(f"Pipeline {pipeline_id} not found")
                return False
            
            pipeline = self.pipelines[pipeline_id]
            
            if node_id not in pipeline.nodes:
                logger.error(f"Node {node_id} not part of pipeline {pipeline_id}")
                return False
            
            node = pipeline.nodes[node_id]
            node.address = address
            node.is_healthy = True
            node.last_heartbeat = datetime.now()
            
            logger.info(f"Registered node {node_id} for pipeline {pipeline_id}")
            
            # Check if all nodes are registered
            if all(n.is_healthy for n in pipeline.nodes.values()):
                pipeline.status = "READY"
                logger.info(f"Pipeline {pipeline_id} is now READY")
            
            return True
    
    async def update_node_heartbeat(
        self, pipeline_id: str, node_id: str, memory_used_bytes: int = 0
    ) -> bool:
        """Update node heartbeat."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return False
            
            pipeline = self.pipelines[pipeline_id]
            
            if node_id not in pipeline.nodes:
                return False
            
            node = pipeline.nodes[node_id]
            node.last_heartbeat = datetime.now()
            node.memory_used_bytes = memory_used_bytes
            node.is_healthy = True
            
            return True
    
    async def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline by ID."""
        return self.pipelines.get(pipeline_id)
    
    async def list_pipelines(self) -> List[Pipeline]:
        """List all pipelines."""
        return list(self.pipelines.values())
    
    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return False
            
            del self.pipelines[pipeline_id]
            logger.info(f"Deleted pipeline {pipeline_id}")
            return True
    
    async def get_pipeline_health(
        self, pipeline_id: str
    ) -> Optional[distributed_pb2.PipelineHealthResponse]:
        """Get pipeline health status."""
        pipeline = await self.get_pipeline(pipeline_id)
        if not pipeline:
            return None
        
        # Convert status to proto enum
        status_map = {
            "INITIALIZING": distributed_pb2.INITIALIZING,
            "READY": distributed_pb2.READY,
            "RUNNING": distributed_pb2.RUNNING,
            "DEGRADED": distributed_pb2.DEGRADED,
            "FAILED": distributed_pb2.FAILED
        }
        
        # Build node health list
        node_health_list = []
        for node in pipeline.nodes.values():
            node_health = distributed_pb2.NodeHealth(
                node_id=node.node_id,
                is_healthy=node.is_healthy,
                assigned_layers=node.end_layer - node.start_layer,
                memory_used_bytes=node.memory_used_bytes
            )
            node_health_list.append(node_health)
        
        response = distributed_pb2.PipelineHealthResponse(
            pipeline_id=pipeline_id,
            status=status_map.get(pipeline.status, distributed_pb2.INITIALIZING),
            nodes=node_health_list
        )
        
        return response
    
    async def _health_check_loop(self):
        """Background task to check node health."""
        logger.info("Starting health check loop")
        
        while True:
            try:
                await asyncio.sleep(10)
                
                async with self._lock:
                    now = datetime.now()
                    timeout_threshold = timedelta(seconds=self.heartbeat_timeout)
                    
                    for pipeline in self.pipelines.values():
                        for node in pipeline.nodes.values():
                            if now - node.last_heartbeat > timeout_threshold:
                                if node.is_healthy:
                                    logger.warning(
                                        f"Node {node.node_id} in pipeline {pipeline.pipeline_id} "
                                        f"missed heartbeat"
                                    )
                                    node.is_healthy = False
                        
                        # Update pipeline status
                        unhealthy = [n for n in pipeline.nodes.values() if not n.is_healthy]
                        if unhealthy:
                            if len(unhealthy) == len(pipeline.nodes):
                                pipeline.status = "FAILED"
                            else:
                                pipeline.status = "DEGRADED"
                        elif pipeline.status == "DEGRADED":
                            pipeline.status = "READY"
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}", exc_info=True)
    
    def get_stats(self) -> Dict:
        """Get coordinator statistics."""
        total_pipelines = len(self.pipelines)
        pipelines_by_status = {}
        total_nodes = 0
        healthy_nodes = 0
        
        for pipeline in self.pipelines.values():
            status = pipeline.status
            pipelines_by_status[status] = pipelines_by_status.get(status, 0) + 1
            
            for node in pipeline.nodes.values():
                total_nodes += 1
                if node.is_healthy:
                    healthy_nodes += 1
        
        return {
            "total_pipelines": total_pipelines,
            "pipelines_by_status": pipelines_by_status,
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
            "unhealthy_nodes": total_nodes - healthy_nodes
        }
