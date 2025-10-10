"""
Sharding Coordinator for Automatic Model Sharding

Manages automatic model sharding decisions, shard plan creation,
and shard metadata tracking for distributed model inference.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from threading import Lock

from worker.model_partitioner import ModelPartitioner, LayerAssignment, PartitionStrategy
from common.logging_system import get_logger

logger = get_logger(__name__)


class ShardStatus(str, Enum):
    """Status of a sharded model"""
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"
    PARTIAL = "partial"  # Some workers have completed, others haven't


class DownloadStatus(str, Enum):
    """Download status for individual worker"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ShardedModelMetadata:
    """Metadata for a sharded model"""
    model_name: str
    total_layers: int
    shard_assignments: Dict[str, LayerAssignment]  # worker_id -> assignment
    created_at: float
    status: ShardStatus
    
    def get_assignment(self, worker_id: str) -> Optional[LayerAssignment]:
        """Get layer assignment for a specific worker"""
        return self.shard_assignments.get(worker_id)
    
    def update_download_status(self, worker_id: str, status: DownloadStatus) -> None:
        """Update download status for a worker"""
        if worker_id in self.shard_assignments:
            assignment = self.shard_assignments[worker_id]
            assignment.download_status = status


class ShardingCoordinator:
    """
    Coordinates automatic model sharding across workers.
    
    Responsibilities:
    - Determine if a model should be sharded
    - Create shard plans using ModelPartitioner
    - Track shard metadata and download status
    - Provide shard information for inference routing
    """
    
    def __init__(
        self,
        enable_auto_sharding: bool = False,
        shard_threshold_gb: float = 30.0,
        strategy: str = "balanced",
        min_workers_for_sharding: int = 2
    ):
        """
        Initialize ShardingCoordinator.
        
        Args:
            enable_auto_sharding: Whether automatic sharding is enabled
            shard_threshold_gb: Minimum model size (GB) to trigger sharding
            strategy: Partitioning strategy (balanced, memory_aware)
            min_workers_for_sharding: Minimum workers required for sharding
        """
        self.enable_auto_sharding = enable_auto_sharding
        self.shard_threshold_gb = shard_threshold_gb
        self.strategy = strategy
        self.min_workers_for_sharding = min_workers_for_sharding
        
        # Model partitioner for creating shard plans
        self.partitioner = ModelPartitioner()
        
        # Shard metadata storage: model_name -> ShardedModelMetadata
        self.shard_metadata: Dict[str, ShardedModelMetadata] = {}
        self.metadata_lock = Lock()
        
        logger.info(
            f"Initialized ShardingCoordinator: "
            f"auto_sharding={enable_auto_sharding}, "
            f"threshold={shard_threshold_gb}GB, "
            f"strategy={strategy}"
        )
    
    def should_shard_model(
        self,
        model_name: str,
        model_size_gb: float,
        num_workers: int
    ) -> bool:
        """
        Determine if a model should be sharded.
        
        Args:
            model_name: Name of the model
            model_size_gb: Estimated model size in GB
            num_workers: Number of available workers
            
        Returns:
            True if model should be sharded, False otherwise
        """
        # ============================================================
        # SHARDING DECISION POINT: Model Download Strategy
        # ============================================================
        # This is the primary decision point for whether to shard a model.
        # The decision is based on three criteria:
        #
        # 1. Auto-sharding must be enabled (enable_auto_sharding=True)
        # 2. Sufficient workers available (num_workers >= min_workers_for_sharding)
        # 3. Model size exceeds threshold (model_size_gb >= shard_threshold_gb)
        #
        # If all criteria are met, the model will be sharded across workers.
        # Otherwise, the model will be loaded fully on each worker.
        #
        # This decision is made by the coordinator during model download
        # (see CoordinatorService.DownloadModel).
        # ============================================================
        
        # Check if auto-sharding is enabled
        if not self.enable_auto_sharding:
            logger.debug(f"Auto-sharding disabled for {model_name}")
            return False
        
        # Check if enough workers are available
        if num_workers < self.min_workers_for_sharding:
            logger.debug(
                f"Not enough workers for sharding {model_name}: "
                f"{num_workers} < {self.min_workers_for_sharding}"
            )
            return False
        
        # Check if model size exceeds threshold
        if model_size_gb < self.shard_threshold_gb:
            logger.debug(
                f"Model {model_name} below sharding threshold: "
                f"{model_size_gb}GB < {self.shard_threshold_gb}GB"
            )
            return False
        
        logger.info(
            f"Model {model_name} should be sharded: "
            f"size={model_size_gb}GB, workers={num_workers}"
        )
        return True
    
    def create_shard_plan(
        self,
        model_name: str,
        num_layers: int,
        workers: List[Dict[str, any]]
    ) -> ShardedModelMetadata:
        """
        Create a shard plan for a model using ModelPartitioner.
        
        Args:
            model_name: Name of the model
            num_layers: Total number of layers in the model
            workers: List of worker info dicts with 'worker_id', 'address', 
                    and optionally 'available_memory' (GB)
            
        Returns:
            ShardedModelMetadata with layer assignments
        """
        import time
        
        logger.info(
            f"Creating shard plan for {model_name}",
            extra={
                "model_name": model_name,
                "total_layers": num_layers,
                "num_workers": len(workers),
                "strategy": self.strategy,
                "operation": "create_shard_plan"
            }
        )
        
        # Convert strategy string to PartitionStrategy enum
        if self.strategy == "balanced":
            partition_strategy = PartitionStrategy.BALANCED
        elif self.strategy == "memory_aware":
            partition_strategy = PartitionStrategy.MEMORY_AWARE
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using balanced")
            partition_strategy = PartitionStrategy.BALANCED
        
        # Prepare node info for partitioner
        nodes = []
        for worker in workers:
            node_info = {
                'node_id': worker['worker_id'],
                'address': worker['address']
            }
            # Add memory info if available for memory-aware partitioning
            if 'available_memory' in worker:
                node_info['available_memory'] = worker['available_memory']
            nodes.append(node_info)
        
        # Create pipeline config using ModelPartitioner
        pipeline_config = self.partitioner.partition_model(
            model_name=model_name,
            total_layers=num_layers,
            nodes=nodes,
            strategy=partition_strategy
        )
        
        # Convert LayerAssignments to dict with download status
        shard_assignments = {}
        for assignment in pipeline_config.assignments:
            # Add download status to assignment
            assignment.download_status = DownloadStatus.PENDING
            shard_assignments[assignment.node_id] = assignment
        
        # Create metadata
        metadata = ShardedModelMetadata(
            model_name=model_name,
            total_layers=num_layers,
            shard_assignments=shard_assignments,
            created_at=time.time(),
            status=ShardStatus.DOWNLOADING
        )
        
        # Store metadata
        with self.metadata_lock:
            self.shard_metadata[model_name] = metadata
        
        logger.info(
            f"Created shard plan for {model_name}",
            extra={
                "model_name": model_name,
                "total_layers": num_layers,
                "num_workers": len(workers),
                "operation": "shard_plan_created"
            }
        )
        
        # Log worker assignments with structured data
        for worker_id, assignment in shard_assignments.items():
            logger.info(
                f"Shard assignment: {worker_id} -> layers {assignment.start_layer}-{assignment.end_layer}",
                extra={
                    "model_name": model_name,
                    "worker_id": worker_id,
                    "start_layer": assignment.start_layer,
                    "end_layer": assignment.end_layer,
                    "num_layers": assignment.num_layers,
                    "layer_range": f"{assignment.start_layer}-{assignment.end_layer}",
                    "next_worker": assignment.next_node_id,
                    "operation": "worker_assignment"
                }
            )
        
        return metadata
    
    def get_shard_metadata(self, model_name: str) -> Optional[ShardedModelMetadata]:
        """
        Get shard metadata for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ShardedModelMetadata if model is sharded, None otherwise
        """
        with self.metadata_lock:
            return self.shard_metadata.get(model_name)
    
    def update_shard_status(
        self,
        model_name: str,
        worker_id: str,
        download_status: DownloadStatus
    ) -> None:
        """
        Update download status for a worker's shard.
        
        Args:
            model_name: Name of the model
            worker_id: ID of the worker
            download_status: New download status
        """
        with self.metadata_lock:
            metadata = self.shard_metadata.get(model_name)
            if not metadata:
                logger.warning(f"No shard metadata found for {model_name}")
                return
            
            # Update worker's download status
            if worker_id in metadata.shard_assignments:
                assignment = metadata.shard_assignments[worker_id]
                assignment.download_status = download_status
                
                logger.info(
                    f"Updated shard download status: {model_name} on {worker_id}",
                    extra={
                        "model_name": model_name,
                        "worker_id": worker_id,
                        "download_status": download_status.value,
                        "layer_range": f"{assignment.start_layer}-{assignment.end_layer}",
                        "operation": "shard_download_status_update"
                    }
                )
                
                # Update overall model status
                self._update_model_status(metadata)
            else:
                logger.warning(
                    f"Worker {worker_id} not found in shard assignments for {model_name}"
                )
    
    def _update_model_status(self, metadata: ShardedModelMetadata) -> None:
        """
        Update overall model status based on worker download statuses.
        
        Args:
            metadata: ShardedModelMetadata to update
        """
        statuses = [
            assignment.download_status
            for assignment in metadata.shard_assignments.values()
        ]
        
        # Check if all workers completed
        if all(status == DownloadStatus.COMPLETED for status in statuses):
            metadata.status = ShardStatus.READY
            logger.info(f"Model {metadata.model_name} is ready (all shards downloaded)")
        
        # Check if any worker failed
        elif any(status == DownloadStatus.FAILED for status in statuses):
            metadata.status = ShardStatus.FAILED
            logger.error(f"Model {metadata.model_name} failed (at least one shard failed)")
        
        # Check if some workers completed
        elif any(status == DownloadStatus.COMPLETED for status in statuses):
            metadata.status = ShardStatus.PARTIAL
            logger.info(f"Model {metadata.model_name} partially downloaded")
        
        # Otherwise, still downloading
        else:
            metadata.status = ShardStatus.DOWNLOADING
    
    def is_model_sharded(self, model_name: str) -> bool:
        """
        Check if a model is sharded.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model is sharded, False otherwise
        """
        with self.metadata_lock:
            return model_name in self.shard_metadata
    
    def remove_shard_metadata(self, model_name: str) -> bool:
        """
        Remove shard metadata for a model (e.g., after deletion).
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if metadata was removed, False if not found
        """
        with self.metadata_lock:
            if model_name in self.shard_metadata:
                del self.shard_metadata[model_name]
                logger.info(f"Removed shard metadata for {model_name}")
                return True
            return False
    
    def list_sharded_models(self) -> List[str]:
        """
        List all sharded models.
        
        Returns:
            List of model names that are sharded
        """
        with self.metadata_lock:
            return list(self.shard_metadata.keys())
    
    def get_worker_assignment(
        self,
        model_name: str,
        worker_id: str
    ) -> Optional[LayerAssignment]:
        """
        Get layer assignment for a specific worker.
        
        Args:
            model_name: Name of the model
            worker_id: ID of the worker
            
        Returns:
            LayerAssignment if found, None otherwise
        """
        metadata = self.get_shard_metadata(model_name)
        if not metadata:
            return None
        
        return metadata.get_assignment(worker_id)
