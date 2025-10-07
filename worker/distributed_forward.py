"""
Distributed Forward Pass Implementation

Implements the distributed forward pass through a pipeline of layer workers,
managing activation flow, synchronization, and error handling.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import mlx.core as mx
import grpc

from proto import distributed_pb2, distributed_pb2_grpc
from worker.activation_transfer import ActivationTransfer
from common.retry import retry

logger = logging.getLogger(__name__)


@dataclass
class ForwardPassMetrics:
    """Metrics for distributed forward pass."""
    total_time: float = 0.0
    layer_times: Dict[int, float] = None
    transfer_times: Dict[int, float] = None
    total_transfers: int = 0
    total_bytes_transferred: int = 0
    errors: int = 0
    retries: int = 0
    
    def __post_init__(self):
        if self.layer_times is None:
            self.layer_times = {}
        if self.transfer_times is None:
            self.transfer_times = {}


class DistributedForwardPass:
    """
    Manages distributed forward pass through a pipeline of layer workers.
    
    Coordinates activation flow between nodes, handles synchronization,
    and provides error recovery for distributed inference.
    """
    
    def __init__(
        self,
        pipeline_id: str,
        node_assignments: Dict[str, List[int]],
        node_addresses: Dict[str, str],
        compression: str = "zstd",
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize distributed forward pass.
        
        Args:
            pipeline_id: Unique pipeline identifier
            node_assignments: Map of node_id -> layer indices
            node_addresses: Map of node_id -> gRPC address
            compression: Compression algorithm (zstd, lz4, none)
            timeout: Timeout for operations in seconds
            max_retries: Maximum retry attempts
        """
        self.pipeline_id = pipeline_id
        self.node_assignments = node_assignments
        self.node_addresses = node_addresses
        self.compression = compression
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Build pipeline topology
        self.pipeline_stages = self._build_pipeline_stages()
        
        # Initialize activation transfer
        from worker.activation_transfer import CompressionType
        compression_map = {
            "none": CompressionType.NONE,
            "zstd": CompressionType.ZSTD,
            "lz4": CompressionType.LZ4
        }
        compression_type = compression_map.get(compression.lower(), CompressionType.ZSTD)
        self.activation_transfer = ActivationTransfer(compression=compression_type)
        
        # Metrics
        self.metrics = ForwardPassMetrics()
        
        # gRPC stubs
        self._stubs: Dict[str, distributed_pb2_grpc.DistributedInferenceServiceStub] = {}
        self._channels: Dict[str, grpc.aio.Channel] = {}
        
        logger.info(
            f"Initialized distributed forward pass for pipeline {pipeline_id} "
            f"with {len(self.pipeline_stages)} stages"
        )
    
    def _build_pipeline_stages(self) -> List[Tuple[str, List[int]]]:
        """
        Build ordered pipeline stages from node assignments.
        
        Returns:
            List of (node_id, layer_indices) tuples in execution order
        """
        # Sort nodes by their first layer index
        stages = []
        for node_id, layers in self.node_assignments.items():
            if layers:
                first_layer = min(layers)
                stages.append((first_layer, node_id, sorted(layers)))
        
        # Sort by first layer to get execution order
        stages.sort(key=lambda x: x[0])
        
        return [(node_id, layers) for _, node_id, layers in stages]
    
    async def _get_stub(self, node_id: str) -> distributed_pb2_grpc.DistributedInferenceServiceStub:
        """Get or create gRPC stub for node."""
        if node_id not in self._stubs:
            address = self.node_addresses[node_id]
            channel = grpc.aio.insecure_channel(
                address,
                options=[
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                    ('grpc.keepalive_time_ms', 10000),
                    ('grpc.keepalive_timeout_ms', 5000),
                ]
            )
            self._channels[node_id] = channel
            self._stubs[node_id] = distributed_pb2_grpc.DistributedInferenceServiceStub(channel)
        
        return self._stubs[node_id]
    
    @retry(max_attempts=3, base_delay=1.0, backoff_multiplier=1.5, timeout=30.0)
    async def _execute_stage(
        self,
        node_id: str,
        layer_indices: List[int],
        input_activations: mx.array,
        request_id: str
    ) -> mx.array:
        """
        Execute forward pass on a single pipeline stage.
        
        Args:
            node_id: Node to execute on
            layer_indices: Layers to execute
            input_activations: Input activations
            request_id: Request identifier
            
        Returns:
            Output activations from this stage
        """
        stage_start = time.time()
        
        try:
            # Get gRPC stub
            stub = await self._get_stub(node_id)
            
            # Serialize activations
            serialized = self.activation_transfer.serialize_activation(
                input_activations,
                metadata={
                    'pipeline_id': self.pipeline_id,
                    'request_id': request_id,
                    'stage': node_id,
                    'layers': layer_indices
                }
            )
            
            # Create request
            request = distributed_pb2.ForwardPassRequest(
                pipeline_id=self.pipeline_id,
                request_id=request_id,
                layer_start=min(layer_indices),
                layer_end=max(layer_indices) + 1,
                input_data=serialized.data,
                compression=serialized.compression,
                shape=serialized.shape,
                dtype=serialized.dtype
            )
            
            # Execute forward pass
            logger.debug(
                f"Executing stage {node_id} layers {layer_indices} "
                f"for request {request_id}"
            )
            
            response = await stub.ForwardPass(request, timeout=self.timeout)
            
            # Deserialize output
            output_activations = self.activation_transfer.deserialize_activation(
                response.output_data,
                shape=list(response.shape),
                dtype=response.dtype,
                compression=response.compression
            )
            
            # Update metrics
            stage_time = time.time() - stage_start
            self.metrics.layer_times[min(layer_indices)] = stage_time
            self.metrics.total_bytes_transferred += len(serialized.data) + len(response.output_data)
            self.metrics.total_transfers += 1
            
            logger.debug(
                f"Stage {node_id} completed in {stage_time:.3f}s, "
                f"output shape: {output_activations.shape}"
            )
            
            return output_activations
            
        except grpc.aio.AioRpcError as e:
            logger.error(f"gRPC error in stage {node_id}: {e.code()} - {e.details()}")
            self.metrics.errors += 1
            raise
        except Exception as e:
            logger.error(f"Error executing stage {node_id}: {e}")
            self.metrics.errors += 1
            raise
    
    async def forward(
        self,
        input_ids: mx.array,
        request_id: Optional[str] = None
    ) -> mx.array:
        """
        Execute distributed forward pass through the pipeline.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            request_id: Optional request identifier
            
        Returns:
            Final output activations (logits)
        """
        if request_id is None:
            request_id = f"{self.pipeline_id}_{int(time.time() * 1000)}"
        
        forward_start = time.time()
        
        try:
            logger.info(
                f"Starting distributed forward pass for request {request_id}, "
                f"input shape: {input_ids.shape}"
            )
            
            # Initialize with input embeddings
            # First stage will handle embedding lookup
            current_activations = input_ids
            
            # Execute each pipeline stage in order
            for stage_idx, (node_id, layer_indices) in enumerate(self.pipeline_stages):
                logger.debug(
                    f"Executing stage {stage_idx + 1}/{len(self.pipeline_stages)}: "
                    f"node {node_id}, layers {layer_indices}"
                )
                
                current_activations = await self._execute_stage(
                    node_id=node_id,
                    layer_indices=layer_indices,
                    input_activations=current_activations,
                    request_id=request_id
                )
            
            # Update total metrics
            self.metrics.total_time = time.time() - forward_start
            
            logger.info(
                f"Distributed forward pass completed in {self.metrics.total_time:.3f}s, "
                f"output shape: {current_activations.shape}, "
                f"transfers: {self.metrics.total_transfers}, "
                f"bytes: {self.metrics.total_bytes_transferred}"
            )
            
            return current_activations
            
        except Exception as e:
            logger.error(f"Distributed forward pass failed: {e}")
            self.metrics.errors += 1
            raise
    
    async def forward_with_cache(
        self,
        input_ids: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
        request_id: Optional[str] = None
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        """
        Execute distributed forward pass with KV cache support.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            cache: Optional KV cache from previous forward pass
            request_id: Optional request identifier
            
        Returns:
            Tuple of (output_activations, updated_cache)
        """
        if request_id is None:
            request_id = f"{self.pipeline_id}_{int(time.time() * 1000)}"
        
        forward_start = time.time()
        
        try:
            logger.info(
                f"Starting distributed forward pass with cache for request {request_id}"
            )
            
            current_activations = input_ids
            updated_cache = []
            
            # Execute each pipeline stage with cache
            for stage_idx, (node_id, layer_indices) in enumerate(self.pipeline_stages):
                # Get cache for this stage's layers
                stage_cache = None
                if cache is not None:
                    stage_cache = [cache[i] for i in layer_indices if i < len(cache)]
                
                # Execute stage with cache
                stub = await self._get_stub(node_id)
                
                # Serialize activations and cache
                serialized = self.activation_transfer.serialize_activation(
                    current_activations,
                    metadata={
                        'pipeline_id': self.pipeline_id,
                        'request_id': request_id,
                        'stage': node_id,
                        'has_cache': stage_cache is not None
                    }
                )
                
                # Create request with cache
                request = distributed_pb2.ForwardPassRequest(
                    pipeline_id=self.pipeline_id,
                    request_id=request_id,
                    layer_start=min(layer_indices),
                    layer_end=max(layer_indices) + 1,
                    input_data=serialized.data,
                    compression=serialized.compression,
                    shape=serialized.shape,
                    dtype=serialized.dtype,
                    use_cache=True
                )
                
                # Execute
                response = await stub.ForwardPass(request, timeout=self.timeout)
                
                # Deserialize output and cache
                current_activations = self.activation_transfer.deserialize_activation(
                    response.output_data,
                    shape=list(response.shape),
                    dtype=response.dtype,
                    compression=response.compression
                )
                
                # TODO: Deserialize and collect cache updates
                # For now, cache management is simplified
            
            self.metrics.total_time = time.time() - forward_start
            
            logger.info(
                f"Distributed forward pass with cache completed in "
                f"{self.metrics.total_time:.3f}s"
            )
            
            return current_activations, updated_cache
            
        except Exception as e:
            logger.error(f"Distributed forward pass with cache failed: {e}")
            self.metrics.errors += 1
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get forward pass metrics."""
        return {
            'total_time': self.metrics.total_time,
            'layer_times': self.metrics.layer_times,
            'transfer_times': self.metrics.transfer_times,
            'total_transfers': self.metrics.total_transfers,
            'total_bytes_transferred': self.metrics.total_bytes_transferred,
            'errors': self.metrics.errors,
            'retries': self.metrics.retries,
            'avg_layer_time': (
                sum(self.metrics.layer_times.values()) / len(self.metrics.layer_times)
                if self.metrics.layer_times else 0.0
            ),
            'throughput_mbps': (
                (self.metrics.total_bytes_transferred / 1024 / 1024) / self.metrics.total_time
                if self.metrics.total_time > 0 else 0.0
            )
        }
    
    def reset_metrics(self):
        """Reset metrics counters."""
        self.metrics = ForwardPassMetrics()
    
    async def close(self):
        """Close all gRPC connections."""
        for channel in self._channels.values():
            await channel.close()
        
        self._channels.clear()
        self._stubs.clear()
        
        logger.info(f"Closed distributed forward pass for pipeline {self.pipeline_id}")
