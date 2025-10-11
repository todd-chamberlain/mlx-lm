"""
gRPC Service Implementation for MLX Worker

This module implements the InferenceService gRPC interface with streaming
capabilities, error handling, and connection management.
"""

import logging
import asyncio
import time
from typing import Iterator, Optional
import grpc
from concurrent import futures

# Import generated gRPC code
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from proto.inference_pb2_grpc import InferenceServiceServicer
from proto import inference_pb2 as pb
from proto import distributed_pb2
from proto import distributed_pb2_grpc as pb2_grpc
from proto.grpc_base import GRPCServerBase
from proto.errors import MLXError, MLXErrorCode

from .model_manager import ModelManager
from .inference_engine import InferenceEngine, InferenceRequest as EngineRequest

logger = logging.getLogger(__name__)


class InferenceServiceImpl(InferenceServiceServicer):
    """gRPC service implementation for MLX inference"""
    
    def __init__(self, model_manager: ModelManager, inference_engine: InferenceEngine):
        self.model_manager = model_manager
        self.inference_engine = inference_engine
        self.worker_id = inference_engine.worker_id
        self.start_time = time.time()
        
        logger.info(f"Initialized InferenceService for worker {self.worker_id}")
    
    def _convert_grpc_request(self, grpc_request: pb.InferenceRequest) -> EngineRequest:
        """Convert gRPC request to engine request"""
        return EngineRequest(
            request_id=grpc_request.request_id,
            prompt=grpc_request.prompt,
            max_tokens=grpc_request.max_tokens if grpc_request.max_tokens > 0 else 100,
            temperature=grpc_request.temperature if grpc_request.temperature > 0 else 0.7,
            top_p=grpc_request.top_p if grpc_request.top_p > 0 else 0.9,
            seed=grpc_request.seed if grpc_request.seed > 0 else None,
            model=grpc_request.model if grpc_request.model else "default",
            use_cache=grpc_request.use_cache
        )
    
    async def ProcessInference(self, request: pb.InferenceRequest, context: grpc.aio.ServicerContext) -> pb.InferenceResponse:
        """Process a single inference request"""
        try:
            logger.debug(f"Received inference request: {request.request_id}")
            
            # Convert request
            engine_request = self._convert_grpc_request(request)
            
            # Process inference
            response = await self.inference_engine.process_inference(engine_request)
            
            # Convert response
            grpc_response = pb.InferenceResponse(
                request_id=response.request_id,
                text=response.text,
                tokens=response.tokens,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                tokens_per_second=response.tokens_per_second,
                worker_id=response.worker_id,
                latency_ms=response.latency_ms
            )
            
            logger.debug(f"Completed inference request: {request.request_id}")
            return grpc_response
            
        except ValueError as e:
            logger.error(f"Validation error for request {request.request_id}: {e}")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        
        except RuntimeError as e:
            logger.error(f"Runtime error for request {request.request_id}: {e}")
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        
        except Exception as e:
            logger.error(f"Unexpected error for request {request.request_id}: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal server error: {str(e)}")
    
    async def StreamInference(self, request: pb.InferenceRequest, context: grpc.aio.ServicerContext) -> Iterator[pb.StreamChunk]:
        """Process a streaming inference request"""
        try:
            logger.debug(f"Received streaming inference request: {request.request_id}")
            
            # Convert request
            engine_request = self._convert_grpc_request(request)
            
            # Check for client disconnection
            def check_cancelled():
                if context.cancelled():
                    logger.info(f"Client cancelled streaming request: {request.request_id}")
                    return True
                return False
            
            # Process streaming inference
            chunk_count = 0
            async for chunk in self.inference_engine.stream_inference(engine_request):
                # Check if client disconnected
                if check_cancelled():
                    break
                
                # Convert chunk
                grpc_chunk = pb.StreamChunk(
                    request_id=chunk.request_id,
                    text=chunk.text,
                    token=chunk.token,
                    is_final=chunk.is_final,
                    worker_id=chunk.worker_id
                )
                
                yield grpc_chunk
                chunk_count += 1
                
                # Add small delay to prevent overwhelming the client
                if chunk_count % 10 == 0:
                    await asyncio.sleep(0.001)  # 1ms delay every 10 chunks
            
            logger.debug(f"Completed streaming inference request: {request.request_id} "
                        f"with {chunk_count} chunks")
            
        except ValueError as e:
            logger.error(f"Validation error for streaming request {request.request_id}: {e}")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        
        except RuntimeError as e:
            logger.error(f"Runtime error for streaming request {request.request_id}: {e}")
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        
        except Exception as e:
            logger.error(f"Unexpected error for streaming request {request.request_id}: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal server error: {str(e)}")
    
    async def HealthCheck(self, request: pb.Empty, context: grpc.aio.ServicerContext) -> pb.HealthStatus:
        """Check worker health status"""
        try:
            # Get current metrics
            metrics = self.inference_engine.get_metrics()
            memory_usage = self.model_manager.get_memory_usage()
            
            # Determine health status
            status = "healthy"
            message = "Worker is healthy"
            
            # Check for issues
            if metrics['active_requests'] > 10:
                status = "degraded"
                message = f"High load: {metrics['active_requests']} active requests"
            
            if memory_usage['percent'] > 90:
                status = "degraded" 
                message = f"High memory usage: {memory_usage['percent']:.1f}%"
            
            # Only check success rate if we have enough requests (at least 10)
            # This prevents marking worker unhealthy during startup or after model load
            if metrics['total_requests'] >= 10 and metrics['success_rate'] < 0.9:
                status = "unhealthy"
                message = f"Low success rate: {metrics['success_rate']:.2f}"
            
            # Create metrics message
            worker_metrics = pb.WorkerMetrics(
                avg_latency_ms=metrics['avg_latency_ms'],
                tokens_per_second=metrics['tokens_per_second'],
                total_requests=metrics['total_requests'],
                failed_requests=metrics['failed_requests'],
                active_requests=metrics['active_requests'],
                memory_usage_gb=memory_usage['used_gb']
            )
            
            response = pb.HealthStatus(
                status=status,
                message=message,
                metrics=worker_metrics
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return pb.HealthStatus(
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                metrics=pb.WorkerMetrics()
            )
    
    async def GetMetrics(self, request: pb.Empty, context: grpc.aio.ServicerContext) -> pb.WorkerMetrics:
        """Get detailed worker metrics"""
        try:
            metrics = self.inference_engine.get_metrics()
            memory_usage = self.model_manager.get_memory_usage()
            
            response = pb.WorkerMetrics(
                avg_latency_ms=metrics['avg_latency_ms'],
                tokens_per_second=metrics['tokens_per_second'],
                total_requests=metrics['total_requests'],
                failed_requests=metrics['failed_requests'],
                active_requests=metrics['active_requests'],
                memory_usage_gb=memory_usage['used_gb']
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Failed to get metrics: {str(e)}")
    
    async def DownloadModel(self, request: pb.ModelDownloadRequest, context: grpc.aio.ServicerContext) -> pb.ModelDownloadResponse:
        """Download a model on this worker"""
        try:
            import asyncio
            from mlx_lm import load
            import os
            from pathlib import Path
            from .model_partitioner import LayerAssignment as LoaderLayerAssignment
            
            logger.info(f"Model download requested: {request.model_name}")
            
            # Check if this is a sharded download
            is_sharded = request.HasField('layer_assignment')
            
            if is_sharded:
                logger.info(
                    f"Sharded download requested: layers "
                    f"{request.layer_assignment.start_layer}-{request.layer_assignment.end_layer}"
                )
            
            # Check if model is already loaded
            if request.model_name in self.model_manager.loaded_models:
                logger.info(f"Model {request.model_name} is already loaded")
                return pb.ModelDownloadResponse(
                    success=True,
                    message=f"Model {request.model_name} is already loaded",
                    download_id="",
                    workers_notified=1
                )
            
            # Start download/load in background
            async def download_and_load_model():
                try:
                    # Set HF_HOME to avoid permission issues
                    hf_home = os.getenv('HF_HOME', '/Users/Shared/models/.cache')
                    os.environ['HF_HOME'] = hf_home
                    
                    if is_sharded:
                        # Sharded download and load
                        logger.info(f"Starting sharded download for {request.model_name}")
                        
                        # Convert proto LayerAssignment to loader LayerAssignment
                        layer_assignment = LoaderLayerAssignment(
                            worker_id=request.layer_assignment.worker_id,
                            start_layer=request.layer_assignment.start_layer,
                            end_layer=request.layer_assignment.end_layer,
                            next_worker_id=request.layer_assignment.next_worker_id if request.layer_assignment.next_worker_id else None,
                            next_worker_address=request.layer_assignment.next_worker_address if request.layer_assignment.next_worker_address else None
                        )
                        
                        # Resolve model name to local path using registry
                        from .model_registry import ModelRegistry
                        registry = ModelRegistry(worker_id=self.worker_id)
                        model_path = registry.get_model_path(request.model_name)

                        if not model_path:
                            raise RuntimeError(f"Model '{request.model_name}' not found in worker registry")

                        logger.info(f"Resolved model '{request.model_name}' to path: {model_path}")

                        # Load sharded model
                        success = await asyncio.to_thread(
                            self.model_manager.load_sharded_model,
                            model_path,
                            layer_assignment,
                            request.model_name
                        )
                        
                        if not success:
                            raise RuntimeError("Failed to load sharded model")
                        
                        logger.info(f"Successfully loaded sharded model {request.model_name}")
                        
                        # Update download tracking
                        self.model_manager._downloaded_models = getattr(self.model_manager, '_downloaded_models', {})
                        self.model_manager._downloaded_models[request.model_name] = {
                            'status': 'ready',
                            'progress': 100.0,
                            'is_sharded': True,
                            'start_layer': layer_assignment.start_layer,
                            'end_layer': layer_assignment.end_layer
                        }
                        
                    else:
                        # Regular full model download
                        # Check if model already exists locally
                        model_cache_path = Path(hf_home) / "huggingface" / "hub" / f"models--{request.model_name.replace('/', '--')}"
                        already_downloaded = model_cache_path.exists()
                        
                        if already_downloaded:
                            logger.info(f"Model {request.model_name} already exists locally, loading...")
                        else:
                            logger.info(f"Downloading model {request.model_name}...")
                        
                        # Download/load model using mlx_lm (will use cache if available)
                        model, tokenizer = await asyncio.to_thread(load, request.model_name)
                        logger.info(f"Successfully loaded {request.model_name}")
                        
                        # Add model directly to model manager's loaded models
                        from .model_manager import ModelInfo
                        import time
                        
                        model_info = ModelInfo(
                            model_path=request.model_name,
                            model_name=request.model_name,
                            model=model,
                            tokenizer=tokenizer,
                            loaded_at=time.time(),
                            memory_usage_gb=2.0,  # Estimate
                            max_tokens=4096,
                            context_length=4096
                        )
                        
                        with self.model_manager.model_lock:
                            self.model_manager.loaded_models[request.model_name] = model_info
                            logger.info(f"Added model to loaded_models. Current keys: {list(self.model_manager.loaded_models.keys())}")
                        
                        logger.info(f"Model {request.model_name} added to model manager")
                        
                        # Update download tracking
                        self.model_manager._downloaded_models = getattr(self.model_manager, '_downloaded_models', {})
                        self.model_manager._downloaded_models[request.model_name] = {
                            'status': 'ready',
                            'progress': 100.0,
                            'is_sharded': False
                        }
                    
                    # Note: Worker will need to re-register to notify coordinator of new models
                    logger.info(f"Model {request.model_name} is now available for inference")
                    logger.info("Note: Restart worker or wait for periodic re-registration to update coordinator")
                    
                except Exception as e:
                    logger.error(f"Failed to download/load model {request.model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    self.model_manager._downloaded_models = getattr(self.model_manager, '_downloaded_models', {})
                    self.model_manager._downloaded_models[request.model_name] = {
                        'status': 'failed',
                        'progress': 0.0,
                        'error': str(e)
                    }
            
            # Initialize download tracking
            self.model_manager._downloaded_models = getattr(self.model_manager, '_downloaded_models', {})
            self.model_manager._downloaded_models[request.model_name] = {
                'status': 'downloading',
                'progress': 0.0
            }
            
            # Start download task
            asyncio.create_task(download_and_load_model())
            
            shard_info = ""
            if is_sharded:
                shard_info = f" (layers {request.layer_assignment.start_layer}-{request.layer_assignment.end_layer})"
            
            return pb.ModelDownloadResponse(
                success=True,
                message=f"Model download/load started for {request.model_name}{shard_info}",
                download_id="",
                workers_notified=1
            )
            
        except Exception as e:
            logger.error(f"Error in DownloadModel: {e}", exc_info=True)
            return pb.ModelDownloadResponse(
                success=False,
                message=f"Failed to start download: {str(e)}",
                download_id="",
                workers_notified=0
            )
    
    async def GetModelStatus(self, request: pb.ModelStatusRequest, context: grpc.aio.ServicerContext) -> pb.ModelStatusResponse:
        """Get model download status"""
        try:
            logger.debug(f"Model status requested: {request.model_name}")
            
            # Check if model is downloaded
            downloaded_models = getattr(self.model_manager, '_downloaded_models', {})
            
            if request.model_name in downloaded_models:
                model_info = downloaded_models[request.model_name]
                status = model_info.get('status', 'unknown')
                progress = model_info.get('progress', 0.0)
            else:
                # Check if model exists in cache
                import os
                cache_dir = os.environ.get('HF_HOME', '/Users/Shared/models/.cache')
                model_cache_path = os.path.join(cache_dir, 'hub', f"models--{request.model_name.replace('/', '--')}")
                
                if os.path.exists(model_cache_path):
                    status = 'ready'
                    progress = 100.0
                else:
                    status = 'not_found'
                    progress = 0.0
            
            return pb.ModelStatusResponse(
                model_name=request.model_name,
                status=status,
                progress_percent=progress,
                size_bytes=0,
                worker_status=[]
            )
            
        except Exception as e:
            logger.error(f"Error in GetModelStatus: {e}", exc_info=True)
            return pb.ModelStatusResponse(
                model_name=request.model_name,
                status="error",
                progress_percent=0.0,
                size_bytes=0,
                worker_status=[]
            )
    
    async def DeleteModel(self, request: pb.ModelDeleteRequest, context: grpc.aio.ServicerContext) -> pb.ModelDeleteResponse:
        """Delete a model from this worker"""
        try:
            logger.info(f"Model deletion requested: {request.model_name}")
            
            # Check if model is loaded
            model_info = self.model_manager.get_model_info_dict(request.model_name)
            
            if not model_info:
                logger.warning(f"Model {request.model_name} not loaded on this worker")
                return pb.ModelDeleteResponse(
                    success=True,
                    message=f"Model {request.model_name} was not loaded",
                    workers_deleted=0,
                    worker_status=[]
                )
            
            # Check if this is a sharded model
            is_sharded = model_info.get('is_sharded', False)
            
            if is_sharded:
                logger.info(
                    f"Deleting sharded model {request.model_name} "
                    f"(layers {model_info['shard_metadata']['start_layer']}-"
                    f"{model_info['shard_metadata']['end_layer']})"
                )
            
            # Unload model from memory
            success = await asyncio.to_thread(
                self.model_manager.unload_model,
                request.model_name
            )
            
            if not success:
                error_msg = f"Failed to unload model {request.model_name}"
                logger.error(error_msg)
                return pb.ModelDeleteResponse(
                    success=False,
                    message=error_msg,
                    workers_deleted=0,
                    worker_status=[]
                )
            
            # Clean up download tracking
            downloaded_models = getattr(self.model_manager, '_downloaded_models', {})
            if request.model_name in downloaded_models:
                del downloaded_models[request.model_name]
            
            # Clean up LayerWorker resources if sharded
            if is_sharded:
                # LayerWorker cleanup is handled by model_manager.unload_model
                # which deletes the model reference and triggers garbage collection
                logger.info(f"Cleaned up LayerWorker resources for {request.model_name}")
            
            message = f"Successfully deleted model {request.model_name}"
            if is_sharded:
                message += " (sharded model)"
            
            logger.info(message)
            
            return pb.ModelDeleteResponse(
                success=True,
                message=message,
                workers_deleted=1,
                worker_status=[]
            )
            
        except Exception as e:
            error_msg = f"Error deleting model {request.model_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return pb.ModelDeleteResponse(
                success=False,
                message=error_msg,
                workers_deleted=0,
                worker_status=[]
            )


class WorkerGrpcServer:
    """gRPC server for MLX worker with streaming support"""
    
    def __init__(self, model_manager: ModelManager, inference_engine: InferenceEngine, 
                 port: int = 8080, max_workers: int = 10):
        self.model_manager = model_manager
        self.inference_engine = inference_engine
        self.port = port
        self.max_workers = max_workers
        self.server = None
        self.service_impl = None
        self.coordinator_client = None
        
        logger.info(f"Initializing WorkerGrpcServer on port {port}")
    
    def set_coordinator_client(self, coordinator_client):
        """Set the coordinator client reference for the service"""
        self.coordinator_client = coordinator_client
        if self.service_impl:
            self.service_impl.coordinator_client = coordinator_client
    
    async def start(self):
        """Start the gRPC server"""
        try:
            # Create service implementations
            self.service_impl = InferenceServiceImpl(self.model_manager, self.inference_engine)
            self.distributed_service_impl = DistributedInferenceServiceImpl(
                self.model_manager,
                self.inference_engine.worker_id
            )
            
            # Set coordinator client if available
            if self.coordinator_client:
                self.service_impl.coordinator_client = self.coordinator_client
            
            # Create server
            self.server = grpc.aio.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=[
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.http2.min_time_between_pings_ms', 10000),
                    ('grpc.http2.min_ping_interval_without_data_ms', 300000),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB for activations
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),     # 100MB for activations
                ]
            )
            
            # Add services to server
            from proto.inference_pb2_grpc import add_InferenceServiceServicer_to_server
            from proto.distributed_pb2_grpc import add_DistributedInferenceServiceServicer_to_server
            
            add_InferenceServiceServicer_to_server(self.service_impl, self.server)
            add_DistributedInferenceServiceServicer_to_server(self.distributed_service_impl, self.server)
            
            logger.info("Registered InferenceService and DistributedInferenceService")
            
            # Add insecure port
            listen_addr = f'[::]:{self.port}'
            self.server.add_insecure_port(listen_addr)
            
            # Start server
            await self.server.start()
            logger.info(f"gRPC server started on {listen_addr}")
            
        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")
            raise
    
    async def stop(self, grace_period: float = 5.0):
        """Stop the gRPC server gracefully"""
        if self.server:
            logger.info("Stopping gRPC server...")
            await self.server.stop(grace_period)
            logger.info("gRPC server stopped")
    
    async def wait_for_termination(self):
        """Wait for server termination"""
        if self.server:
            await self.server.wait_for_termination()
    
    def get_service_info(self) -> dict:
        """Get service information"""
        return {
            'port': self.port,
            'max_workers': self.max_workers,
            'worker_id': self.inference_engine.worker_id,
            'status': 'running' if self.server else 'stopped'
        }


class StreamingManager:
    """Manages streaming connections and backpressure"""
    
    def __init__(self, max_concurrent_streams: int = 10):
        self.max_concurrent_streams = max_concurrent_streams
        self.active_streams = {}
        self.stream_lock = asyncio.Lock()
        
    async def register_stream(self, request_id: str) -> bool:
        """Register a new streaming request"""
        async with self.stream_lock:
            if len(self.active_streams) >= self.max_concurrent_streams:
                logger.warning(f"Max concurrent streams reached, rejecting {request_id}")
                return False
            
            self.active_streams[request_id] = time.time()
            logger.debug(f"Registered stream: {request_id}")
            return True
    
    async def unregister_stream(self, request_id: str):
        """Unregister a streaming request"""
        async with self.stream_lock:
            if request_id in self.active_streams:
                del self.active_streams[request_id]
                logger.debug(f"Unregistered stream: {request_id}")
    
    async def get_stream_count(self) -> int:
        """Get current number of active streams"""
        async with self.stream_lock:
            return len(self.active_streams)
    
    async def cleanup_stale_streams(self, max_age_seconds: float = 300):
        """Clean up streams that have been active too long"""
        current_time = time.time()
        stale_streams = []
        
        async with self.stream_lock:
            for request_id, start_time in self.active_streams.items():
                if current_time - start_time > max_age_seconds:
                    stale_streams.append(request_id)
            
            for request_id in stale_streams:
                del self.active_streams[request_id]
                logger.warning(f"Cleaned up stale stream: {request_id}")
        
        return len(stale_streams)


class DistributedInferenceServiceImpl(pb2_grpc.DistributedInferenceServiceServicer):
    """gRPC service implementation for distributed inference"""
    
    def __init__(self, model_manager: ModelManager, worker_id: str):
        self.model_manager = model_manager
        self.worker_id = worker_id
        
        logger.info(f"Initialized DistributedInferenceService for worker {worker_id}")
    
    async def ForwardPass(
        self,
        request: distributed_pb2.ForwardPassRequest,
        context: grpc.aio.ServicerContext
    ) -> distributed_pb2.ForwardPassResponse:
        """
        Execute forward pass on assigned layers.
        
        Args:
            request: ForwardPassRequest with input activations
            context: gRPC context
            
        Returns:
            ForwardPassResponse with output activations
        """
        try:
            compute_start = time.time()
            
            logger.info(
                f"Processing forward pass request",
                extra={
                    "request_id": request.request_id,
                    "pipeline_id": request.pipeline_id,
                    "worker_id": self.worker_id,
                    "layer_start": request.layer_start,
                    "layer_end": request.layer_end,
                    "layer_range": f"{request.layer_start}-{request.layer_end}",
                    "input_shape": list(request.shape),
                    "compression": request.compression,
                    "operation": "forward_pass_start"
                }
            )
            
            # Import activation transfer
            from worker.activation_transfer import ActivationTransfer, CompressionType
            
            # Create activation transfer instance
            compression_map = {
                "none": CompressionType.NONE,
                "zstd": CompressionType.ZSTD,
                "lz4": CompressionType.LZ4
            }
            compression_type = compression_map.get(request.compression.lower(), CompressionType.ZSTD)
            activation_transfer = ActivationTransfer(compression=compression_type)
            
            # Deserialize input activations
            input_activations = activation_transfer.deserialize_activation(
                data=request.input_data,
                shape=list(request.shape),
                dtype=request.dtype,
                compression=request.compression
            )
            
            logger.debug(
                f"Deserialized input activations: shape={input_activations.shape}, "
                f"dtype={input_activations.dtype}"
            )
            
            # Get the model's layer worker
            # For sharded models, we need to get the LayerWorker instance
            # Currently only supports single model per worker (default)
            model_info = self.model_manager.get_model_info_dict("default")
            if not model_info:
                raise RuntimeError("Model not loaded")
            
            # Check if this is a sharded model
            if not model_info.get('is_sharded', False):
                raise RuntimeError("ForwardPass RPC only works with sharded models")
            
            # Get the layer worker from model manager
            # The model manager should have stored the LayerWorker instance
            layer_worker = model_info.get('layer_worker')
            if not layer_worker:
                raise RuntimeError("LayerWorker not found in model info")
            
            # Execute forward pass through assigned layers
            output_activations, cache = layer_worker.forward(input_activations, cache=None)
            
            logger.debug(
                f"Forward pass completed: output_shape={output_activations.shape}"
            )
            
            # Serialize output activations
            serialized = activation_transfer.serialize_activation(
                activation=output_activations,
                metadata={
                    'pipeline_id': request.pipeline_id,
                    'request_id': request.request_id,
                    'layer_start': request.layer_start,
                    'layer_end': request.layer_end
                }
            )
            
            compute_time_ms = (time.time() - compute_start) * 1000
            
            logger.info(
                f"Forward pass completed",
                extra={
                    "request_id": request.request_id,
                    "pipeline_id": request.pipeline_id,
                    "worker_id": self.worker_id,
                    "layer_start": request.layer_start,
                    "layer_end": request.layer_end,
                    "compute_time_ms": compute_time_ms,
                    "output_shape": str(output_activations.shape),
                    "output_bytes": len(serialized.data),
                    "operation": "forward_pass_complete"
                }
            )
            
            # Create response
            response = distributed_pb2.ForwardPassResponse(
                output_data=serialized.data,
                compression=serialized.compression,
                shape=serialized.shape,
                dtype=serialized.dtype,
                compute_time_ms=int(compute_time_ms)
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Forward pass failed",
                extra={
                    "request_id": request.request_id,
                    "pipeline_id": request.pipeline_id,
                    "worker_id": self.worker_id,
                    "error": str(e),
                    "operation": "forward_pass_error"
                }
            )
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Forward pass failed: {str(e)}"
            )
    
    async def ProcessLayers(
        self,
        request: distributed_pb2.DistributedInferenceRequest,
        context: grpc.aio.ServicerContext
    ) -> distributed_pb2.DistributedInferenceResponse:
        """Process inference on assigned layers (placeholder)"""
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "ProcessLayers not yet implemented"
        )
    
    async def StreamPipeline(
        self,
        request_iterator,
        context: grpc.aio.ServicerContext
    ):
        """Stream inference through pipeline (placeholder)"""
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "StreamPipeline not yet implemented"
        )
    
    async def TransferActivation(
        self,
        request: distributed_pb2.ActivationTransfer,
        context: grpc.aio.ServicerContext
    ) -> distributed_pb2.TransferAck:
        """Transfer activation to next node (placeholder)"""
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "TransferActivation not yet implemented"
        )
    
    async def CheckPipelineHealth(
        self,
        request: distributed_pb2.PipelineHealthRequest,
        context: grpc.aio.ServicerContext
    ) -> distributed_pb2.PipelineHealthResponse:
        """Check pipeline health (placeholder)"""
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "CheckPipelineHealth not yet implemented"
        )
