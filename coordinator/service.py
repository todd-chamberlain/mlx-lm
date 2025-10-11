"""
Coordinator service implementation.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
import grpc

from proto.inference_pb2 import (
    RegistrationResponse,
    HeartbeatResponse,
    HealthCheckResponse,
    WorkerHealthInfo,
    InferenceResponse,
    StreamChunk
)
from proto.inference_pb2_grpc import CoordinatorServiceServicer
# For async support
from grpc import aio as grpc_aio
from proto.grpc_base import GRPCAsyncServerBase
from proto.errors import MLXError, MLXErrorCode, handle_grpc_error
from proto.validation import validate_message

from .worker_pool import WorkerPool
from .models import WorkerInfo, WorkerStatus
from .worker_selector import SelectionConfig
from .health_monitor import HealthThresholds
from .request_router import RequestRouter, RequestContext
from common.logging_system import get_logger, log_error
from .metrics import coordinator_metrics, GRPCMetricsInterceptor
from .sharding_coordinator import ShardingCoordinator, DownloadStatus


logger = logging.getLogger(__name__)


class CoordinatorService(CoordinatorServiceServicer):
    """Coordinator service implementation."""
    
    def __init__(
        self,
        worker_pool: WorkerPool,
        sharding_coordinator: Optional[ShardingCoordinator] = None
    ):
        self.worker_pool = worker_pool
        self.request_router = RequestRouter(worker_pool)
        self.sharding_coordinator = sharding_coordinator or ShardingCoordinator()
        logger.info("CoordinatorService initialized")
    
    async def RegisterWorker(self, request, context):
        """Register a new worker or update existing worker."""
        try:
            # Validate request
            errors = validate_message(request)
            if errors:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Validation errors: {errors}")
                return RegistrationResponse(
                    success=False,
                    message=f"Validation failed: {errors}"
                )
            
            logger.info(f"Registering worker: {request.worker_id} at {request.address}")
            
            # Register worker
            result = self.worker_pool.register_worker(request)
            
            return RegistrationResponse(
                success=result['success'],
                message=result['message'],
                coordinator_id=result['coordinator_id']
            )
            
        except MLXError as e:
            logger.error(f"MLX error in RegisterWorker: {e}")
            handle_grpc_error(context, e)
            return RegistrationResponse(
                success=False,
                message=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error in RegisterWorker: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return RegistrationResponse(
                success=False,
                message="Internal server error"
            )
    
    def Heartbeat(self, request, context):
        """Process worker heartbeat."""
        try:
            # Validate request
            errors = validate_message(request)
            if errors:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Validation errors: {errors}")
                return HeartbeatResponse(
                    acknowledged=False,
                    message=f"Validation failed: {errors}"
                )
            
            logger.debug(f"Heartbeat from worker: {request.worker_id}")
            
            # Update heartbeat
            result = self.worker_pool.update_worker_heartbeat(request.worker_id, request)
            
            return HeartbeatResponse(
                acknowledged=result['acknowledged'],
                message=result['message'],
                heartbeat_interval=result['heartbeat_interval']
            )
            
        except MLXError as e:
            logger.error(f"MLX error in Heartbeat: {e}")
            handle_grpc_error(context, e)
            return HeartbeatResponse(
                acknowledged=False,
                message=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error in Heartbeat: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return HeartbeatResponse(
                acknowledged=False,
                message="Internal server error"
            )
    
    def HealthCheck(self, request, context):
        """Get coordinator and worker health status."""
        try:
            logger.debug("Health check requested")
            
            # Get current cluster health
            cluster_health = self.worker_pool.check_cluster_health()
            
            # Build worker health info if requested
            worker_health = []
            if request.include_workers:
                worker_health_info = self.worker_pool.get_all_worker_health()
                for worker_id, health_info in worker_health_info.items():
                    worker_health.append(WorkerHealthInfo(
                        worker_id=health_info.worker_id,
                        status=health_info.status.value,
                        last_heartbeat_seconds=health_info.time_since_heartbeat,
                        current_load=health_info.current_load
                    ))
            
            # Handle sharded models info if requested
            # For now, just acknowledge the field exists (implementation can be added later)
            if hasattr(request, 'include_sharded_models'):
                logger.debug(f"Sharded models info requested: {request.include_sharded_models}")
            
            return HealthCheckResponse(
                status=cluster_health.status.value,
                total_workers=cluster_health.total_workers,
                healthy_workers=cluster_health.healthy_workers,
                active_requests=cluster_health.active_requests,
                worker_health=worker_health
            )
            
        except Exception as e:
            logger.error(f"Unexpected error in HealthCheck: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return HealthCheckResponse(
                status="error",
                total_workers=0,
                healthy_workers=0,
                active_requests=0
            )
    
    def RouteInference(self, request, context):
        """Route inference request to appropriate worker."""
        try:
            # Validate request
            errors = validate_message(request)
            if errors:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Validation errors: {errors}")
                return InferenceResponse(
                    request_id=request.request_id,
                    text="",
                    worker_id="",
                    latency_ms=0.0
                )
            
            logger.debug(f"Routing inference request: {request.request_id}")
            
            # Extract client context
            client_ip = None
            user_agent = None
            
            # Get metadata from gRPC context
            metadata = dict(context.invocation_metadata())
            if 'x-forwarded-for' in metadata:
                client_ip = metadata['x-forwarded-for']
            elif 'x-real-ip' in metadata:
                client_ip = metadata['x-real-ip']
            
            if 'user-agent' in metadata:
                user_agent = metadata['user-agent']
            
            # Create request context
            request_context = RequestContext(
                request_id=request.request_id,
                model=request.model,
                client_ip=client_ip,
                user_agent=user_agent,
                timeout=30.0,
                max_retries=3
            )
            
            # Route request asynchronously
            # Create or get event loop for this thread
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            response = loop.run_until_complete(
                self.request_router.route_inference(request, request_context)
            )
            
            return response
            
        except MLXError as e:
            logger.error(f"MLX error in RouteInference: {e}")
            handle_grpc_error(context, e)
            return InferenceResponse(
                request_id=request.request_id,
                text="",
                worker_id="",
                latency_ms=0.0
            )
        except Exception as e:
            logger.error(f"Unexpected error in RouteInference: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return InferenceResponse(
                request_id=request.request_id,
                text="",
                worker_id="",
                latency_ms=0.0
            )
    
    def StreamInference(self, request, context):
        """Route streaming inference request to appropriate worker."""
        try:
            # Validate request
            errors = validate_message(request)
            if errors:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Validation errors: {errors}")
                return
            
            logger.debug(f"Routing streaming inference request: {request.request_id}")
            
            # Extract client context
            client_ip = None
            user_agent = None
            
            # Get metadata from gRPC context
            metadata = dict(context.invocation_metadata())
            if 'x-forwarded-for' in metadata:
                client_ip = metadata['x-forwarded-for']
            elif 'x-real-ip' in metadata:
                client_ip = metadata['x-real-ip']
            
            if 'user-agent' in metadata:
                user_agent = metadata['user-agent']
            
            # Create request context
            request_context = RequestContext(
                request_id=request.request_id,
                model=request.model,
                client_ip=client_ip,
                user_agent=user_agent,
                timeout=60.0,  # Longer timeout for streaming
                max_retries=1   # Limited retries for streaming
            )
            
            # Stream request asynchronously
            async def stream_generator():
                async for chunk in self.request_router.stream_inference(request, request_context):
                    yield chunk
            
            # Convert async generator to sync generator for gRPC
            loop = asyncio.get_event_loop()
            
            async_gen = stream_generator()
            while True:
                try:
                    chunk = loop.run_until_complete(async_gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
                except Exception as e:
                    logger.error(f"Error in streaming: {e}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(f"Streaming error: {str(e)}")
                    break
            
        except MLXError as e:
            logger.error(f"MLX error in StreamInference: {e}")
            handle_grpc_error(context, e)
        except Exception as e:
            logger.error(f"Unexpected error in StreamInference: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
    
    async def DownloadModel(self, request, context):
        """Coordinate model download across all workers."""
        try:
            from proto.inference_pb2 import ModelDownloadResponse, LayerAssignment
            import uuid
            
            download_id = str(uuid.uuid4())
            
            # Log initial request with structured data
            logger.info(
                "Model download requested",
                extra={
                    "model_name": request.model_name,
                    "download_id": download_id,
                    "operation": "download_model_start"
                }
            )
            
            workers = self.worker_pool.get_all_workers()
            healthy_workers = [w for w in workers if w.status == WorkerStatus.HEALTHY]
            
            if not healthy_workers:
                logger.warning(
                    "No healthy workers available for model download",
                    extra={
                        "model_name": request.model_name,
                        "download_id": download_id,
                        "total_workers": len(workers),
                        "operation": "download_model_no_workers"
                    }
                )
                return ModelDownloadResponse(
                    success=False,
                    message="No healthy workers available",
                    download_id=download_id,
                    workers_notified=0
                )
            
            # Check if sharding should be enabled
            enable_sharding = getattr(request, 'enable_sharding', False)
            model_size_gb = getattr(request, 'model_size_gb', 0.0)
            num_layers = getattr(request, 'num_layers', 0)
            
            # Log sharding decision inputs
            logger.info(
                "Evaluating sharding decision",
                extra={
                    "model_name": request.model_name,
                    "download_id": download_id,
                    "model_size_gb": model_size_gb,
                    "num_layers": num_layers,
                    "num_healthy_workers": len(healthy_workers),
                    "enable_sharding_requested": enable_sharding,
                    "operation": "sharding_decision_evaluation"
                }
            )
            
            # ============================================================
            # SHARDING DECISION POINT: Model Download Strategy
            # ============================================================
            # This is where the coordinator decides whether to shard a model
            # across workers or load it fully on each worker.
            #
            # Decision flow:
            # 1. Call ShardingCoordinator.should_shard_model() which checks:
            #    - Auto-sharding enabled in config
            #    - Sufficient workers available
            #    - Model size exceeds threshold
            # 2. Override with explicit enable_sharding parameter if provided
            #
            # If sharding is enabled:
            # - Create shard plan with layer assignments
            # - Send layer assignments to each worker
            # - Track shard download status
            #
            # If sharding is disabled:
            # - Send full model download to all workers
            # - Each worker loads complete model
            # ============================================================
            
            # Determine if sharding should be used
            should_shard = self.sharding_coordinator.should_shard_model(
                model_name=request.model_name,
                model_size_gb=model_size_gb,
                num_workers=len(healthy_workers)
            )
            
            # Override with explicit request parameter if provided
            if enable_sharding:
                should_shard = True
                logger.info(
                    "Sharding explicitly enabled by request",
                    extra={
                        "model_name": request.model_name,
                        "download_id": download_id,
                        "operation": "sharding_explicit_enable"
                    }
                )
            
            # Log final sharding decision
            logger.info(
                "Sharding decision made",
                extra={
                    "model_name": request.model_name,
                    "download_id": download_id,
                    "should_shard": should_shard,
                    "num_layers": num_layers,
                    "num_workers": len(healthy_workers),
                    "operation": "sharding_decision_final"
                }
            )
            
            # If sharding is enabled, create shard plan
            shard_metadata = None
            if should_shard and num_layers > 0:
                logger.info(
                    "Creating shard plan",
                    extra={
                        "model_name": request.model_name,
                        "download_id": download_id,
                        "num_layers": num_layers,
                        "num_workers": len(healthy_workers),
                        "operation": "shard_plan_creation_start"
                    }
                )
                
                # Prepare worker info for shard plan
                worker_info_list = []
                for worker in healthy_workers:
                    worker_dict = {
                        'worker_id': worker.worker_id,
                        'address': worker.address
                    }
                    # Add available memory if present
                    if hasattr(worker, 'available_memory_gb'):
                        worker_dict['available_memory'] = worker.available_memory_gb
                    worker_info_list.append(worker_dict)
                
                # Create shard plan using ShardingCoordinator
                shard_metadata = self.sharding_coordinator.create_shard_plan(
                    model_name=request.model_name,
                    num_layers=num_layers,
                    workers=worker_info_list
                )
                
                # Log shard plan creation success
                logger.info(
                    "Shard plan created successfully",
                    extra={
                        "model_name": request.model_name,
                        "download_id": download_id,
                        "num_shards": len(shard_metadata.shard_assignments),
                        "total_layers": shard_metadata.total_layers,
                        "operation": "shard_plan_created"
                    }
                )
            elif should_shard and num_layers == 0:
                logger.warning(
                    "Sharding requested but num_layers is 0, skipping shard plan",
                    extra={
                        "model_name": request.model_name,
                        "download_id": download_id,
                        "operation": "shard_plan_skipped_no_layers"
                    }
                )
            
            # Send download requests to workers
            workers_notified = 0
            
            logger.info(
                "Sending download requests to workers",
                extra={
                    "model_name": request.model_name,
                    "download_id": download_id,
                    "num_workers": len(healthy_workers),
                    "is_sharded": shard_metadata is not None,
                    "operation": "worker_notification_start"
                }
            )
            
            for worker in healthy_workers:
                try:
                    channel = grpc.aio.insecure_channel(worker.address)
                    from proto.inference_pb2_grpc import InferenceServiceStub
                    stub = InferenceServiceStub(channel)
                    
                    # Create worker-specific request
                    worker_request = type(request)()
                    worker_request.CopyFrom(request)
                    
                    # Add layer assignment if sharding
                    if shard_metadata:
                        assignment = shard_metadata.get_assignment(worker.worker_id)
                        if assignment:
                            # Log layer assignment being sent
                            logger.info(
                                "Sending layer assignment to worker",
                                extra={
                                    "model_name": request.model_name,
                                    "download_id": download_id,
                                    "worker_id": worker.worker_id,
                                    "start_layer": assignment.start_layer,
                                    "end_layer": assignment.end_layer,
                                    "num_layers": assignment.num_layers,
                                    "layer_range": f"{assignment.start_layer}-{assignment.end_layer}",
                                    "next_worker": assignment.next_node_id,
                                    "operation": "layer_assignment_sent"
                                }
                            )
                            
                            # Set layer assignment fields if they exist in proto
                            if hasattr(worker_request, 'layer_assignment'):
                                layer_assignment = LayerAssignment(
                                    start_layer=assignment.start_layer,
                                    end_layer=assignment.end_layer,
                                    next_worker_id=assignment.next_node_id or "",
                                    next_worker_address=assignment.next_node_address or ""
                                )
                                worker_request.layer_assignment.CopyFrom(layer_assignment)
                            
                            # Mark download as in progress
                            self.sharding_coordinator.update_shard_status(
                                model_name=request.model_name,
                                worker_id=worker.worker_id,
                                download_status=DownloadStatus.IN_PROGRESS
                            )
                    
                    worker_response = await stub.DownloadModel(worker_request, timeout=5.0)
                    
                    if worker_response.success:
                        workers_notified += 1
                        
                        logger.info(
                            "Worker download initiated successfully",
                            extra={
                                "model_name": request.model_name,
                                "download_id": download_id,
                                "worker_id": worker.worker_id,
                                "is_sharded": shard_metadata is not None,
                                "operation": "worker_download_success"
                            }
                        )
                        
                        # Update shard status if sharding
                        if shard_metadata:
                            self.sharding_coordinator.update_shard_status(
                                model_name=request.model_name,
                                worker_id=worker.worker_id,
                                download_status=DownloadStatus.COMPLETED
                            )
                    else:
                        logger.warning(
                            "Worker download failed",
                            extra={
                                "model_name": request.model_name,
                                "download_id": download_id,
                                "worker_id": worker.worker_id,
                                "error_message": worker_response.message,
                                "operation": "worker_download_failed"
                            }
                        )
                        
                        # Mark as failed if sharding
                        if shard_metadata:
                            self.sharding_coordinator.update_shard_status(
                                model_name=request.model_name,
                                worker_id=worker.worker_id,
                                download_status=DownloadStatus.FAILED
                            )
                    
                    await channel.close()
                except Exception as e:
                    # Log error with shard context
                    error_context = {
                        "model_name": request.model_name,
                        "worker_id": worker.worker_id,
                        "worker_address": worker.address,
                        "operation": "shard_download_error"
                    }
                    
                    if shard_metadata:
                        assignment = shard_metadata.get_assignment(worker.worker_id)
                        if assignment:
                            error_context.update({
                                "start_layer": assignment.start_layer,
                                "end_layer": assignment.end_layer,
                                "layer_range": f"{assignment.start_layer}-{assignment.end_layer}",
                                "is_sharded": True
                            })
                    
                    log_error(
                        logger,
                        "shard_download_failed",
                        f"Failed to notify worker {worker.worker_id} for model download",
                        exception=e,
                        **error_context
                    )
                    
                    # Mark as failed if sharding
                    if shard_metadata:
                        self.sharding_coordinator.update_shard_status(
                            model_name=request.model_name,
                            worker_id=worker.worker_id,
                            download_status=DownloadStatus.FAILED
                        )
            
            message = f"Model download initiated on {workers_notified}/{len(healthy_workers)} workers"
            if shard_metadata:
                message += f" (sharded across {len(shard_metadata.shard_assignments)} workers)"
            
            # Log final download status
            logger.info(
                "Model download coordination complete",
                extra={
                    "model_name": request.model_name,
                    "download_id": download_id,
                    "workers_notified": workers_notified,
                    "total_workers": len(healthy_workers),
                    "is_sharded": shard_metadata is not None,
                    "num_shards": len(shard_metadata.shard_assignments) if shard_metadata else 0,
                    "success": workers_notified > 0,
                    "operation": "download_model_complete"
                }
            )
            
            return ModelDownloadResponse(
                success=workers_notified > 0,
                message=message,
                download_id=download_id,
                workers_notified=workers_notified,
                is_sharded=shard_metadata is not None
            )
        except Exception as e:
            logger.error(f"Error in DownloadModel: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return ModelDownloadResponse(success=False, message=str(e), download_id="", workers_notified=0)
    
    async def GetDownloadStatus(self, request, context):
        """Get model download status from all workers."""
        try:
            from proto.inference_pb2 import ModelStatusResponse, WorkerModelStatus
            
            workers = self.worker_pool.get_all_workers()
            worker_statuses = []
            
            for worker in workers:
                try:
                    channel = grpc.aio.insecure_channel(worker.address)
                    from proto.inference_pb2_grpc import InferenceServiceStub
                    stub = InferenceServiceStub(channel)
                    worker_response = await stub.GetModelStatus(request, timeout=5.0)
                    
                    worker_statuses.append(WorkerModelStatus(
                        worker_id=worker.worker_id,
                        status=worker_response.status,
                        progress_percent=worker_response.progress_percent,
                        error=""
                    ))
                    await channel.close()
                except Exception as e:
                    worker_statuses.append(WorkerModelStatus(
                        worker_id=worker.worker_id,
                        status="error",
                        progress_percent=0.0,
                        error=str(e)
                    ))
            
            if not worker_statuses:
                overall_status = "not_found"
            elif all(ws.status == "ready" for ws in worker_statuses):
                overall_status = "ready"
            elif any(ws.status == "downloading" for ws in worker_statuses):
                overall_status = "downloading"
            else:
                overall_status = "unknown"
            
            avg_progress = sum(ws.progress_percent for ws in worker_statuses) / len(worker_statuses) if worker_statuses else 0.0
            
            return ModelStatusResponse(
                model_name=request.model_name,
                status=overall_status,
                progress_percent=avg_progress,
                size_bytes=0,
                worker_status=worker_statuses
            )
        except Exception as e:
            logger.error(f"Error in GetDownloadStatus: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return ModelStatusResponse(model_name=request.model_name, status="error", progress_percent=0.0, size_bytes=0, worker_status=[])
    
    async def GetShardStatus(self, request, context):
        """Get shard status for a model."""
        try:
            from proto.inference_pb2 import ShardStatusResponse, ShardAssignmentStatus
            
            logger.info(
                "Shard status requested",
                extra={
                    "model_name": request.model_name,
                    "operation": "get_shard_status"
                }
            )
            
            # Get shard metadata from coordinator
            shard_metadata = self.sharding_coordinator.get_shard_metadata(request.model_name)
            
            if not shard_metadata:
                # Model is not sharded
                logger.info(
                    "Model is not sharded",
                    extra={
                        "model_name": request.model_name,
                        "operation": "get_shard_status_not_sharded"
                    }
                )
                return ShardStatusResponse(
                    is_sharded=False,
                    model_name=request.model_name,
                    total_layers=0,
                    status="not_sharded",
                    shard_assignments=[]
                )
            
            # Build shard assignment status list
            shard_assignments = []
            for worker_id, assignment in shard_metadata.shard_assignments.items():
                shard_assignments.append(ShardAssignmentStatus(
                    worker_id=worker_id,
                    start_layer=assignment.start_layer,
                    end_layer=assignment.end_layer,
                    download_status=assignment.download_status.value,
                    next_worker_id=assignment.next_node_id or ""
                ))
            
            logger.info(
                "Shard status retrieved",
                extra={
                    "model_name": request.model_name,
                    "is_sharded": True,
                    "total_layers": shard_metadata.total_layers,
                    "num_shards": len(shard_assignments),
                    "status": shard_metadata.status.value,
                    "operation": "get_shard_status_success"
                }
            )
            
            return ShardStatusResponse(
                is_sharded=True,
                model_name=shard_metadata.model_name,
                total_layers=shard_metadata.total_layers,
                status=shard_metadata.status.value,
                shard_assignments=shard_assignments
            )
            
        except Exception as e:
            logger.error(f"Error in GetShardStatus: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return ShardStatusResponse(
                is_sharded=False,
                model_name=request.model_name,
                total_layers=0,
                status="error",
                shard_assignments=[]
            )
    
    async def DeleteModel(self, request, context):
        """Delete model from all workers holding shards."""
        try:
            from proto.inference_pb2 import ModelDeleteResponse, WorkerDeleteStatus
            
            logger.info(
                "Model deletion requested",
                extra={
                    "model_name": request.model_name,
                    "operation": "delete_model_start"
                }
            )
            
            # Check if model is sharded in coordinator metadata
            shard_metadata = self.sharding_coordinator.get_shard_metadata(request.model_name)
            is_sharded = shard_metadata is not None
            
            # Get workers to delete from
            if is_sharded:
                # Get workers holding shards
                worker_ids = list(shard_metadata.shard_assignments.keys())
                workers = [
                    w for w in self.worker_pool.get_all_workers()
                    if w.worker_id in worker_ids
                ]
                logger.info(
                    "Deleting sharded model from specific workers",
                    extra={
                        "model_name": request.model_name,
                        "num_workers": len(workers),
                        "worker_ids": worker_ids,
                        "is_sharded": True,
                        "operation": "delete_sharded_model"
                    }
                )
            else:
                # Delete from all workers
                workers = self.worker_pool.get_all_workers()
                logger.info(
                    "Deleting non-sharded model from all workers",
                    extra={
                        "model_name": request.model_name,
                        "num_workers": len(workers),
                        "is_sharded": False,
                        "operation": "delete_non_sharded_model"
                    }
                )
            
            if not workers:
                logger.warning(
                    "No workers available for model deletion",
                    extra={
                        "model_name": request.model_name,
                        "operation": "delete_model_no_workers"
                    }
                )
                return ModelDeleteResponse(
                    success=False,
                    message="No workers available",
                    workers_deleted=0,
                    worker_status=[]
                )
            
            # Send delete requests to all workers holding the model
            worker_statuses = []
            workers_deleted = 0
            failed_workers = []
            
            for worker in workers:
                try:
                    channel = grpc.aio.insecure_channel(worker.address)
                    from proto.inference_pb2_grpc import InferenceServiceStub
                    stub = InferenceServiceStub(channel)
                    
                    worker_response = await stub.DeleteModel(request, timeout=10.0)
                    
                    worker_statuses.append(WorkerDeleteStatus(
                        worker_id=worker.worker_id,
                        success=worker_response.success,
                        error="" if worker_response.success else worker_response.message
                    ))
                    
                    if worker_response.success:
                        workers_deleted += 1
                        logger.info(
                            "Model deleted from worker",
                            extra={
                                "model_name": request.model_name,
                                "worker_id": worker.worker_id,
                                "operation": "worker_delete_success"
                            }
                        )
                    else:
                        failed_workers.append(worker.worker_id)
                        logger.warning(
                            "Failed to delete model from worker",
                            extra={
                                "model_name": request.model_name,
                                "worker_id": worker.worker_id,
                                "error": worker_response.message,
                                "operation": "worker_delete_failed"
                            }
                        )
                    
                    await channel.close()
                    
                except Exception as e:
                    failed_workers.append(worker.worker_id)
                    logger.error(
                        "Exception deleting model from worker",
                        extra={
                            "model_name": request.model_name,
                            "worker_id": worker.worker_id,
                            "error": str(e),
                            "operation": "worker_delete_exception"
                        },
                        exc_info=True
                    )
                    worker_statuses.append(WorkerDeleteStatus(
                        worker_id=worker.worker_id,
                        success=False,
                        error=str(e)
                    ))
            
            # Verify deletion on all workers
            all_deleted = workers_deleted == len(workers)
            
            if not all_deleted:
                logger.warning(
                    "Not all workers successfully deleted model",
                    extra={
                        "model_name": request.model_name,
                        "workers_deleted": workers_deleted,
                        "total_workers": len(workers),
                        "failed_workers": failed_workers,
                        "operation": "delete_verification_failed"
                    }
                )
            else:
                logger.info(
                    "All workers successfully deleted model",
                    extra={
                        "model_name": request.model_name,
                        "workers_deleted": workers_deleted,
                        "operation": "delete_verification_success"
                    }
                )
            
            # Clean up shard metadata from coordinator if sharded
            if is_sharded:
                if all_deleted:
                    # Only clean up metadata if all shards were successfully deleted
                    cleanup_success = self.sharding_coordinator.remove_shard_metadata(
                        request.model_name
                    )
                    if cleanup_success:
                        logger.info(
                            "Cleaned up shard metadata",
                            extra={
                                "model_name": request.model_name,
                                "operation": "shard_metadata_cleanup_success"
                            }
                        )
                    else:
                        logger.warning(
                            "Failed to clean up shard metadata",
                            extra={
                                "model_name": request.model_name,
                                "operation": "shard_metadata_cleanup_failed"
                            }
                        )
                else:
                    logger.warning(
                        "Skipping shard metadata cleanup due to incomplete deletion",
                        extra={
                            "model_name": request.model_name,
                            "workers_deleted": workers_deleted,
                            "total_workers": len(workers),
                            "operation": "shard_metadata_cleanup_skipped"
                        }
                    )
            
            message = f"Model deleted from {workers_deleted}/{len(workers)} workers"
            if is_sharded:
                message += " (sharded model)"
            if not all_deleted:
                message += f". Failed on: {', '.join(failed_workers)}"
            
            logger.info(
                "Model deletion complete",
                extra={
                    "model_name": request.model_name,
                    "workers_deleted": workers_deleted,
                    "total_workers": len(workers),
                    "all_deleted": all_deleted,
                    "is_sharded": is_sharded,
                    "operation": "delete_model_complete"
                }
            )
            
            return ModelDeleteResponse(
                success=all_deleted,
                message=message,
                workers_deleted=workers_deleted,
                worker_status=worker_statuses
            )
            
        except Exception as e:
            logger.error(f"Error in DeleteModel: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return ModelDeleteResponse(
                success=False,
                message=str(e),
                workers_deleted=0,
                worker_status=[]
            )


class CoordinatorServer(GRPCAsyncServerBase):
    """Coordinator gRPC server."""
    
    def __init__(
        self,
        port: int = 50051,
        max_workers: int = 10,
        worker_pool_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[SelectionConfig] = None,
        health_thresholds: Optional[HealthThresholds] = None,
        sharding_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(port, max_workers)
        
        # Initialize worker pool
        pool_config = worker_pool_config or {}
        pool_config['selection_config'] = selection_config
        pool_config['health_thresholds'] = health_thresholds
        self.worker_pool = WorkerPool(**pool_config)
        
        # Initialize and validate sharding configuration
        shard_config = sharding_config or {}
        
        # Extract sharding parameters with defaults
        enable_auto_sharding = shard_config.get('enable_auto_sharding', False)
        shard_threshold_gb = shard_config.get('shard_threshold_gb', 30.0)
        strategy = shard_config.get('strategy', 'balanced')
        min_workers_for_sharding = shard_config.get('min_workers_for_sharding', 2)
        
        # Validate sharding configuration
        self._validate_sharding_config(
            enable_auto_sharding,
            shard_threshold_gb,
            strategy,
            min_workers_for_sharding
        )
        
        # Initialize sharding coordinator
        self.sharding_coordinator = ShardingCoordinator(
            enable_auto_sharding=enable_auto_sharding,
            shard_threshold_gb=shard_threshold_gb,
            strategy=strategy,
            min_workers_for_sharding=min_workers_for_sharding
        )
        
        # Log sharding configuration on startup
        logger.info(
            "Sharding configuration loaded",
            extra={
                "enable_auto_sharding": enable_auto_sharding,
                "shard_threshold_gb": shard_threshold_gb,
                "strategy": strategy,
                "min_workers_for_sharding": min_workers_for_sharding,
                "operation": "sharding_config_loaded"
            }
        )
        
        # Initialize service
        self.coordinator_service = CoordinatorService(
            self.worker_pool,
            self.sharding_coordinator
        )
        
        logger.info(f"CoordinatorServer initialized on port {port}")
    
    def _validate_sharding_config(
        self,
        enable_auto_sharding: bool,
        shard_threshold_gb: float,
        strategy: str,
        min_workers_for_sharding: int
    ) -> None:
        """
        Validate sharding configuration parameters.
        
        Args:
            enable_auto_sharding: Whether auto-sharding is enabled
            shard_threshold_gb: Minimum model size for sharding
            strategy: Sharding strategy
            min_workers_for_sharding: Minimum workers required
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate shard_threshold_gb
        if shard_threshold_gb <= 0:
            raise ValueError(
                f"shard_threshold_gb must be positive, got {shard_threshold_gb}"
            )
        
        if shard_threshold_gb < 1.0:
            logger.warning(
                "shard_threshold_gb is very small, may cause excessive sharding",
                extra={
                    "shard_threshold_gb": shard_threshold_gb,
                    "operation": "config_validation_warning"
                }
            )
        
        if shard_threshold_gb > 1000.0:
            logger.warning(
                "shard_threshold_gb is very large, sharding may rarely trigger",
                extra={
                    "shard_threshold_gb": shard_threshold_gb,
                    "operation": "config_validation_warning"
                }
            )
        
        # Validate strategy
        valid_strategies = {'balanced', 'memory_aware'}
        if strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {valid_strategies}, got '{strategy}'"
            )
        
        # Validate min_workers_for_sharding
        if min_workers_for_sharding < 2:
            raise ValueError(
                f"min_workers_for_sharding must be at least 2, got {min_workers_for_sharding}"
            )
        
        if min_workers_for_sharding > 10:
            logger.warning(
                "min_workers_for_sharding is high, sharding may rarely trigger",
                extra={
                    "min_workers_for_sharding": min_workers_for_sharding,
                    "operation": "config_validation_warning"
                }
            )
        
        # Log validation success
        logger.info(
            "Sharding configuration validated successfully",
            extra={
                "enable_auto_sharding": enable_auto_sharding,
                "shard_threshold_gb": shard_threshold_gb,
                "strategy": strategy,
                "min_workers_for_sharding": min_workers_for_sharding,
                "operation": "config_validation_success"
            }
        )
    
    async def add_servicers(self, server: grpc_aio.Server) -> None:
        """Add coordinator service to the server."""
        from proto.inference_pb2_grpc import add_CoordinatorServiceServicer_to_server
        add_CoordinatorServiceServicer_to_server(self.coordinator_service, server)
        logger.info("CoordinatorService added to server")
    
    async def start(self) -> None:
        """Start the coordinator server and worker pool."""
        logger.info("Starting coordinator server")
        
        # Start worker pool
        await self.worker_pool.start()
        
        # Start gRPC server
        await super().start()
    
    async def stop(self, grace_period: float = 5.0) -> None:
        """Stop the coordinator server and worker pool."""
        logger.info("Stopping coordinator server")
        
        # Stop gRPC server
        await super().stop(grace_period)
        
        # Close request router
        await self.coordinator_service.request_router.close()
        
        # Stop worker pool
        await self.worker_pool.stop()
        
        logger.info("Coordinator server stopped")
    
    def get_worker_pool(self) -> WorkerPool:
        """Get the worker pool instance."""
        return self.worker_pool
    
    def get_coordinator_service(self) -> CoordinatorService:
        """Get the coordinator service instance."""
        return self.coordinator_service
    
    def get_request_router(self) -> RequestRouter:
        """Get the request router instance."""
        return self.coordinator_service.request_router
    
    def get_sharding_coordinator(self) -> ShardingCoordinator:
        """Get the sharding coordinator instance."""
        return self.sharding_coordinator
