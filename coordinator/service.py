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
from proto.grpc_base import GRPCServerBase
from proto.errors import MLXError, MLXErrorCode, handle_grpc_error
from proto.validation import validate_message

from .worker_pool import WorkerPool
from .models import WorkerInfo, WorkerStatus
from .worker_selector import SelectionConfig
from .health_monitor import HealthThresholds
from .request_router import RequestRouter, RequestContext


logger = logging.getLogger(__name__)


class CoordinatorService(CoordinatorServiceServicer):
    """Coordinator service implementation."""
    
    def __init__(self, worker_pool: WorkerPool):
        self.worker_pool = worker_pool
        self.request_router = RequestRouter(worker_pool)
        logger.info("CoordinatorService initialized")
    
    def RegisterWorker(self, request, context):
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
                timeout=30.0,  # TODO: Make configurable
                max_retries=3   # TODO: Make configurable
            )
            
            # Route request asynchronously
            loop = asyncio.get_event_loop()
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


class CoordinatorServer(GRPCServerBase):
    """Coordinator gRPC server."""
    
    def __init__(
        self,
        port: int = 50051,
        max_workers: int = 10,
        worker_pool_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[SelectionConfig] = None,
        health_thresholds: Optional[HealthThresholds] = None
    ):
        super().__init__(port, max_workers)
        
        # Initialize worker pool
        pool_config = worker_pool_config or {}
        pool_config['selection_config'] = selection_config
        pool_config['health_thresholds'] = health_thresholds
        self.worker_pool = WorkerPool(**pool_config)
        
        # Initialize service
        self.coordinator_service = CoordinatorService(self.worker_pool)
        
        logger.info(f"CoordinatorServer initialized on port {port}")
    
    def add_servicers(self, server: grpc.Server) -> None:
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