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
from proto.grpc_base import GrpcServer
from proto.errors import GrpcError, ErrorCode

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
            
            if metrics['success_rate'] < 0.9:
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
        
        logger.info(f"Initializing WorkerGrpcServer on port {port}")
    
    async def start(self):
        """Start the gRPC server"""
        try:
            # Create service implementation
            self.service_impl = InferenceServiceImpl(self.model_manager, self.inference_engine)
            
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
                    ('grpc.max_receive_message_length', 4 * 1024 * 1024),  # 4MB
                    ('grpc.max_send_message_length', 4 * 1024 * 1024),     # 4MB
                ]
            )
            
            # Add service to server
            from proto.inference_pb2_grpc import add_InferenceServiceServicer_to_server
            add_InferenceServiceServicer_to_server(self.service_impl, self.server)
            
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