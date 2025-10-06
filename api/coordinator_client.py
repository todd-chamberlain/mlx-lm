"""
gRPC client for communicating with the coordinator service.
"""

import asyncio
import logging
import uuid
import random
from typing import List, Optional, AsyncGenerator
import grpc
from grpc import aio

from proto.inference_pb2 import (
    InferenceRequest,
    InferenceResponse,
    StreamChunk,
    HealthCheckRequest,
    HealthCheckResponse,
    Empty
)
from proto.inference_pb2_grpc import CoordinatorServiceStub
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatMessage
)


logger = logging.getLogger(__name__)


class CoordinatorClient:
    """gRPC client for coordinator service communication with connection pooling and retry logic."""
    
    def __init__(self, coordinator_host: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize coordinator client.
        
        Args:
            coordinator_host: Host and port of coordinator service (e.g., "coordinator:50051")
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.coordinator_host = coordinator_host
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.channel: Optional[aio.Channel] = None
        self.stub: Optional[CoordinatorServiceStub] = None
        self._connection_lock = asyncio.Lock()
        self._request_id_counter = 0
    
    async def connect(self) -> None:
        """Establish connection to coordinator service."""
        async with self._connection_lock:
            if self.channel is not None:
                return
            
            try:
                # Create gRPC channel with retry configuration
                options = [
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.http2.min_time_between_pings_ms', 10000),
                    ('grpc.http2.min_ping_interval_without_data_ms', 300000)
                ]
                
                self.channel = aio.insecure_channel(self.coordinator_host, options=options)
                self.stub = CoordinatorServiceStub(self.channel)
                
                # Test connection
                await self._test_connection()
                logger.info(f"Connected to coordinator at {self.coordinator_host}")
                
            except Exception as e:
                logger.error(f"Failed to connect to coordinator: {e}")
                if self.channel:
                    await self.channel.close()
                    self.channel = None
                    self.stub = None
                raise
    
    async def close(self) -> None:
        """Close connection to coordinator service."""
        async with self._connection_lock:
            if self.channel:
                await self.channel.close()
                self.channel = None
                self.stub = None
                logger.info("Disconnected from coordinator")
    
    async def _test_connection(self) -> None:
        """Test connection to coordinator service."""
        if not self.stub:
            raise RuntimeError("Not connected to coordinator")
        
        try:
            request = HealthCheckRequest(include_workers=False)
            await self.stub.HealthCheck(request, timeout=5.0)
        except grpc.RpcError as e:
            raise RuntimeError(f"Coordinator health check failed: {e}")
    
    async def _ensure_connected(self) -> None:
        """Ensure connection to coordinator service."""
        if self.channel is None or self.stub is None:
            await self.connect()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID with tracking."""
        self._request_id_counter += 1
        return f"req_{int(asyncio.get_event_loop().time())}_{self._request_id_counter}"
    
    def _convert_to_grpc_request(self, request: ChatCompletionRequest) -> InferenceRequest:
        """Convert ChatCompletionRequest to gRPC InferenceRequest."""
        request_id = request.id or self._generate_request_id()
        
        return InferenceRequest(
            request_id=request_id,
            prompt=request.to_prompt(),
            max_tokens=request.max_tokens or 100,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            seed=request.seed or 0,
            model=request.model,
            use_cache=True
        )
    
    def _convert_from_grpc_response(
        self, 
        grpc_response: InferenceResponse, 
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Convert gRPC InferenceResponse to ChatCompletionResponse."""
        return ChatCompletionResponse(
            id=grpc_response.request_id,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=grpc_response.text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=grpc_response.prompt_tokens,
                completion_tokens=grpc_response.completion_tokens,
                total_tokens=grpc_response.prompt_tokens + grpc_response.completion_tokens
            ),
            system_fingerprint=f"worker-{grpc_response.worker_id}"
        )
    
    def _convert_stream_chunk(
        self, 
        chunk: StreamChunk, 
        request: ChatCompletionRequest
    ) -> ChatCompletionStreamResponse:
        """Convert gRPC StreamChunk to ChatCompletionStreamResponse."""
        delta = {}
        if chunk.text:
            delta["content"] = chunk.text
        
        return ChatCompletionStreamResponse(
            id=chunk.request_id,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=delta,
                    finish_reason="stop" if chunk.is_final else None
                )
            ]
        )
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RuntimeError: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
                
            except grpc.RpcError as e:
                last_exception = e
                
                # Don't retry on certain error codes
                if e.code() in [grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.UNAUTHENTICATED]:
                    raise RuntimeError(f"Non-retryable error: {e.details()}")
                
                if attempt < self.max_retries:
                    # Exponential backoff with jitter
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                    
                    # Try to reconnect on connection errors
                    if e.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]:
                        try:
                            await self.close()
                            await self.connect()
                        except Exception as reconnect_error:
                            logger.warning(f"Reconnection failed: {reconnect_error}")
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
        
        # All retries failed
        if isinstance(last_exception, grpc.RpcError):
            raise RuntimeError(f"Request failed after {self.max_retries + 1} attempts: {last_exception.details()}")
        else:
            raise RuntimeError(f"Request failed after {self.max_retries + 1} attempts: {str(last_exception)}")
    
    async def route_inference(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Route inference request to coordinator with retry logic.
        
        Args:
            request: Chat completion request
            
        Returns:
            Chat completion response
            
        Raises:
            RuntimeError: If inference fails
        """
        async def _do_inference():
            await self._ensure_connected()
            grpc_request = self._convert_to_grpc_request(request)
            grpc_response = await self.stub.RouteInference(grpc_request, timeout=60.0)
            return self._convert_from_grpc_response(grpc_response, request)
        
        return await self._retry_with_backoff(_do_inference)
    
    async def stream_inference(
        self, 
        request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        """
        Stream inference request to coordinator with retry logic.
        
        Args:
            request: Chat completion request
            
        Yields:
            Chat completion stream responses
            
        Raises:
            RuntimeError: If streaming fails
        """
        async def _do_stream():
            await self._ensure_connected()
            grpc_request = self._convert_to_grpc_request(request)
            
            async for chunk in self.stub.StreamInference(grpc_request, timeout=120.0):
                yield self._convert_stream_chunk(chunk, request)
        
        # For streaming, we implement a simpler retry mechanism
        # since we can't easily retry in the middle of a stream
        for attempt in range(self.max_retries + 1):
            try:
                async for response in _do_stream():
                    yield response
                return  # Success, exit retry loop
                
            except grpc.RpcError as e:
                if e.code() in [grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.UNAUTHENTICATED]:
                    raise RuntimeError(f"Non-retryable streaming error: {e.details()}")
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Streaming attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                    
                    # Try to reconnect
                    if e.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]:
                        try:
                            await self.close()
                            await self.connect()
                        except Exception as reconnect_error:
                            logger.warning(f"Reconnection failed: {reconnect_error}")
                else:
                    logger.error(f"All streaming attempts failed: {e}")
                    raise RuntimeError(f"Streaming failed after {self.max_retries + 1} attempts: {e.details()}")
                    
            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Streaming attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All streaming attempts failed: {e}")
                    raise RuntimeError(f"Streaming failed after {self.max_retries + 1} attempts: {str(e)}")
    
    async def get_health(self) -> HealthCheckResponse:
        """
        Get coordinator health status with retry logic.
        
        Returns:
            Health check response
            
        Raises:
            RuntimeError: If health check fails
        """
        async def _do_health_check():
            await self._ensure_connected()
            request = HealthCheckRequest(include_workers=True)
            return await self.stub.HealthCheck(request, timeout=10.0)
        
        return await self._retry_with_backoff(_do_health_check)
    
    async def get_available_models(self) -> List[str]:
        """
        Get list of available models from coordinator.
        
        Returns:
            List of model names
            
        Raises:
            RuntimeError: If model listing fails
        """
        await self._ensure_connected()
        
        try:
            # For now, return a default set of models
            # This could be extended to query the coordinator for actual available models
            health = await self.get_health()
            if health.healthy_workers > 0:
                return [
                    "mistral-7b-instruct-v0.1",
                    "llama-2-7b-chat",
                    "codellama-7b-instruct"
                ]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            raise RuntimeError(f"Failed to get models: {str(e)}")