"""
Base classes for gRPC client and server implementations.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
import grpc
from grpc import aio as grpc_aio

from .errors import (
    MLXError, 
    MLXErrorCode, 
    handle_grpc_error, 
    wrap_grpc_call,
    map_grpc_status_to_mlx_error
)
from .validation import validate_message, ValidationError


logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


class GRPCAsyncServerBase:
    """Base class for async gRPC servers."""
    
    def __init__(
        self,
        port: int,
        max_workers: int = 10,
        max_receive_message_length: int = 4 * 1024 * 1024,
        max_send_message_length: int = 4 * 1024 * 1024,
        enable_reflection: bool = False
    ):
        self.port = port
        self.max_workers = max_workers
        self.max_receive_message_length = max_receive_message_length
        self.max_send_message_length = max_send_message_length
        self.enable_reflection = enable_reflection
        
        self.server: Optional[grpc_aio.Server] = None
        self._shutdown_event = asyncio.Event()
        
        # Server options
        self.options = [
            ('grpc.keepalive_time_ms', 60000),
            ('grpc.keepalive_timeout_ms', 10000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 2),
            ('grpc.http2.min_time_between_pings_ms', 60000),
            ('grpc.http2.min_ping_interval_without_data_ms', 60000),
            ('grpc.max_receive_message_length', max_receive_message_length),
            ('grpc.max_send_message_length', max_send_message_length),
        ]
    
    async def create_server(self) -> grpc_aio.Server:
        """Create and configure async gRPC server."""
        server = grpc_aio.server(options=self.options)
        
        # Add servicers (to be implemented by subclasses)
        await self.add_servicers(server)
        
        # Add insecure port
        listen_addr = f'[::]:{self.port}'
        server.add_insecure_port(listen_addr)
        
        return server
    
    async def add_servicers(self, server: grpc_aio.Server) -> None:
        """Add servicers to server. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement add_servicers")
    
    async def start(self) -> None:
        """Start the async gRPC server."""
        if self.server:
            logger.warning("Server already started")
            return
        
        self.server = await self.create_server()
        await self.server.start()
        logger.info(f"Async gRPC server started on port {self.port}")
    
    async def stop(self, grace: float = 5.0) -> None:
        """Stop the async gRPC server."""
        if not self.server:
            return
        
        logger.info("Stopping async gRPC server...")
        await self.server.stop(grace)
        self.server = None
        logger.info("Async gRPC server stopped")
    
    async def wait_for_termination(self) -> None:
        """Wait for server termination."""
        if self.server:
            await self.server.wait_for_termination()


class GRPCServerBase:
    """Base class for gRPC servers with common functionality."""
    
    def __init__(
        self,
        port: int,
        max_workers: int = 10,
        max_receive_message_length: int = 4 * 1024 * 1024,  # 4MB
        max_send_message_length: int = 4 * 1024 * 1024,     # 4MB
        enable_reflection: bool = False
    ):
        self.port = port
        self.max_workers = max_workers
        self.max_receive_message_length = max_receive_message_length
        self.max_send_message_length = max_send_message_length
        self.enable_reflection = enable_reflection
        
        self.server: Optional[grpc.Server] = None
        self._shutdown_event = asyncio.Event()
        
        # Server options - more lenient keepalive settings
        self.options = [
            ('grpc.keepalive_time_ms', 60000),  # 60 seconds
            ('grpc.keepalive_timeout_ms', 10000),  # 10 seconds
            ('grpc.keepalive_permit_without_calls', True),  # Allow keepalive without calls
            ('grpc.http2.max_pings_without_data', 2),  # Allow 2 pings without data
            ('grpc.http2.min_time_between_pings_ms', 60000),  # 60 seconds between pings
            ('grpc.http2.min_ping_interval_without_data_ms', 60000),  # 60 seconds for pings without data
            ('grpc.max_receive_message_length', max_receive_message_length),
            ('grpc.max_send_message_length', max_send_message_length),
        ]
    
    def create_server(self) -> grpc.Server:
        """Create and configure gRPC server."""
        server = grpc.server(
            ThreadPoolExecutor(max_workers=self.max_workers),
            options=self.options
        )
        
        # Add servicers (to be implemented by subclasses)
        self.add_servicers(server)
        
        # Enable reflection if requested
        if self.enable_reflection:
            from grpc_reflection.v1alpha import reflection
            service_names = [
                service.full_name
                for service in server._state.generic_handlers[0]._method_handlers.keys()
            ]
            reflection.enable_server_reflection(service_names, server)
        
        # Add insecure port
        listen_addr = f'[::]:{self.port}'
        server.add_insecure_port(listen_addr)
        
        logger.info(f"gRPC server configured on {listen_addr}")
        return server
    
    def add_servicers(self, server: grpc.Server) -> None:
        """Add servicers to the server. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement add_servicers")
    
    async def start(self) -> None:
        """Start the gRPC server."""
        self.server = self.create_server()
        self.server.start()
        logger.info(f"gRPC server started on port {self.port}")
        
        # Wait for shutdown
        await self._shutdown_event.wait()
    
    async def stop(self, grace_period: float = 5.0) -> None:
        """Stop the gRPC server gracefully."""
        if self.server:
            logger.info("Shutting down gRPC server...")
            self.server.stop(grace_period)
            self._shutdown_event.set()
            logger.info("gRPC server stopped")
    
    def validate_request(self, request: Any) -> None:
        """Validate incoming request."""
        try:
            errors = validate_message(request)
            if errors:
                raise ValidationError("Request validation failed", errors)
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            raise ValidationError(f"Request validation failed: {str(e)}")
    
    def handle_error(self, context: grpc.ServicerContext, error: Exception) -> None:
        """Handle errors in service methods."""
        if isinstance(error, MLXError):
            handle_grpc_error(context, error)
        elif isinstance(error, ValidationError):
            handle_grpc_error(context, error)
        else:
            mlx_error = MLXError(
                message=f"Internal server error: {str(error)}",
                error_code=MLXErrorCode.UNKNOWN,
                grpc_status=grpc.StatusCode.INTERNAL
            )
            handle_grpc_error(context, mlx_error)
        
        logger.error(f"gRPC service error: {error}", exc_info=True)


class GRPCClientBase:
    """Base class for gRPC clients with connection management and retry logic."""
    
    def __init__(
        self,
        target: str,
        retry_config: Optional[RetryConfig] = None,
        timeout: float = 30.0,
        max_receive_message_length: int = 4 * 1024 * 1024,  # 4MB
        max_send_message_length: int = 4 * 1024 * 1024,     # 4MB
        enable_keepalive: bool = False  # Disabled to avoid too_many_pings error
    ):
        self.target = target
        self.retry_config = retry_config or RetryConfig()
        self.timeout = timeout
        self.max_receive_message_length = max_receive_message_length
        self.max_send_message_length = max_send_message_length
        self.enable_keepalive = enable_keepalive
        
        self.channel: Optional[grpc.Channel] = None
        self._connection_lock = asyncio.Lock()
        self._last_connection_attempt = 0.0
        self._connection_backoff = 1.0
        
        # Channel options
        self.options = [
            ('grpc.max_receive_message_length', max_receive_message_length),
            ('grpc.max_send_message_length', max_send_message_length),
        ]
        
        if enable_keepalive:
            self.options.extend([
                ('grpc.keepalive_time_ms', 60000),  # 60 seconds - less aggressive
                ('grpc.keepalive_timeout_ms', 10000),  # 10 seconds timeout
                ('grpc.keepalive_permit_without_calls', False),  # Don't ping without calls
                ('grpc.http2.max_pings_without_data', 0),  # Unlimited pings with data
                ('grpc.http2.min_time_between_pings_ms', 120000),  # 2 minutes between pings
            ])
    
    async def connect(self) -> grpc.Channel:
        """Establish connection to gRPC server."""
        async with self._connection_lock:
            if self.channel and self._is_channel_ready():
                return self.channel
            
            # Close existing channel if any
            if self.channel:
                await self.close()
            
            try:
                logger.info(f"Connecting to gRPC server at {self.target}")
                self.channel = grpc.insecure_channel(self.target, options=self.options)
                
                # Wait for channel to be ready
                await self._wait_for_ready()
                
                logger.info(f"Connected to gRPC server at {self.target}")
                self._connection_backoff = 1.0  # Reset backoff on successful connection
                return self.channel
                
            except Exception as e:
                logger.error(f"Failed to connect to {self.target}: {e}")
                self._connection_backoff = min(self._connection_backoff * 2, 60.0)
                raise MLXError(
                    message=f"Failed to connect to {self.target}: {str(e)}",
                    error_code=MLXErrorCode.CONNECTION_FAILED,
                    grpc_status=grpc.StatusCode.UNAVAILABLE
                )
    
    def _is_channel_ready(self) -> bool:
        """Check if channel is ready."""
        if not self.channel:
            return False
        
        try:
            state = self.channel.get_state(try_to_connect=False)
            return state == grpc.ChannelConnectivity.READY
        except Exception:
            return False
    
    async def _wait_for_ready(self, timeout: float = 10.0) -> None:
        """Wait for channel to be ready."""
        if not self.channel:
            raise MLXError("No channel available", MLXErrorCode.CONNECTION_FAILED)
        
        try:
            await asyncio.wait_for(
                self._wait_for_channel_ready(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise MLXError(
                f"Channel not ready within {timeout}s",
                MLXErrorCode.CONNECTION_FAILED,
                grpc.StatusCode.DEADLINE_EXCEEDED
            )
    
    async def _wait_for_channel_ready(self) -> None:
        """Wait for channel to reach READY state."""
        while True:
            state = self.channel.get_state(try_to_connect=True)
            if state == grpc.ChannelConnectivity.READY:
                break
            elif state == grpc.ChannelConnectivity.TRANSIENT_FAILURE:
                await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.1)
    
    async def close(self) -> None:
        """Close the gRPC channel."""
        if self.channel:
            logger.info(f"Closing connection to {self.target}")
            self.channel.close()
            self.channel = None
    
    async def call_with_retry(
        self,
        method: Callable,
        request: Any,
        timeout: Optional[float] = None,
        metadata: Optional[List[tuple]] = None
    ) -> Any:
        """Make gRPC call with retry logic."""
        timeout = timeout or self.timeout
        last_error = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                # Ensure connection is established
                channel = await self.connect()
                
                # Validate request
                self.validate_request(request)
                
                # Make the call
                logger.debug(f"Making gRPC call (attempt {attempt + 1}/{self.retry_config.max_attempts})")
                
                response = await asyncio.wait_for(
                    method(request, metadata=metadata),
                    timeout=timeout
                )
                
                logger.debug("gRPC call successful")
                return response
                
            except grpc.RpcError as e:
                last_error = e
                error_code = map_grpc_status_to_mlx_error(e.code())
                
                # Don't retry certain errors
                if e.code() in [
                    grpc.StatusCode.INVALID_ARGUMENT,
                    grpc.StatusCode.NOT_FOUND,
                    grpc.StatusCode.PERMISSION_DENIED,
                    grpc.StatusCode.UNAUTHENTICATED,
                ]:
                    logger.error(f"Non-retryable gRPC error: {e}")
                    raise MLXError(
                        message=e.details() or "gRPC call failed",
                        error_code=error_code,
                        grpc_status=e.code()
                    )
                
                logger.warning(f"gRPC call failed (attempt {attempt + 1}): {e}")
                
                # Close channel on connection errors
                if e.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]:
                    await self.close()
                
                # Wait before retry (except on last attempt)
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.debug(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
            
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error in gRPC call: {e}")
                
                # Don't retry unexpected errors
                raise MLXError(
                    message=f"Unexpected error: {str(e)}",
                    error_code=MLXErrorCode.UNKNOWN,
                    grpc_status=grpc.StatusCode.INTERNAL
                )
        
        # All retries exhausted
        if isinstance(last_error, grpc.RpcError):
            error_code = map_grpc_status_to_mlx_error(last_error.code())
            raise MLXError(
                message=f"gRPC call failed after {self.retry_config.max_attempts} attempts: {last_error.details()}",
                error_code=error_code,
                grpc_status=last_error.code()
            )
        else:
            raise MLXError(
                message=f"Call failed after {self.retry_config.max_attempts} attempts: {str(last_error)}",
                error_code=MLXErrorCode.UNKNOWN,
                grpc_status=grpc.StatusCode.INTERNAL
            )
    
    def validate_request(self, request: Any) -> None:
        """Validate outgoing request."""
        try:
            errors = validate_message(request)
            if errors:
                raise ValidationError("Request validation failed", errors)
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            raise ValidationError(f"Request validation failed: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class MessageSerializer:
    """Utilities for message serialization and deserialization."""
    
    @staticmethod
    def serialize_message(message: Any) -> bytes:
        """Serialize protobuf message to bytes."""
        try:
            return message.SerializeToString()
        except Exception as e:
            raise MLXError(
                message=f"Failed to serialize message: {str(e)}",
                error_code=MLXErrorCode.UNKNOWN,
                grpc_status=grpc.StatusCode.INTERNAL
            )
    
    @staticmethod
    def deserialize_message(data: bytes, message_class: type) -> Any:
        """Deserialize bytes to protobuf message."""
        try:
            message = message_class()
            message.ParseFromString(data)
            return message
        except Exception as e:
            raise MLXError(
                message=f"Failed to deserialize message: {str(e)}",
                error_code=MLXErrorCode.UNKNOWN,
                grpc_status=grpc.StatusCode.INTERNAL
            )
    
    @staticmethod
    def message_to_dict(message: Any) -> Dict[str, Any]:
        """Convert protobuf message to dictionary."""
        try:
            from google.protobuf.json_format import MessageToDict
            return MessageToDict(message)
        except Exception as e:
            raise MLXError(
                message=f"Failed to convert message to dict: {str(e)}",
                error_code=MLXErrorCode.UNKNOWN,
                grpc_status=grpc.StatusCode.INTERNAL
            )
    
    @staticmethod
    def dict_to_message(data: Dict[str, Any], message_class: type) -> Any:
        """Convert dictionary to protobuf message."""
        try:
            from google.protobuf.json_format import ParseDict
            return ParseDict(data, message_class())
        except Exception as e:
            raise MLXError(
                message=f"Failed to convert dict to message: {str(e)}",
                error_code=MLXErrorCode.UNKNOWN,
                grpc_status=grpc.StatusCode.INTERNAL
            )


class HealthChecker:
    """Health checking utilities for gRPC services."""
    
    def __init__(self, client: GRPCClientBase):
        self.client = client
    
    async def check_health(self, timeout: float = 5.0) -> bool:
        """Check if the gRPC service is healthy."""
        try:
            # Try to establish connection
            await self.client.connect()
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def wait_for_healthy(
        self, 
        timeout: float = 30.0, 
        check_interval: float = 1.0
    ) -> bool:
        """Wait for service to become healthy."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await self.check_health():
                return True
            await asyncio.sleep(check_interval)
        
        return False