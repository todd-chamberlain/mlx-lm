"""
Request routing and retry logic for the coordinator service.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Set, Any, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import grpc

from proto.inference_pb2 import InferenceRequest, InferenceResponse, StreamChunk, Empty
from proto.inference_pb2_grpc import InferenceServiceStub
from proto.grpc_base import GRPCClientBase, RetryConfig
from proto.errors import MLXError, MLXErrorCode

from .models import WorkerInfo
from .worker_pool import WorkerPool


logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """Request status enumeration."""
    PENDING = "pending"
    ROUTING = "routing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RequestContext:
    """Context information for a request."""
    request_id: str
    model: Optional[str]
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    priority: int = 0
    timeout: float = 30.0
    max_retries: int = 3
    requirements: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'model': self.model,
            'client_ip': self.client_ip,
            'user_agent': self.user_agent,
            'priority': self.priority,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'requirements': self.requirements,
            'metadata': self.metadata
        }


@dataclass
class RequestTracker:
    """Tracks the lifecycle of a request."""
    request_id: str
    context: RequestContext
    status: RequestStatus = RequestStatus.PENDING
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    selected_worker: Optional[str] = None
    attempted_workers: Set[str] = field(default_factory=set)
    retry_count: int = 0
    error_messages: List[str] = field(default_factory=list)
    
    def duration(self) -> float:
        """Get request duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration(),
            'selected_worker': self.selected_worker,
            'attempted_workers': list(self.attempted_workers),
            'retry_count': self.retry_count,
            'error_messages': self.error_messages,
            'context': self.context.to_dict()
        }


class CircuitBreaker:
    """Circuit breaker for worker connections."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if (self.last_failure_time and 
                time.time() - self.last_failure_time > self.recovery_timeout):
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
                return True
            return False
        elif self.state == "HALF_OPEN":
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self) -> None:
        """Record successful execution."""
        if self.state == "HALF_OPEN":
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "HALF_OPEN":
            self.state = "OPEN"
        elif self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class WorkerClient:
    """Client for communicating with individual workers."""
    
    def __init__(self, worker_info: WorkerInfo):
        self.worker_info = worker_info
        self.client = GRPCClientBase(
            target=worker_info.address,
            retry_config=RetryConfig(max_attempts=1),  # No retries at client level
            timeout=30.0
        )
        self.circuit_breaker = CircuitBreaker()
        self._stub: Optional[InferenceServiceStub] = None
    
    async def connect(self) -> None:
        """Connect to the worker."""
        if not self.circuit_breaker.can_execute():
            raise MLXError(
                message=f"Circuit breaker open for worker {self.worker_info.worker_id}",
                error_code=MLXErrorCode.UNAVAILABLE
            )
        
        try:
            channel = await self.client.connect()
            self._stub = InferenceServiceStub(channel)
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise
    
    async def process_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Process inference request."""
        if not self._stub:
            await self.connect()
        
        try:
            response = await self._stub.ProcessInference(request)
            self.circuit_breaker.record_success()
            return response
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise
    
    async def stream_inference(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        """Process streaming inference request."""
        if not self._stub:
            await self.connect()
        
        try:
            async for chunk in self._stub.StreamInference(request):
                yield chunk
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise
    
    async def health_check(self) -> bool:
        """Check worker health."""
        if not self._stub:
            try:
                await self.connect()
            except Exception:
                return False
        
        try:
            await self._stub.HealthCheck(Empty())
            return True
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close connection."""
        if self.client:
            await self.client.close()
        self._stub = None


class RequestRouter:
    """Routes requests to appropriate workers with retry logic."""
    
    def __init__(
        self,
        worker_pool: WorkerPool,
        max_concurrent_requests: int = 1000,
        default_timeout: float = 30.0,
        default_max_retries: int = 3
    ):
        self.worker_pool = worker_pool
        self.max_concurrent_requests = max_concurrent_requests
        self.default_timeout = default_timeout
        self.default_max_retries = default_max_retries
        
        # Request tracking
        self._active_requests: Dict[str, RequestTracker] = {}
        self._request_history: List[RequestTracker] = []
        self._max_history_size = 1000
        
        # Worker clients
        self._worker_clients: Dict[str, WorkerClient] = {}
        
        # Concurrency control
        self._request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        logger.info(f"RequestRouter initialized with max {max_concurrent_requests} concurrent requests")
    
    async def route_inference(
        self,
        request: InferenceRequest,
        context: Optional[RequestContext] = None
    ) -> InferenceResponse:
        """Route inference request to appropriate worker."""
        # Create request context if not provided
        if context is None:
            context = RequestContext(
                request_id=request.request_id or self._generate_request_id(),
                model=request.model,
                timeout=self.default_timeout,
                max_retries=self.default_max_retries
            )
        
        # Ensure request has ID
        if not request.request_id:
            request.request_id = context.request_id
        
        # Create request tracker
        tracker = RequestTracker(
            request_id=context.request_id,
            context=context
        )
        
        # Add to active requests
        self._active_requests[context.request_id] = tracker
        
        try:
            # Acquire semaphore for concurrency control
            async with self._request_semaphore:
                tracker.status = RequestStatus.ROUTING
                
                # Route request with retries
                response = await self._route_with_retries(request, tracker)
                
                tracker.status = RequestStatus.COMPLETED
                tracker.end_time = time.time()
                
                return response
                
        except Exception as e:
            tracker.status = RequestStatus.FAILED
            tracker.end_time = time.time()
            tracker.error_messages.append(str(e))
            
            logger.error(f"Request {context.request_id} failed: {e}")
            raise
        
        finally:
            # Move to history and cleanup
            self._cleanup_request(tracker)
    
    async def stream_inference(
        self,
        request: InferenceRequest,
        context: Optional[RequestContext] = None
    ) -> AsyncIterator[StreamChunk]:
        """Route streaming inference request to appropriate worker."""
        # Create request context if not provided
        if context is None:
            context = RequestContext(
                request_id=request.request_id or self._generate_request_id(),
                model=request.model,
                timeout=self.default_timeout,
                max_retries=self.default_max_retries
            )
        
        # Ensure request has ID
        if not request.request_id:
            request.request_id = context.request_id
        
        # Create request tracker
        tracker = RequestTracker(
            request_id=context.request_id,
            context=context
        )
        
        # Add to active requests
        self._active_requests[context.request_id] = tracker
        
        try:
            # Acquire semaphore for concurrency control
            async with self._request_semaphore:
                tracker.status = RequestStatus.ROUTING
                
                # Stream request with retries
                async for chunk in self._stream_with_retries(request, tracker):
                    yield chunk
                
                tracker.status = RequestStatus.COMPLETED
                tracker.end_time = time.time()
                
        except Exception as e:
            tracker.status = RequestStatus.FAILED
            tracker.end_time = time.time()
            tracker.error_messages.append(str(e))
            
            logger.error(f"Streaming request {context.request_id} failed: {e}")
            raise
        
        finally:
            # Move to history and cleanup
            self._cleanup_request(tracker)
    
    async def _route_with_retries(
        self,
        request: InferenceRequest,
        tracker: RequestTracker
    ) -> InferenceResponse:
        """Route request with retry logic."""
        last_error = None
        
        for attempt in range(tracker.context.max_retries + 1):
            try:
                # Select worker
                worker = self._select_worker_for_request(request, tracker)
                if not worker:
                    raise MLXError(
                        message="No available workers",
                        error_code=MLXErrorCode.UNAVAILABLE
                    )
                
                tracker.selected_worker = worker.worker_id
                tracker.attempted_workers.add(worker.worker_id)
                tracker.status = RequestStatus.PROCESSING
                
                # Record request start
                self.worker_pool.record_request_start(worker.worker_id)
                
                try:
                    # Get or create worker client
                    client = await self._get_worker_client(worker)
                    
                    # Process request
                    start_time = time.time()
                    response = await asyncio.wait_for(
                        client.process_inference(request),
                        timeout=tracker.context.timeout
                    )
                    
                    # Record success
                    latency_ms = (time.time() - start_time) * 1000
                    self.worker_pool.record_request_complete(
                        worker.worker_id, success=True, latency_ms=latency_ms
                    )
                    
                    # Add worker ID to response
                    response.worker_id = worker.worker_id
                    response.latency_ms = latency_ms
                    
                    return response
                    
                except Exception as e:
                    # Record failure
                    latency_ms = (time.time() - start_time) * 1000
                    self.worker_pool.record_request_complete(
                        worker.worker_id, success=False, latency_ms=latency_ms
                    )
                    raise
                
            except Exception as e:
                last_error = e
                tracker.retry_count = attempt
                tracker.error_messages.append(f"Attempt {attempt + 1}: {str(e)}")
                
                logger.warning(f"Request {tracker.request_id} attempt {attempt + 1} failed: {e}")
                
                # Don't retry on certain errors
                if isinstance(e, MLXError) and e.error_code in [
                    MLXErrorCode.INVALID_ARGUMENT,
                    MLXErrorCode.PERMISSION_DENIED,
                    MLXErrorCode.UNAUTHENTICATED
                ]:
                    break
                
                # Wait before retry (except on last attempt)
                if attempt < tracker.context.max_retries:
                    delay = min(2.0 ** attempt, 10.0)  # Exponential backoff, max 10s
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        if isinstance(last_error, MLXError):
            raise last_error
        else:
            raise MLXError(
                message=f"Request failed after {tracker.context.max_retries + 1} attempts: {str(last_error)}",
                error_code=MLXErrorCode.INTERNAL
            )
    
    async def _stream_with_retries(
        self,
        request: InferenceRequest,
        tracker: RequestTracker
    ) -> AsyncIterator[StreamChunk]:
        """Stream request with retry logic."""
        # For streaming, we typically don't retry as the stream may have already started
        # Select worker
        worker = self._select_worker_for_request(request, tracker)
        if not worker:
            raise MLXError(
                message="No available workers",
                error_code=MLXErrorCode.UNAVAILABLE
            )
        
        tracker.selected_worker = worker.worker_id
        tracker.attempted_workers.add(worker.worker_id)
        tracker.status = RequestStatus.PROCESSING
        
        # Record request start
        self.worker_pool.record_request_start(worker.worker_id)
        
        try:
            # Get or create worker client
            client = await self._get_worker_client(worker)
            
            # Process streaming request
            start_time = time.time()
            chunk_count = 0
            
            async for chunk in client.stream_inference(request):
                chunk.worker_id = worker.worker_id
                chunk_count += 1
                yield chunk
            
            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self.worker_pool.record_request_complete(
                worker.worker_id, success=True, latency_ms=latency_ms
            )
            
            logger.debug(f"Streaming request {tracker.request_id} completed with {chunk_count} chunks")
            
        except Exception as e:
            # Record failure
            latency_ms = (time.time() - start_time) * 1000
            self.worker_pool.record_request_complete(
                worker.worker_id, success=False, latency_ms=latency_ms
            )
            
            logger.error(f"Streaming request {tracker.request_id} failed: {e}")
            raise
    
    def _select_worker_for_request(
        self,
        request: InferenceRequest,
        tracker: RequestTracker
    ) -> Optional[WorkerInfo]:
        """Select appropriate worker for request."""
        # Build request context for worker selection
        request_context = {
            'client_ip': tracker.context.client_ip,
            'request_id': tracker.request_id
        }
        
        # Select worker with fallback, excluding already attempted workers
        return self.worker_pool.select_worker_with_fallback(
            model=request.model,
            request_requirements=tracker.context.requirements,
            request_context=request_context,
            exclude_workers=tracker.attempted_workers
        )
    
    async def _get_worker_client(self, worker: WorkerInfo) -> WorkerClient:
        """Get or create worker client."""
        if worker.worker_id not in self._worker_clients:
            self._worker_clients[worker.worker_id] = WorkerClient(worker)
        
        client = self._worker_clients[worker.worker_id]
        
        # Update worker info if changed
        if client.worker_info.address != worker.address:
            await client.close()
            client = WorkerClient(worker)
            self._worker_clients[worker.worker_id] = client
        
        return client
    
    def _cleanup_request(self, tracker: RequestTracker) -> None:
        """Move request from active to history."""
        # Remove from active requests
        if tracker.request_id in self._active_requests:
            del self._active_requests[tracker.request_id]
        
        # Add to history
        self._request_history.append(tracker)
        
        # Limit history size
        if len(self._request_history) > self._max_history_size:
            self._request_history.pop(0)
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return f"req-{uuid.uuid4().hex[:12]}"
    
    def get_active_requests(self) -> List[RequestTracker]:
        """Get list of active requests."""
        return list(self._active_requests.values())
    
    def get_request_history(self, limit: int = 100) -> List[RequestTracker]:
        """Get recent request history."""
        return self._request_history[-limit:] if self._request_history else []
    
    def get_request_stats(self) -> Dict[str, Any]:
        """Get request routing statistics."""
        active_count = len(self._active_requests)
        total_processed = len(self._request_history)
        
        # Calculate success rate from recent history
        recent_requests = self._request_history[-100:] if self._request_history else []
        successful_requests = sum(1 for r in recent_requests if r.status == RequestStatus.COMPLETED)
        success_rate = successful_requests / max(len(recent_requests), 1)
        
        # Calculate average latency
        completed_requests = [r for r in recent_requests if r.status == RequestStatus.COMPLETED]
        avg_latency = (
            sum(r.duration() for r in completed_requests) / max(len(completed_requests), 1)
            if completed_requests else 0.0
        )
        
        return {
            'active_requests': active_count,
            'total_processed': total_processed,
            'success_rate': success_rate,
            'average_latency_seconds': avg_latency,
            'worker_clients': len(self._worker_clients),
            'max_concurrent': self.max_concurrent_requests
        }
    
    async def cleanup_worker_clients(self) -> None:
        """Cleanup inactive worker clients."""
        # Get current worker IDs
        current_workers = {w.worker_id for w in self.worker_pool.get_all_workers()}
        
        # Close clients for workers that no longer exist
        clients_to_remove = []
        for worker_id, client in self._worker_clients.items():
            if worker_id not in current_workers:
                await client.close()
                clients_to_remove.append(worker_id)
        
        # Remove closed clients
        for worker_id in clients_to_remove:
            del self._worker_clients[worker_id]
        
        logger.debug(f"Cleaned up {len(clients_to_remove)} worker clients")
    
    async def close(self) -> None:
        """Close all worker clients."""
        logger.info("Closing request router")
        
        # Close all worker clients
        for client in self._worker_clients.values():
            await client.close()
        
        self._worker_clients.clear()
        
        logger.info("Request router closed")