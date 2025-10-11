"""
FastAPI server with OpenAI-compatible endpoints for MLX distributed inference.
"""

import os
import time
import uuid
import asyncio
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    HealthResponse,
    ModelsResponse,
    ErrorResponse,
    ModelDownloadRequest,
    ModelDownloadResponse,
    ShardStatusResponse,
    WorkerShardStatus
)
from .coordinator_client import CoordinatorClient
from .auth import verify_api_key
from .middleware import setup_middleware
from .metrics import setup_metrics
from .tracing import setup_tracing, trace_inference_request, trace_worker_selection, trace_streaming_response, custom_span
from .streaming import streaming_manager
from .middleware import limiter
from .audit import audit


# Setup logging
logger = logging.getLogger(__name__)

# Global coordinator client
coordinator_client: Optional[CoordinatorClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global coordinator_client
    
    # Startup
    coordinator_host = os.getenv("COORDINATOR_HOST", "localhost:50051")
    coordinator_client = CoordinatorClient(coordinator_host)
    await coordinator_client.connect()
    
    yield
    
    # Shutdown
    if coordinator_client:
        await coordinator_client.close()


# Create FastAPI app
app = FastAPI(
    title="MLX Distributed Inference API",
    description="OpenAI-compatible API for distributed MLX inference",
    version="1.0.0",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Setup metrics
setup_metrics(app)

# Setup tracing
setup_tracing(app)

# Security
security = HTTPBearer(auto_error=False)


def get_coordinator_client() -> CoordinatorClient:
    """Get the global coordinator client."""
    logger.info("get_coordinator_client called")
    if coordinator_client is None:
        logger.error("Coordinator client is None!")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Coordinator client not initialized"
        )
    logger.info(f"Returning coordinator client: {coordinator_client}")
    return coordinator_client


@app.post("/v1/test_simple")
async def test_simple_post(request: Request):
    """Minimal test endpoint."""
    from fastapi.responses import JSONResponse
    logger.info("test_simple_post called")
    return JSONResponse(content={"status": "ok", "test": "simple"})

@app.post("/v1/test_with_model")
async def test_with_model(chat_request: ChatCompletionRequest):
    """Test with Pydantic model."""
    from fastapi.responses import JSONResponse
    logger.info(f"test_with_model called with: {chat_request}")
    return JSONResponse(content={"status": "ok", "model": chat_request.model})

@app.post("/v1/test_with_coordinator")
async def test_with_coordinator(
    chat_request: ChatCompletionRequest,
    coordinator: CoordinatorClient = Depends(get_coordinator_client)
):
    """Test with coordinator dependency."""
    from fastapi.responses import JSONResponse
    logger.info(f"test_with_coordinator called")
    return JSONResponse(content={"status": "ok", "has_coordinator": coordinator is not None})

@app.post("/v1/test_inference")
async def test_inference(
    chat_request: ChatCompletionRequest,
    coordinator: CoordinatorClient = Depends(get_coordinator_client)
):
    """Test actual inference call."""
    from fastapi.responses import JSONResponse
    logger.info(f"test_inference: calling coordinator.route_inference")
    response = await coordinator.route_inference(chat_request)
    logger.info(f"test_inference: got response type={type(response)}")
    return JSONResponse(content=response.model_dump())

@app.post("/v1/chat/completions")
# @limiter.limit("100/minute")  # 100 requests per minute per client
async def create_chat_completion(
    chat_request: ChatCompletionRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    coordinator: CoordinatorClient = Depends(get_coordinator_client)
):
    """
    Create a chat completion, optionally streaming.
    Compatible with OpenAI's chat completions API.
    """
    start_time = time.time()
    
    # Verify authentication if enabled
    if os.getenv("ENABLE_AUTH", "false").lower() == "true":
        verify_api_key(credentials, request)
    
    # Handle streaming requests
    if chat_request.stream:
        request_id = chat_request.id or str(uuid.uuid4())
        return await streaming_manager.create_streaming_response(
            request,
            stream_chat_completion(chat_request, coordinator),
            request_id
        )
    
    # Handle non-streaming requests
    try:
        logger.info(f"Routing inference request for model: {chat_request.model}")
        response = await coordinator.route_inference(chat_request)
        logger.info(f"Got response type: {type(response)}, value: {response}")
        
        # Log successful request
        duration_ms = (time.time() - start_time) * 1000
        audit.log_request(
            request,
            credentials,
            request_data=chat_request.model_dump(),
            response_status=200,
            response_data=response.model_dump() if hasattr(response, 'model_dump') else response,
            duration_ms=duration_ms
        )
        
        # Return as JSON response
        from fastapi.responses import JSONResponse
        if isinstance(response, dict):
            logger.warning("Response is a dict, wrapping in JSONResponse")
            return JSONResponse(content=response)
        else:
            logger.info("Response is a Pydantic model, converting to dict")
            return JSONResponse(content=response.model_dump())
    except Exception as e:
        # Log failed request
        duration_ms = (time.time() - start_time) * 1000
        audit.log_request(
            request,
            credentials,
            request_data=chat_request.model_dump(),
            response_status=500,
            duration_ms=duration_ms,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


async def stream_chat_completion(
    request: ChatCompletionRequest,
    coordinator: CoordinatorClient
) -> AsyncGenerator[str, None]:
    """
    Stream chat completion responses in OpenAI format with proper error handling and cleanup.
    """
    request_id = request.id or str(uuid.uuid4())
    
    try:
        # Send initial chunk to establish connection
        initial_chunk = ChatCompletionStreamResponse(
            id=request_id,
            model=request.model,
            choices=[{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"
        
        # Stream inference responses
        chunk_count = 0
        async for chunk in coordinator.stream_inference(request):
            chunk_count += 1
            yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Periodic check for client disconnection (every 10 chunks)
            if chunk_count % 10 == 0:
                await asyncio.sleep(0)  # Allow other tasks to run
        
        # Send final chunk
        final_chunk = ChatCompletionStreamResponse(
            id=request_id,
            model=request.model,
            choices=[{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
    except asyncio.CancelledError:
        # Client disconnected
        logger.info(f"Streaming request {request_id} cancelled by client")
        # Send cancellation chunk
        cancel_chunk = ChatCompletionStreamResponse(
            id=request_id,
            model=request.model,
            choices=[{
                "index": 0,
                "delta": {},
                "finish_reason": "cancelled"
            }]
        )
        yield f"data: {cancel_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        raise
        
    except Exception as e:
        logger.error(f"Streaming error for request {request_id}: {e}")
        # Send error chunk
        error_chunk = ChatCompletionStreamResponse(
            id=request_id,
            model=request.model,
            choices=[{
                "index": 0,
                "delta": {},
                "finish_reason": "error"
            }],
            error=str(e)
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/health", response_model=HealthResponse)
async def health_check(
    include_sharded_models: bool = False,
    coordinator: CoordinatorClient = Depends(get_coordinator_client)
):
    """
    Get system health status.
    
    Args:
        include_sharded_models: Include detailed health info for sharded models
    """
    try:
        health_info = await coordinator.get_health(include_sharded_models=include_sharded_models)
        
        # Build response
        response = HealthResponse(
            status="healthy" if health_info.healthy_workers > 0 else "unhealthy",
            total_workers=health_info.total_workers,
            healthy_workers=health_info.healthy_workers,
            active_requests=health_info.active_requests,
            timestamp=int(time.time())
        )
        
        # Add sharded model information if available
        if hasattr(health_info, 'sharded_models') and health_info.sharded_models:
            from .models import ShardedModelHealth
            
            response.sharded_model_count = len(health_info.sharded_models)
            response.sharded_models = [
                ShardedModelHealth(
                    model_name=sm.model_name,
                    total_shards=sm.total_shards,
                    healthy_shards=sm.healthy_shards,
                    status=sm.status
                )
                for sm in health_info.sharded_models
            ]
            
            # Adjust overall status if any sharded models are degraded
            if any(sm.status == "degraded" for sm in health_info.sharded_models):
                response.status = "degraded"
            elif any(sm.status == "offline" for sm in health_info.sharded_models):
                if response.status == "healthy":
                    response.status = "degraded"
        
        return response
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            total_workers=0,
            healthy_workers=0,
            active_requests=0,
            timestamp=int(time.time()),
            error=str(e)
        )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    # This will be implemented by the metrics middleware
    pass


@app.get("/models", response_model=ModelsResponse)
@limiter.limit("60/minute")  # 60 requests per minute per client
async def list_models(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    coordinator: CoordinatorClient = Depends(get_coordinator_client)
):
    """List available models."""
    # Verify authentication if enabled
    if os.getenv("ENABLE_AUTH", "false").lower() == "true":
        verify_api_key(credentials, request)
    
    try:
        models = await coordinator.get_available_models()
        return ModelsResponse(
            object="list",
            data=[
                {
                    "id": model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mlx-distributed-inference"
                }
                for model in models
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@app.post("/admin/download_model")
@limiter.limit("10/minute")
async def download_model(
    download_request: ModelDownloadRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    coordinator: CoordinatorClient = Depends(get_coordinator_client)
):
    """
    Download a model in the background without interrupting service.
    
    Supports automatic sharding across workers when enable_sharding is True.
    Requires at least 2 workers for sharding to be enabled.
    """
    # Verify authentication
    if os.getenv("ENABLE_AUTH", "false").lower() == "true":
        verify_api_key(credentials, request)
    
    try:
        # Validate minimum worker count if sharding is enabled
        if download_request.enable_sharding:
            health_info = await coordinator.get_health()
            if health_info.healthy_workers < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Sharding requires at least 2 workers, but only {health_info.healthy_workers} available"
                )
        
        result = await coordinator.download_model(
            download_request.model_name,
            enable_sharding=download_request.enable_sharding,
            force_redownload=download_request.force_redownload,
            num_layers=download_request.num_layers,
            model_size_gb=download_request.model_size_gb
        )
        
        # Return result directly
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download failed: {str(e)}"
        )


@app.get("/admin/download_status")
async def download_status(
    model_name: Optional[str] = None,
    coordinator: CoordinatorClient = Depends(get_coordinator_client)
):
    """Check download status."""
    try:
        result = await coordinator.get_download_status(model_name)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )


@app.get("/admin/models/{model_name}/shards", response_model=ShardStatusResponse)
async def get_model_shards(
    model_name: str,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    coordinator: CoordinatorClient = Depends(get_coordinator_client)
):
    """
    Get shard status for a model.
    
    Returns shard assignments per worker, download/load status per shard,
    and inference metrics for sharded models.
    """
    # Verify authentication if enabled
    if os.getenv("ENABLE_AUTH", "false").lower() == "true":
        verify_api_key(credentials, request)
    
    try:
        result = await coordinator.get_shard_status(model_name)
        
        # Convert to response model
        response = ShardStatusResponse(
            model_name=result['model_name'],
            is_sharded=result['is_sharded'],
            total_layers=result['total_layers'],
            shard_assignments=[
                WorkerShardStatus(**shard)
                for shard in result['shard_assignments']
            ],
            overall_status=result['overall_status'],
            inference_count=result.get('inference_count'),
            avg_latency_ms=result.get('avg_latency_ms')
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get shard status: {str(e)}"
        )


@app.post("/admin/load_model")
@limiter.limit("5/minute")
async def load_model(
    model_name: str,
    request: Request,
    drain_timeout: int = 30,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    coordinator: CoordinatorClient = Depends(get_coordinator_client)
):
    """Hot-swap to a different model with graceful draining."""
    # Verify authentication
    if os.getenv("ENABLE_AUTH", "false").lower() == "true":
        verify_api_key(credentials, request)
    
    try:
        result = await coordinator.load_model(model_name, drain_timeout)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model load failed: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with OpenAI-compatible error format."""
    return ErrorResponse(
        error={
            "message": exc.detail,
            "type": "invalid_request_error",
            "code": exc.status_code
        }
    ).model_dump()


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )