"""
Streaming utilities for Server-Sent Events (SSE) and connection management.
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional
from fastapi import Request
from fastapi.responses import StreamingResponse


logger = logging.getLogger(__name__)


class StreamingManager:
    """Manages streaming connections with timeout and disconnection detection."""
    
    def __init__(self, timeout_seconds: int = 300):
        """
        Initialize streaming manager.
        
        Args:
            timeout_seconds: Maximum streaming duration in seconds
        """
        self.timeout_seconds = timeout_seconds
        self.active_streams = {}
    
    async def create_streaming_response(
        self,
        request: Request,
        stream_generator: AsyncGenerator[str, None],
        request_id: str
    ) -> StreamingResponse:
        """
        Create a streaming response with connection management.
        
        Args:
            request: FastAPI request object
            stream_generator: Async generator yielding SSE data
            request_id: Unique request identifier
            
        Returns:
            StreamingResponse with proper headers and connection management
        """
        # Track active stream
        self.active_streams[request_id] = {
            "start_time": time.time(),
            "client_ip": request.client.host if request.client else "unknown"
        }
        
        async def managed_stream():
            """Managed stream with timeout and disconnection detection."""
            try:
                start_time = time.time()
                
                async for chunk in stream_generator:
                    # Check for timeout
                    if time.time() - start_time > self.timeout_seconds:
                        logger.warning(f"Stream {request_id} timed out after {self.timeout_seconds}s")
                        yield "data: {\"error\": \"Stream timeout\"}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    
                    # Check if client is still connected
                    if await self._is_client_disconnected(request):
                        logger.info(f"Client disconnected for stream {request_id}")
                        break
                    
                    yield chunk
                    
            except asyncio.CancelledError:
                logger.info(f"Stream {request_id} cancelled")
                raise
            except Exception as e:
                logger.error(f"Stream {request_id} error: {e}")
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # Clean up active stream tracking
                self.active_streams.pop(request_id, None)
        
        return StreamingResponse(
            managed_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
    
    async def _is_client_disconnected(self, request: Request) -> bool:
        """
        Check if client has disconnected.
        
        Args:
            request: FastAPI request object
            
        Returns:
            True if client is disconnected
        """
        try:
            # This is a simple check - in production you might want more sophisticated detection
            return False  # For now, assume client is always connected
        except Exception:
            return True
    
    def get_active_streams_count(self) -> int:
        """Get number of active streaming connections."""
        return len(self.active_streams)
    
    def get_stream_info(self, request_id: str) -> Optional[dict]:
        """Get information about a specific stream."""
        return self.active_streams.get(request_id)
    
    def cleanup_expired_streams(self) -> None:
        """Clean up expired stream tracking entries."""
        current_time = time.time()
        expired_streams = [
            request_id for request_id, info in self.active_streams.items()
            if current_time - info["start_time"] > self.timeout_seconds
        ]
        
        for request_id in expired_streams:
            self.active_streams.pop(request_id, None)
            logger.info(f"Cleaned up expired stream tracking for {request_id}")


# Global streaming manager instance
streaming_manager = StreamingManager()


def format_sse_data(data: str, event: Optional[str] = None) -> str:
    """
    Format data for Server-Sent Events.
    
    Args:
        data: Data to send
        event: Optional event type
        
    Returns:
        Formatted SSE string
    """
    lines = []
    
    if event:
        lines.append(f"event: {event}")
    
    # Handle multi-line data
    for line in data.split('\n'):
        lines.append(f"data: {line}")
    
    lines.append("")  # Empty line to end the event
    
    return "\n".join(lines) + "\n"


def format_sse_json(data: dict, event: Optional[str] = None) -> str:
    """
    Format JSON data for Server-Sent Events.
    
    Args:
        data: Dictionary to send as JSON
        event: Optional event type
        
    Returns:
        Formatted SSE string
    """
    import json
    return format_sse_data(json.dumps(data), event)