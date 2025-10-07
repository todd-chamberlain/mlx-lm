"""
Distributed Streaming Inference Implementation

Implements token-by-token streaming for distributed layer-level inference,
with backpressure handling and stream error recovery.
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

import mlx.core as mx

from .distributed_generation import DistributedGenerator, GenerationConfig
from .activation_transfer import ActivationTransfer
from proto.distributed_pb2 import (
    ActivationData,
    PipelineStage,
    StreamToken,
    StreamStatus,
    StreamControl
)

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """Stream state enumeration."""
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class StreamMetrics:
    """Metrics for streaming inference."""
    tokens_generated: int = 0
    total_latency: float = 0.0
    first_token_latency: float = 0.0
    inter_token_latency: List[float] = field(default_factory=list)
    backpressure_events: int = 0
    buffer_overflows: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    
    def add_token(self, latency: float):
        """Add token generation metrics."""
        self.tokens_generated += 1
        self.total_latency += latency
        
        if self.tokens_generated == 1:
            self.first_token_latency = latency
        else:
            self.inter_token_latency.append(latency)
    
    def get_average_latency(self) -> float:
        """Get average token generation latency."""
        if self.tokens_generated == 0:
            return 0.0
        return self.total_latency / self.tokens_generated
    
    def get_throughput(self) -> float:
        """Get tokens per second."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.tokens_generated / elapsed


@dataclass
class StreamBuffer:
    """Buffer for managing streaming tokens with backpressure."""
    max_size: int = 10
    buffer: asyncio.Queue = field(default_factory=lambda: asyncio.Queue())
    high_water_mark: int = 8
    low_water_mark: int = 3
    paused: bool = False
    
    async def put(self, item: Any) -> bool:
        """
        Put item in buffer with backpressure handling.
        
        Returns:
            True if item was added, False if buffer is full
        """
        if self.buffer.qsize() >= self.max_size:
            return False
        
        await self.buffer.put(item)
        
        # Check if we should apply backpressure
        if self.buffer.qsize() >= self.high_water_mark:
            self.paused = True
        
        return True
    
    async def get(self) -> Any:
        """Get item from buffer."""
        item = await self.buffer.get()
        
        # Check if we should resume
        if self.paused and self.buffer.qsize() <= self.low_water_mark:
            self.paused = False
        
        return item
    
    def should_pause(self) -> bool:
        """Check if producer should pause."""
        return self.paused
    
    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer.qsize()
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.buffer.empty()


class DistributedStreamer:
    """
    Manages distributed streaming inference with token-by-token generation.
    
    Handles:
    - Token-by-token streaming through pipeline
    - Backpressure management
    - Stream error recovery
    - Stream lifecycle management
    """
    
    def __init__(
        self,
        generator: DistributedGenerator,
        activation_transfer: ActivationTransfer,
        buffer_size: int = 10,
        backpressure_enabled: bool = True
    ):
        """
        Initialize distributed streamer.
        
        Args:
            generator: Distributed generator instance
            activation_transfer: Activation transfer instance
            buffer_size: Size of streaming buffer
            backpressure_enabled: Enable backpressure handling
        """
        self.generator = generator
        self.activation_transfer = activation_transfer
        self.buffer_size = buffer_size
        self.backpressure_enabled = backpressure_enabled
        
        # Stream state
        self.streams: Dict[str, StreamState] = {}
        self.buffers: Dict[str, StreamBuffer] = {}
        self.metrics: Dict[str, StreamMetrics] = {}
        
        # Control
        self._shutdown = False
        
        logger.info(
            f"Initialized DistributedStreamer with buffer_size={buffer_size}, "
            f"backpressure_enabled={backpressure_enabled}"
        )
    
    async def stream_generate(
        self,
        request_id: str,
        prompt_tokens: List[int],
        config: GenerationConfig,
        pipeline_stages: List[PipelineStage]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream token generation through distributed pipeline.
        
        Args:
            request_id: Unique request identifier
            prompt_tokens: Input prompt tokens
            config: Generation configuration
            pipeline_stages: Pipeline stage definitions
            
        Yields:
            Dictionary with token and metadata for each generated token
        """
        # Initialize stream
        self.streams[request_id] = StreamState.ACTIVE
        self.buffers[request_id] = StreamBuffer(max_size=self.buffer_size)
        self.metrics[request_id] = StreamMetrics()
        
        buffer = self.buffers[request_id]
        metrics = self.metrics[request_id]
        
        try:
            # Start generation task
            generation_task = asyncio.create_task(
                self._generate_tokens(
                    request_id,
                    prompt_tokens,
                    config,
                    pipeline_stages,
                    buffer,
                    metrics
                )
            )
            
            # Stream tokens as they become available
            while True:
                # Check stream state
                if self.streams[request_id] == StreamState.CANCELLED:
                    generation_task.cancel()
                    break
                
                if self.streams[request_id] == StreamState.ERROR:
                    break
                
                # Get next token from buffer
                try:
                    token_data = await asyncio.wait_for(
                        buffer.get(),
                        timeout=config.timeout if config.timeout else 30.0
                    )
                    
                    # Check for completion marker
                    if token_data is None:
                        self.streams[request_id] = StreamState.COMPLETED
                        break
                    
                    yield token_data
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Stream timeout for request {request_id}")
                    self.streams[request_id] = StreamState.ERROR
                    metrics.errors += 1
                    break
            
            # Wait for generation to complete
            try:
                await generation_task
            except asyncio.CancelledError:
                logger.info(f"Generation cancelled for request {request_id}")
            
        except Exception as e:
            logger.error(f"Stream error for request {request_id}: {e}")
            self.streams[request_id] = StreamState.ERROR
            metrics.errors += 1
            raise
        
        finally:
            # Cleanup
            self._cleanup_stream(request_id)
    
    async def _generate_tokens(
        self,
        request_id: str,
        prompt_tokens: List[int],
        config: GenerationConfig,
        pipeline_stages: List[PipelineStage],
        buffer: StreamBuffer,
        metrics: StreamMetrics
    ):
        """
        Generate tokens and put them in the stream buffer.
        
        Args:
            request_id: Request identifier
            prompt_tokens: Input prompt tokens
            config: Generation configuration
            pipeline_stages: Pipeline stages
            buffer: Stream buffer
            metrics: Stream metrics
        """
        try:
            # Process prompt
            prompt_start = time.time()
            current_tokens = prompt_tokens.copy()
            
            # Generate tokens one at a time
            for step in range(config.max_tokens):
                # Check if stream is cancelled
                if self.streams[request_id] == StreamState.CANCELLED:
                    break
                
                # Handle backpressure
                if self.backpressure_enabled:
                    while buffer.should_pause():
                        await asyncio.sleep(0.01)
                        metrics.backpressure_events += 1
                        
                        # Check for cancellation during pause
                        if self.streams[request_id] == StreamState.CANCELLED:
                            return
                
                # Generate next token
                token_start = time.time()
                
                try:
                    # Run forward pass through pipeline
                    next_token = await self._generate_next_token(
                        current_tokens,
                        config,
                        pipeline_stages
                    )
                    
                    token_latency = time.time() - token_start
                    metrics.add_token(token_latency)
                    
                    # Create token data
                    token_data = {
                        "token": int(next_token),
                        "step": step,
                        "latency": token_latency,
                        "timestamp": time.time()
                    }
                    
                    # Add to buffer with overflow handling
                    added = await buffer.put(token_data)
                    if not added:
                        logger.warning(f"Buffer overflow for request {request_id}")
                        metrics.buffer_overflows += 1
                        # Wait and retry
                        await asyncio.sleep(0.1)
                        await buffer.put(token_data)
                    
                    # Check stopping criteria
                    if self._should_stop(next_token, current_tokens, config):
                        break
                    
                    # Update tokens
                    current_tokens.append(int(next_token))
                    
                except Exception as e:
                    logger.error(f"Token generation error: {e}")
                    self.streams[request_id] = StreamState.ERROR
                    metrics.errors += 1
                    break
            
            # Send completion marker
            await buffer.put(None)
            
        except Exception as e:
            logger.error(f"Generation task error: {e}")
            self.streams[request_id] = StreamState.ERROR
            metrics.errors += 1
            raise
    
    async def _generate_next_token(
        self,
        tokens: List[int],
        config: GenerationConfig,
        pipeline_stages: List[PipelineStage]
    ) -> int:
        """
        Generate next token through distributed pipeline.
        
        Args:
            tokens: Current token sequence
            config: Generation configuration
            pipeline_stages: Pipeline stages
            
        Returns:
            Next token ID
        """
        # Use the distributed generator for single token generation
        result = await self.generator.generate(
            tokens,
            GenerationConfig(
                max_tokens=1,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                stop_tokens=config.stop_tokens,
                timeout=config.timeout
            ),
            pipeline_stages
        )
        
        if not result.tokens:
            raise RuntimeError("No token generated")
        
        return result.tokens[0]
    
    def _should_stop(
        self,
        token: int,
        tokens: List[int],
        config: GenerationConfig
    ) -> bool:
        """
        Check if generation should stop.
        
        Args:
            token: Current token
            tokens: Token sequence
            config: Generation configuration
            
        Returns:
            True if should stop
        """
        # Check stop tokens
        if config.stop_tokens and token in config.stop_tokens:
            return True
        
        # Check max length
        if len(tokens) >= config.max_tokens:
            return True
        
        return False
    
    def cancel_stream(self, request_id: str):
        """
        Cancel an active stream.
        
        Args:
            request_id: Request identifier
        """
        if request_id in self.streams:
            logger.info(f"Cancelling stream {request_id}")
            self.streams[request_id] = StreamState.CANCELLED
    
    def get_stream_state(self, request_id: str) -> Optional[StreamState]:
        """
        Get current stream state.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Stream state or None if not found
        """
        return self.streams.get(request_id)
    
    def get_stream_metrics(self, request_id: str) -> Optional[StreamMetrics]:
        """
        Get stream metrics.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Stream metrics or None if not found
        """
        return self.metrics.get(request_id)
    
    def _cleanup_stream(self, request_id: str):
        """
        Clean up stream resources.
        
        Args:
            request_id: Request identifier
        """
        # Keep metrics for a while for debugging
        # Remove buffers and state
        if request_id in self.buffers:
            del self.buffers[request_id]
        
        logger.debug(f"Cleaned up stream {request_id}")
    
    async def shutdown(self):
        """Shutdown the streamer and cancel all active streams."""
        self._shutdown = True
        
        # Cancel all active streams
        for request_id in list(self.streams.keys()):
            if self.streams[request_id] == StreamState.ACTIVE:
                self.cancel_stream(request_id)
        
        logger.info("DistributedStreamer shutdown complete")
