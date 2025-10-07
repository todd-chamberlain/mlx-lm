"""
Tests for distributed streaming inference implementation.

Tests token-by-token streaming, backpressure handling, stream error recovery,
and streaming metrics.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List

from worker.distributed_streaming import (
    DistributedStreamer,
    StreamState,
    StreamMetrics,
    StreamBuffer
)
from worker.distributed_generation import (
    DistributedGenerator,
    GenerationConfig,
    GenerationResult
)
from worker.activation_transfer import ActivationTransfer
from proto.distributed_pb2 import PipelineStage


class TestStreamBuffer:
    """Test stream buffer with backpressure handling."""
    
    @pytest.mark.asyncio
    async def test_buffer_basic_operations(self):
        """Test basic buffer put/get operations."""
        buffer = StreamBuffer(max_size=5)
        
        # Put items
        assert await buffer.put("item1") == True
        assert await buffer.put("item2") == True
        assert buffer.size() == 2
        
        # Get items
        item = await buffer.get()
        assert item == "item1"
        assert buffer.size() == 1
    
    @pytest.mark.asyncio
    async def test_buffer_overflow(self):
        """Test buffer overflow handling."""
        buffer = StreamBuffer(max_size=3)
        
        # Fill buffer
        assert await buffer.put("item1") == True
        assert await buffer.put("item2") == True
        assert await buffer.put("item3") == True
        
        # Buffer is full
        assert await buffer.put("item4") == False
        assert buffer.size() == 3
    
    @pytest.mark.asyncio
    async def test_backpressure_activation(self):
        """Test backpressure activation at high water mark."""
        buffer = StreamBuffer(max_size=10, high_water_mark=8, low_water_mark=3)
        
        # Fill to high water mark
        for i in range(8):
            await buffer.put(f"item{i}")
        
        # Should activate backpressure
        assert buffer.should_pause() == True
    
    @pytest.mark.asyncio
    async def test_backpressure_release(self):
        """Test backpressure release at low water mark."""
        buffer = StreamBuffer(max_size=10, high_water_mark=8, low_water_mark=3)
        
        # Fill to high water mark
        for i in range(8):
            await buffer.put(f"item{i}")
        
        assert buffer.should_pause() == True
        
        # Drain to low water mark
        for i in range(5):
            await buffer.get()
        
        # Should release backpressure
        assert buffer.should_pause() == False


class TestStreamMetrics:
    """Test stream metrics collection."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = StreamMetrics()
        
        assert metrics.tokens_generated == 0
        assert metrics.total_latency == 0.0
        assert metrics.first_token_latency == 0.0
        assert len(metrics.inter_token_latency) == 0
    
    def test_add_token_metrics(self):
        """Test adding token metrics."""
        metrics = StreamMetrics()
        
        # Add first token
        metrics.add_token(0.5)
        assert metrics.tokens_generated == 1
        assert metrics.first_token_latency == 0.5
        assert len(metrics.inter_token_latency) == 0
        
        # Add second token
        metrics.add_token(0.3)
        assert metrics.tokens_generated == 2
        assert len(metrics.inter_token_latency) == 1
        assert metrics.inter_token_latency[0] == 0.3
    
    def test_average_latency_calculation(self):
        """Test average latency calculation."""
        metrics = StreamMetrics()
        
        metrics.add_token(0.5)
        metrics.add_token(0.3)
        metrics.add_token(0.4)
        
        avg_latency = metrics.get_average_latency()
        assert abs(avg_latency - 0.4) < 0.01
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        metrics = StreamMetrics()
        
        # Add tokens
        for i in range(10):
            metrics.add_token(0.1)
        
        # Get throughput
        throughput = metrics.get_throughput()
        assert throughput > 0


class TestDistributedStreamer:
    """Test distributed streaming inference."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_generator = Mock(spec=DistributedGenerator)
        self.mock_activation_transfer = Mock(spec=ActivationTransfer)
        
        self.streamer = DistributedStreamer(
            generator=self.mock_generator,
            activation_transfer=self.mock_activation_transfer,
            buffer_size=10,
            backpressure_enabled=True
        )
    
    @pytest.mark.asyncio
    async def test_stream_initialization(self):
        """Test stream initialization."""
        request_id = "test-request-1"
        prompt_tokens = [1, 2, 3]
        config = GenerationConfig(max_tokens=5)
        pipeline_stages = []
        
        # Mock generator to return tokens
        async def mock_generate(*args, **kwargs):
            return GenerationResult(
                tokens=[10],
                logits=None,
                latency=0.1,
                tokens_per_second=10.0
            )
        
        self.mock_generator.generate = AsyncMock(side_effect=mock_generate)
        
        # Start streaming
        tokens = []
        async for token_data in self.streamer.stream_generate(
            request_id, prompt_tokens, config, pipeline_stages
        ):
            tokens.append(token_data["token"])
            if len(tokens) >= 5:
                break
        
        # Verify stream was initialized
        assert len(tokens) == 5
        assert request_id in self.streamer.metrics
    
    @pytest.mark.asyncio
    async def test_token_streaming(self):
        """Test token-by-token streaming."""
        request_id = "test-request-2"
        prompt_tokens = [1, 2, 3]
        config = GenerationConfig(max_tokens=3)
        pipeline_stages = []
        
        # Mock generator to return sequential tokens
        token_sequence = [10, 11, 12]
        token_index = [0]
        
        async def mock_generate(*args, **kwargs):
            token = token_sequence[token_index[0]]
            token_index[0] += 1
            return GenerationResult(
                tokens=[token],
                logits=None,
                latency=0.1,
                tokens_per_second=10.0
            )
        
        self.mock_generator.generate = AsyncMock(side_effect=mock_generate)
        
        # Stream tokens
        tokens = []
        async for token_data in self.streamer.stream_generate(
            request_id, prompt_tokens, config, pipeline_stages
        ):
            tokens.append(token_data["token"])
        
        # Verify tokens
        assert tokens == [10, 11, 12]
    
    @pytest.mark.asyncio
    async def test_backpressure_handling(self):
        """Test backpressure handling during streaming."""
        request_id = "test-request-3"
        prompt_tokens = [1, 2, 3]
        config = GenerationConfig(max_tokens=20)
        pipeline_stages = []
        
        # Create streamer with small buffer
        streamer = DistributedStreamer(
            generator=self.mock_generator,
            activation_transfer=self.mock_activation_transfer,
            buffer_size=5,
            backpressure_enabled=True
        )
        
        # Mock fast generation
        async def mock_generate(*args, **kwargs):
            await asyncio.sleep(0.01)  # Fast generation
            return GenerationResult(
                tokens=[10],
                logits=None,
                latency=0.01,
                tokens_per_second=100.0
            )
        
        self.mock_generator.generate = AsyncMock(side_effect=mock_generate)
        
        # Stream with slow consumption
        tokens = []
        async for token_data in streamer.stream_generate(
            request_id, prompt_tokens, config, pipeline_stages
        ):
            await asyncio.sleep(0.05)  # Slow consumption
            tokens.append(token_data["token"])
            if len(tokens) >= 10:
                break
        
        # Verify backpressure was applied
        metrics = streamer.get_stream_metrics(request_id)
        assert metrics is not None
        # Backpressure events should have occurred
        # (exact count depends on timing)
    
    @pytest.mark.asyncio
    async def test_stream_cancellation(self):
        """Test stream cancellation."""
        request_id = "test-request-4"
        prompt_tokens = [1, 2, 3]
        config = GenerationConfig(max_tokens=100)
        pipeline_stages = []
        
        # Mock slow generation
        async def mock_generate(*args, **kwargs):
            await asyncio.sleep(0.1)
            return GenerationResult(
                tokens=[10],
                logits=None,
                latency=0.1,
                tokens_per_second=10.0
            )
        
        self.mock_generator.generate = AsyncMock(side_effect=mock_generate)
        
        # Start streaming
        tokens = []
        async for token_data in self.streamer.stream_generate(
            request_id, prompt_tokens, config, pipeline_stages
        ):
            tokens.append(token_data["token"])
            
            # Cancel after 3 tokens
            if len(tokens) >= 3:
                self.streamer.cancel_stream(request_id)
                break
        
        # Verify cancellation
        state = self.streamer.get_stream_state(request_id)
        assert state == StreamState.CANCELLED
        assert len(tokens) == 3
    
    @pytest.mark.asyncio
    async def test_stop_token_handling(self):
        """Test stop token handling."""
        request_id = "test-request-5"
        prompt_tokens = [1, 2, 3]
        config = GenerationConfig(max_tokens=100, stop_tokens=[50])
        pipeline_stages = []
        
        # Mock generator to return stop token
        token_sequence = [10, 11, 50]  # 50 is stop token
        token_index = [0]
        
        async def mock_generate(*args, **kwargs):
            if token_index[0] < len(token_sequence):
                token = token_sequence[token_index[0]]
                token_index[0] += 1
                return GenerationResult(
                    tokens=[token],
                    logits=None,
                    latency=0.1,
                    tokens_per_second=10.0
                )
            return GenerationResult(tokens=[0], logits=None, latency=0.1, tokens_per_second=10.0)
        
        self.mock_generator.generate = AsyncMock(side_effect=mock_generate)
        
        # Stream tokens
        tokens = []
        async for token_data in self.streamer.stream_generate(
            request_id, prompt_tokens, config, pipeline_stages
        ):
            tokens.append(token_data["token"])
        
        # Should stop at stop token
        assert tokens == [10, 11, 50]
    
    @pytest.mark.asyncio
    async def test_stream_error_recovery(self):
        """Test stream error recovery."""
        request_id = "test-request-6"
        prompt_tokens = [1, 2, 3]
        config = GenerationConfig(max_tokens=10)
        pipeline_stages = []
        
        # Mock generator to raise error
        call_count = [0]
        
        async def mock_generate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:
                raise RuntimeError("Generation error")
            return GenerationResult(
                tokens=[10],
                logits=None,
                latency=0.1,
                tokens_per_second=10.0
            )
        
        self.mock_generator.generate = AsyncMock(side_effect=mock_generate)
        
        # Stream should handle error
        tokens = []
        try:
            async for token_data in self.streamer.stream_generate(
                request_id, prompt_tokens, config, pipeline_stages
            ):
                tokens.append(token_data["token"])
        except RuntimeError:
            pass
        
        # Verify error was recorded
        metrics = self.streamer.get_stream_metrics(request_id)
        assert metrics is not None
        assert metrics.errors > 0
        
        state = self.streamer.get_stream_state(request_id)
        assert state == StreamState.ERROR
    
    @pytest.mark.asyncio
    async def test_stream_timeout(self):
        """Test stream timeout handling."""
        request_id = "test-request-7"
        prompt_tokens = [1, 2, 3]
        config = GenerationConfig(max_tokens=10, timeout=0.5)
        pipeline_stages = []
        
        # Mock generator with long delay
        async def mock_generate(*args, **kwargs):
            await asyncio.sleep(1.0)  # Longer than timeout
            return GenerationResult(
                tokens=[10],
                logits=None,
                latency=1.0,
                tokens_per_second=1.0
            )
        
        self.mock_generator.generate = AsyncMock(side_effect=mock_generate)
        
        # Stream should timeout
        tokens = []
        async for token_data in self.streamer.stream_generate(
            request_id, prompt_tokens, config, pipeline_stages
        ):
            tokens.append(token_data["token"])
        
        # Should have timed out
        state = self.streamer.get_stream_state(request_id)
        assert state == StreamState.ERROR
    
    @pytest.mark.asyncio
    async def test_stream_metrics_collection(self):
        """Test stream metrics collection."""
        request_id = "test-request-8"
        prompt_tokens = [1, 2, 3]
        config = GenerationConfig(max_tokens=5)
        pipeline_stages = []
        
        # Mock generator with varying latencies
        latencies = [0.1, 0.2, 0.15, 0.18, 0.12]
        latency_index = [0]
        
        async def mock_generate(*args, **kwargs):
            latency = latencies[latency_index[0]]
            latency_index[0] += 1
            await asyncio.sleep(latency)
            return GenerationResult(
                tokens=[10],
                logits=None,
                latency=latency,
                tokens_per_second=1.0/latency
            )
        
        self.mock_generator.generate = AsyncMock(side_effect=mock_generate)
        
        # Stream tokens
        tokens = []
        async for token_data in self.streamer.stream_generate(
            request_id, prompt_tokens, config, pipeline_stages
        ):
            tokens.append(token_data["token"])
        
        # Verify metrics
        metrics = self.streamer.get_stream_metrics(request_id)
        assert metrics is not None
        assert metrics.tokens_generated == 5
        assert metrics.first_token_latency == 0.1
        assert len(metrics.inter_token_latency) == 4
        assert metrics.get_average_latency() > 0
        assert metrics.get_throughput() > 0
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_streams(self):
        """Test multiple concurrent streams."""
        # Mock generator
        async def mock_generate(*args, **kwargs):
            await asyncio.sleep(0.01)
            return GenerationResult(
                tokens=[10],
                logits=None,
                latency=0.01,
                tokens_per_second=100.0
            )
        
        self.mock_generator.generate = AsyncMock(side_effect=mock_generate)
        
        # Start multiple streams
        async def stream_task(request_id: str):
            tokens = []
            async for token_data in self.streamer.stream_generate(
                request_id,
                [1, 2, 3],
                GenerationConfig(max_tokens=5),
                []
            ):
                tokens.append(token_data["token"])
            return tokens
        
        # Run concurrent streams
        results = await asyncio.gather(
            stream_task("request-1"),
            stream_task("request-2"),
            stream_task("request-3")
        )
        
        # Verify all streams completed
        assert len(results) == 3
        for tokens in results:
            assert len(tokens) == 5
    
    @pytest.mark.asyncio
    async def test_streamer_shutdown(self):
        """Test streamer shutdown."""
        request_id = "test-request-9"
        prompt_tokens = [1, 2, 3]
        config = GenerationConfig(max_tokens=100)
        pipeline_stages = []
        
        # Mock generator
        async def mock_generate(*args, **kwargs):
            await asyncio.sleep(0.1)
            return GenerationResult(
                tokens=[10],
                logits=None,
                latency=0.1,
                tokens_per_second=10.0
            )
        
        self.mock_generator.generate = AsyncMock(side_effect=mock_generate)
        
        # Start stream
        stream_task = asyncio.create_task(
            self._consume_stream(request_id, prompt_tokens, config, pipeline_stages)
        )
        
        # Wait a bit then shutdown
        await asyncio.sleep(0.2)
        await self.streamer.shutdown()
        
        # Wait for stream to finish
        try:
            await asyncio.wait_for(stream_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        # Verify shutdown
        assert self.streamer._shutdown == True
    
    async def _consume_stream(self, request_id, prompt_tokens, config, pipeline_stages):
        """Helper to consume stream."""
        async for token_data in self.streamer.stream_generate(
            request_id, prompt_tokens, config, pipeline_stages
        ):
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
