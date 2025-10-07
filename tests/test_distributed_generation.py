"""
Tests for distributed token generation.

Tests iterative generation through pipeline, KV cache management,
generation stopping criteria, and streaming support.
"""

import pytest
import asyncio
import mlx.core as mx
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from worker.distributed_generation import (
    DistributedGenerator,
    GenerationState,
    GenerationResult,
    KVCacheManager
)
from worker.distributed_forward import DistributedForwardPass
from worker.activation_transfer import ActivationTransfer


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.eos_token_id = 2
        self.vocab_size = 32000
    
    def encode(self, text: str) -> List[int]:
        """Mock encode."""
        return [1, 100, 200, 300]  # Mock token IDs
    
    def decode(self, tokens: List[int]) -> str:
        """Mock decode."""
        return " ".join([f"token_{t}" for t in tokens])


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def mock_forward_pass():
    """Create mock forward pass."""
    forward_pass = AsyncMock(spec=DistributedForwardPass)
    
    # Mock forward pass to return logits
    async def mock_forward(input_ids, cache=None, return_cache=False):
        batch_size, seq_len = input_ids.shape
        vocab_size = 32000
        # Return random logits
        logits = mx.random.normal((batch_size, seq_len, vocab_size))
        return logits
    
    forward_pass.forward = mock_forward
    return forward_pass


@pytest.fixture
def mock_activation_transfer():
    """Create mock activation transfer."""
    return Mock(spec=ActivationTransfer)


@pytest.fixture
def generator(mock_forward_pass, mock_activation_transfer, mock_tokenizer):
    """Create distributed generator."""
    return DistributedGenerator(
        forward_pass=mock_forward_pass,
        activation_transfer=mock_activation_transfer,
        tokenizer=mock_tokenizer,
        node_id="test-node-1"
    )


class TestGenerationState:
    """Test generation state management."""
    
    def test_generation_state_initialization(self):
        """Test generation state initialization."""
        state = GenerationState(
            max_tokens=100,
            temperature=0.8,
            top_p=0.9,
            top_k=50
        )
        
        assert state.max_tokens == 100
        assert state.temperature == 0.8
        assert state.top_p == 0.9
        assert state.top_k == 50
        assert state.generated_tokens == []
        assert state.current_position == 0
        assert not state.finished
        assert state.stop_reason is None
    
    def test_generation_state_with_stop_sequences(self):
        """Test generation state with stop sequences."""
        state = GenerationState(
            max_tokens=100,
            stop_sequences=["</s>", "\n\n"]
        )
        
        assert len(state.stop_sequences) == 2
        assert "</s>" in state.stop_sequences
        assert "\n\n" in state.stop_sequences


class TestDistributedGenerator:
    """Test distributed generator functionality."""
    
    @pytest.mark.asyncio
    async def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.node_id == "test-node-1"
        assert generator.current_state is None
        assert generator.total_tokens_generated == 0
        assert generator.total_generation_time == 0.0
    
    @pytest.mark.asyncio
    async def test_basic_generation(self, generator):
        """Test basic token generation."""
        prompt_tokens = [1, 100, 200]
        
        # Generate tokens (non-streaming)
        results = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=5,
            stream=False
        ):
            results.append(result)
        
        # Should get one final result
        assert len(results) == 1
        final_result = results[0]
        
        assert isinstance(final_result, GenerationResult)
        assert len(final_result.tokens) > 0
        assert len(final_result.tokens) <= 5
        assert final_result.stop_reason in ["max_tokens", "complete", "eos_token"]
        assert "tokens_generated" in final_result.metrics
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self, generator):
        """Test streaming token generation."""
        prompt_tokens = [1, 100, 200]
        
        # Generate tokens with streaming
        results = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=5,
            stream=True
        ):
            results.append(result)
        
        # Should get multiple partial results plus final
        assert len(results) > 1
        
        # Check partial results
        for i, result in enumerate(results[:-1]):
            assert result.metrics["partial"] == True
            assert len(result.tokens) == i + 1
        
        # Check final result
        final_result = results[-1]
        assert final_result.metrics["partial"] == False
        assert final_result.stop_reason != "in_progress"
    
    @pytest.mark.asyncio
    async def test_generation_with_temperature(self, generator):
        """Test generation with different temperature."""
        prompt_tokens = [1, 100, 200]
        
        # Generate with low temperature (more deterministic)
        results_low = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=3,
            temperature=0.1,
            stream=False
        ):
            results_low.append(result)
        
        # Generate with high temperature (more random)
        results_high = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=3,
            temperature=2.0,
            stream=False
        ):
            results_high.append(result)
        
        # Both should complete successfully
        assert len(results_low) == 1
        assert len(results_high) == 1
    
    @pytest.mark.asyncio
    async def test_generation_with_top_p(self, generator):
        """Test generation with nucleus sampling."""
        prompt_tokens = [1, 100, 200]
        
        results = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=3,
            top_p=0.9,
            stream=False
        ):
            results.append(result)
        
        assert len(results) == 1
        assert len(results[0].tokens) > 0
    
    @pytest.mark.asyncio
    async def test_generation_with_top_k(self, generator):
        """Test generation with top-k sampling."""
        prompt_tokens = [1, 100, 200]
        
        results = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=3,
            top_k=50,
            stream=False
        ):
            results.append(result)
        
        assert len(results) == 1
        assert len(results[0].tokens) > 0
    
    @pytest.mark.asyncio
    async def test_generation_with_repetition_penalty(self, generator):
        """Test generation with repetition penalty."""
        prompt_tokens = [1, 100, 200]
        
        results = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=5,
            repetition_penalty=1.2,
            stream=False
        ):
            results.append(result)
        
        assert len(results) == 1
        assert len(results[0].tokens) > 0
    
    @pytest.mark.asyncio
    async def test_max_tokens_stopping(self, generator):
        """Test generation stops at max_tokens."""
        prompt_tokens = [1, 100, 200]
        max_tokens = 3
        
        results = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            stream=False
        ):
            results.append(result)
        
        final_result = results[0]
        assert len(final_result.tokens) <= max_tokens
        if len(final_result.tokens) == max_tokens:
            assert final_result.stop_reason == "max_tokens"
    
    @pytest.mark.asyncio
    async def test_eos_token_stopping(self, generator, mock_tokenizer):
        """Test generation stops on EOS token."""
        prompt_tokens = [1, 100, 200]
        
        # Mock sampling to return EOS token
        original_sample = generator._sample_token
        call_count = [0]
        
        def mock_sample(logits, state):
            call_count[0] += 1
            if call_count[0] == 2:  # Return EOS on second call
                return mock_tokenizer.eos_token_id
            return original_sample(logits, state)
        
        generator._sample_token = mock_sample
        
        results = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=10,
            stream=False
        ):
            results.append(result)
        
        final_result = results[0]
        assert final_result.stop_reason == "eos_token"
        assert mock_tokenizer.eos_token_id in final_result.tokens
    
    @pytest.mark.asyncio
    async def test_stop_sequence_detection(self, generator):
        """Test generation stops on stop sequence."""
        prompt_tokens = [1, 100, 200]
        
        # Mock tokenizer decode to return stop sequence
        original_decode = generator.tokenizer.decode
        
        def mock_decode(tokens):
            if len(tokens) >= 3:
                return "some text </s> more text"
            return original_decode(tokens)
        
        generator.tokenizer.decode = mock_decode
        
        results = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=10,
            stop_sequences=["</s>"],
            stream=False
        ):
            results.append(result)
        
        final_result = results[0]
        assert "stop_sequence" in final_result.stop_reason
    
    @pytest.mark.asyncio
    async def test_generation_metrics(self, generator):
        """Test generation metrics collection."""
        prompt_tokens = [1, 100, 200]
        
        results = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=5,
            stream=False
        ):
            results.append(result)
        
        final_result = results[0]
        metrics = final_result.metrics
        
        assert "tokens_generated" in metrics
        assert "tokens_per_second" in metrics
        assert "total_latency" in metrics
        assert metrics["tokens_generated"] > 0
        assert metrics["tokens_per_second"] >= 0
        assert metrics["total_latency"] > 0
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, generator):
        """Test getting generator metrics."""
        # Initial metrics
        metrics = generator.get_metrics()
        assert metrics["total_tokens_generated"] == 0
        assert metrics["total_generation_time"] == 0.0
        assert metrics["active_generation"] == False
        
        # Generate some tokens
        prompt_tokens = [1, 100, 200]
        async for _ in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=3,
            stream=False
        ):
            pass
        
        # Check updated metrics
        metrics = generator.get_metrics()
        assert metrics["total_tokens_generated"] > 0
        assert metrics["total_generation_time"] > 0
        assert metrics["average_tokens_per_second"] >= 0
    
    @pytest.mark.asyncio
    async def test_cleanup(self, generator):
        """Test generator cleanup."""
        await generator.cleanup()
        
        assert generator.current_state is None


class TestKVCacheManager:
    """Test KV cache management."""
    
    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        manager = KVCacheManager(node_id="test-node", num_layers=32)
        
        assert manager.node_id == "test-node"
        assert manager.num_layers == 32
        assert len(manager.caches) == 0
    
    def test_initialize_cache(self):
        """Test cache initialization for request."""
        manager = KVCacheManager(node_id="test-node", num_layers=32)
        
        request_id = "req-123"
        manager.initialize_cache(request_id)
        
        assert request_id in manager.caches
        assert isinstance(manager.caches[request_id], list)
    
    def test_get_cache(self):
        """Test getting cache for request."""
        manager = KVCacheManager(node_id="test-node", num_layers=32)
        
        request_id = "req-123"
        manager.initialize_cache(request_id)
        
        cache = manager.get_cache(request_id)
        assert cache is not None
        assert isinstance(cache, list)
        
        # Non-existent request
        cache = manager.get_cache("non-existent")
        assert cache is None
    
    def test_update_cache(self):
        """Test updating cache for specific layer."""
        manager = KVCacheManager(node_id="test-node", num_layers=32)
        
        request_id = "req-123"
        manager.initialize_cache(request_id)
        
        # Create mock key and value tensors
        key = mx.random.normal((1, 8, 128, 64))
        value = mx.random.normal((1, 8, 128, 64))
        
        # Update cache for layer 0
        manager.update_cache(request_id, 0, key, value)
        
        cache = manager.get_cache(request_id)
        assert len(cache) > 0
        assert cache[0] == (key, value)
    
    def test_update_cache_auto_initialize(self):
        """Test cache auto-initialization on update."""
        manager = KVCacheManager(node_id="test-node", num_layers=32)
        
        request_id = "req-123"
        key = mx.random.normal((1, 8, 128, 64))
        value = mx.random.normal((1, 8, 128, 64))
        
        # Update without explicit initialization
        manager.update_cache(request_id, 0, key, value)
        
        assert request_id in manager.caches
        cache = manager.get_cache(request_id)
        assert cache[0] == (key, value)
    
    def test_clear_cache(self):
        """Test clearing cache for request."""
        manager = KVCacheManager(node_id="test-node", num_layers=32)
        
        request_id = "req-123"
        manager.initialize_cache(request_id)
        
        assert request_id in manager.caches
        
        manager.clear_cache(request_id)
        
        assert request_id not in manager.caches
    
    def test_get_memory_usage(self):
        """Test getting memory usage statistics."""
        manager = KVCacheManager(node_id="test-node", num_layers=32)
        
        # Initial usage
        usage = manager.get_memory_usage()
        assert usage["num_active_caches"] == 0
        assert usage["total_memory_bytes"] == 0
        
        # Add some caches
        for i in range(3):
            request_id = f"req-{i}"
            manager.initialize_cache(request_id)
            
            key = mx.random.normal((1, 8, 128, 64))
            value = mx.random.normal((1, 8, 128, 64))
            manager.update_cache(request_id, 0, key, value)
        
        # Check usage
        usage = manager.get_memory_usage()
        assert usage["num_active_caches"] == 3
        assert usage["total_memory_bytes"] > 0
        assert usage["total_memory_mb"] > 0


class TestSamplingStrategies:
    """Test different sampling strategies."""
    
    @pytest.mark.asyncio
    async def test_greedy_sampling(self, generator):
        """Test greedy sampling (temperature=0)."""
        prompt_tokens = [1, 100, 200]
        
        # Note: temperature=0 would cause division by zero,
        # so we use very low temperature instead
        results = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=3,
            temperature=0.01,
            stream=False
        ):
            results.append(result)
        
        assert len(results) == 1
        assert len(results[0].tokens) > 0
    
    @pytest.mark.asyncio
    async def test_combined_sampling(self, generator):
        """Test combined top-k and top-p sampling."""
        prompt_tokens = [1, 100, 200]
        
        results = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=3,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            stream=False
        ):
            results.append(result)
        
        assert len(results) == 1
        assert len(results[0].tokens) > 0


class TestErrorHandling:
    """Test error handling in generation."""
    
    @pytest.mark.asyncio
    async def test_forward_pass_error(self, generator, mock_forward_pass):
        """Test handling of forward pass errors."""
        prompt_tokens = [1, 100, 200]
        
        # Make forward pass raise error
        async def error_forward(*args, **kwargs):
            raise RuntimeError("Forward pass failed")
        
        mock_forward_pass.forward = error_forward
        
        # Generation should raise error
        with pytest.raises(RuntimeError):
            async for _ in generator.generate(
                prompt_tokens=prompt_tokens,
                max_tokens=3,
                stream=False
            ):
                pass
    
    @pytest.mark.asyncio
    async def test_empty_prompt(self, generator):
        """Test handling of empty prompt."""
        prompt_tokens = []
        
        # Should handle empty prompt gracefully
        results = []
        async for result in generator.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=3,
            stream=False
        ):
            results.append(result)
        
        # Should still generate tokens
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
