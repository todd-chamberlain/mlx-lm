"""
Tests for Distributed KV Cache Serialization

Tests multi-turn conversation support with distributed cache propagation.
"""

import pytest
import asyncio
import mlx.core as mx
from unittest.mock import Mock, AsyncMock, patch

from worker.distributed_forward import DistributedForwardPass
from worker.activation_transfer import ActivationTransfer, CompressionType
from common.kv_cache import KVCache


class TestDistributedCacheSerialization:
    """Test KV cache serialization in distributed forward pass"""
    
    def test_kv_cache_serialization_integration(self):
        """Test that KV cache can be serialized and deserialized"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create realistic KV cache
        cache = [
            (mx.random.normal((1, 8, 10, 64)), mx.random.normal((1, 8, 10, 64))),
            (mx.random.normal((1, 8, 10, 64)), mx.random.normal((1, 8, 10, 64))),
        ]
        
        # Serialize
        data, metrics = transfer.serialize_kv_cache(cache)
        assert len(data) > 0
        assert metrics.original_size > 0
        
        # Deserialize
        restored_cache, _ = transfer.deserialize_kv_cache(data)
        assert len(restored_cache) == len(cache)
        
        # Verify
        for (orig_k, orig_v), (rest_k, rest_v) in zip(cache, restored_cache):
            assert mx.allclose(orig_k, rest_k, rtol=1e-5)
            assert mx.allclose(orig_v, rest_v, rtol=1e-5)
    
    @pytest.mark.asyncio
    async def test_distributed_forward_with_cache_flow(self):
        """Test distributed forward pass with cache serialization flow"""
        # Setup pipeline
        pipeline_id = "test-pipeline"
        node_assignments = {
            "node1": [0, 1],
            "node2": [2, 3],
        }
        node_addresses = {
            "node1": "localhost:50051",
            "node2": "localhost:50052",
        }
        
        forward_pass = DistributedForwardPass(
            pipeline_id=pipeline_id,
            node_assignments=node_assignments,
            node_addresses=node_addresses,
            compression="none"
        )
        
        # Create test input and cache
        input_ids = mx.array([[1, 2, 3, 4]])
        cache = [
            (mx.random.normal((1, 8, 4, 64)), mx.random.normal((1, 8, 4, 64))),
            (mx.random.normal((1, 8, 4, 64)), mx.random.normal((1, 8, 4, 64))),
            (mx.random.normal((1, 8, 4, 64)), mx.random.normal((1, 8, 4, 64))),
            (mx.random.normal((1, 8, 4, 64)), mx.random.normal((1, 8, 4, 64))),
        ]
        
        # Mock gRPC stubs
        mock_response1 = Mock()
        mock_response1.output_data = b"test_output_1"
        mock_response1.shape = [1, 4, 768]
        mock_response1.dtype = "float32"
        mock_response1.compression = "none"
        
        # Serialize cache for response
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        cache_data, _ = transfer.serialize_kv_cache(cache[:2])
        mock_response1.updated_cache = cache_data
        
        mock_response2 = Mock()
        mock_response2.output_data = b"test_output_2"
        mock_response2.shape = [1, 4, 768]
        mock_response2.dtype = "float32"
        mock_response2.compression = "none"
        
        cache_data2, _ = transfer.serialize_kv_cache(cache[2:])
        mock_response2.updated_cache = cache_data2
        
        # Mock the stubs
        mock_stub1 = AsyncMock()
        mock_stub1.ForwardPass = AsyncMock(return_value=mock_response1)
        
        mock_stub2 = AsyncMock()
        mock_stub2.ForwardPass = AsyncMock(return_value=mock_response2)
        
        # Patch _get_stub to return our mocks
        async def mock_get_stub(node_id):
            if node_id == "node1":
                return mock_stub1
            else:
                return mock_stub2
        
        forward_pass._get_stub = mock_get_stub
        
        # Mock deserialize_activation to return test data
        def mock_deserialize(data, shape=None, dtype=None, compression=None):
            return mx.random.normal(shape if shape else (1, 4, 768))
        
        forward_pass.activation_transfer.deserialize_activation = mock_deserialize
        
        # Execute forward pass with cache
        try:
            output, updated_cache = await forward_pass.forward_with_cache(
                input_ids=input_ids,
                cache=cache,
                request_id="test-request"
            )
            
            # Verify output
            assert output is not None
            assert output.shape == (1, 4, 768)
            
            # Verify cache was updated
            assert len(updated_cache) == 4
            
            # Verify that ForwardPass was called with cache_data
            assert mock_stub1.ForwardPass.called
            assert mock_stub2.ForwardPass.called
            
            # Check that cache_data was included in requests
            call_args1 = mock_stub1.ForwardPass.call_args[0][0]
            assert hasattr(call_args1, 'cache_data')
            assert len(call_args1.cache_data) > 0  # Should have serialized cache
            
            call_args2 = mock_stub2.ForwardPass.call_args[0][0]
            assert hasattr(call_args2, 'cache_data')
            assert len(call_args2.cache_data) > 0
            
        finally:
            await forward_pass.close()
    
    def test_cache_serialization_preserves_shape_and_dtype(self):
        """Test that cache serialization preserves shape and dtype"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create cache with specific shapes and dtypes
        cache = [
            (
                mx.array(mx.random.normal((2, 4, 10, 64)), dtype=mx.float32),
                mx.array(mx.random.normal((2, 4, 10, 64)), dtype=mx.float32)
            ),
            (
                mx.array(mx.random.normal((2, 4, 15, 64)), dtype=mx.float32),
                mx.array(mx.random.normal((2, 4, 15, 64)), dtype=mx.float32)
            ),
        ]
        
        # Serialize and deserialize
        data, _ = transfer.serialize_kv_cache(cache)
        restored_cache, _ = transfer.deserialize_kv_cache(data)
        
        # Verify shapes and dtypes
        for i, ((orig_k, orig_v), (rest_k, rest_v)) in enumerate(zip(cache, restored_cache)):
            assert rest_k.shape == orig_k.shape, f"Layer {i} key shape mismatch"
            assert rest_v.shape == orig_v.shape, f"Layer {i} value shape mismatch"
            assert rest_k.dtype == orig_k.dtype, f"Layer {i} key dtype mismatch"
            assert rest_v.dtype == orig_v.dtype, f"Layer {i} value dtype mismatch"
    
    def test_empty_cache_handling(self):
        """Test that empty cache is handled correctly"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Empty cache
        cache = []
        
        # Serialize
        data, metrics = transfer.serialize_kv_cache(cache)
        assert len(data) > 0  # Should have header
        
        # Deserialize
        restored_cache, _ = transfer.deserialize_kv_cache(data)
        assert len(restored_cache) == 0
    
    def test_large_cache_serialization(self):
        """Test serialization of large cache (many layers)"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create large cache (32 layers, typical for LLMs)
        cache = [
            (mx.random.normal((1, 8, 100, 64)), mx.random.normal((1, 8, 100, 64)))
            for _ in range(32)
        ]
        
        # Serialize
        import time
        start = time.time()
        data, metrics = transfer.serialize_kv_cache(cache)
        serialize_time = time.time() - start
        
        # Should be reasonably fast
        assert serialize_time < 1.0
        assert len(data) > 0
        
        # Deserialize
        start = time.time()
        restored_cache, _ = transfer.deserialize_kv_cache(data)
        deserialize_time = time.time() - start
        
        # Should be reasonably fast
        assert deserialize_time < 1.0
        assert len(restored_cache) == 32
        
        # Spot check a few layers
        for i in [0, 15, 31]:
            assert mx.allclose(cache[i][0], restored_cache[i][0], rtol=1e-5)
            assert mx.allclose(cache[i][1], restored_cache[i][1], rtol=1e-5)


class TestKVCacheIntegration:
    """Test KV cache integration with distributed inference"""
    
    def test_kv_cache_manager_with_serialization(self):
        """Test KVCache manager with serialization"""
        kv_cache = KVCache(max_cache_size_bytes=100 * 1024 * 1024)  # 100MB
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create and store cache
        cache_key = "test-prompt-123"
        cache = [
            (mx.random.normal((1, 8, 10, 64)), mx.random.normal((1, 8, 10, 64))),
            (mx.random.normal((1, 8, 10, 64)), mx.random.normal((1, 8, 10, 64))),
        ]
        
        # Store in KV cache manager
        for i, (key, value) in enumerate(cache):
            kv_cache.put(f"{cache_key}-layer-{i}", key, value, sequence_length=10)
        
        # Retrieve and serialize
        retrieved_cache = []
        for i in range(len(cache)):
            result = kv_cache.get(f"{cache_key}-layer-{i}")
            if result:
                retrieved_cache.append(result)
        
        assert len(retrieved_cache) == len(cache)
        
        # Serialize retrieved cache
        data, metrics = transfer.serialize_kv_cache(retrieved_cache)
        assert len(data) > 0
        
        # Deserialize
        restored_cache, _ = transfer.deserialize_kv_cache(data)
        assert len(restored_cache) == len(cache)
        
        # Verify
        for (orig_k, orig_v), (rest_k, rest_v) in zip(cache, restored_cache):
            assert mx.allclose(orig_k, rest_k, rtol=1e-5)
            assert mx.allclose(orig_v, rest_v, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
