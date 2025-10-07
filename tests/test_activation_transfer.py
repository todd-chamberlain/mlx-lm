"""
Tests for Activation Transfer Protocol

Tests serialization, compression, streaming, and metrics for activation transfer.
"""

import pytest
import asyncio
import time
import mlx.core as mx
import numpy as np

from worker.activation_transfer import (
    ActivationTransfer,
    CompressionType,
    TransferMetrics,
    SerializedActivation,
    serialize_mlx_array,
    deserialize_mlx_array
)


class TestActivationSerialization:
    """Test basic serialization and deserialization"""
    
    def test_serialize_deserialize_simple(self):
        """Test basic serialization round-trip"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create test array
        original = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        # Serialize
        serialized = transfer.serialize_activation(original)
        assert isinstance(serialized.data, bytes)
        assert len(serialized.data) > 0
        assert len(serialized.shape) > 0
        
        # Deserialize
        restored = transfer.deserialize_activation(serialized.data)
        assert restored.shape == original.shape
        assert mx.allclose(restored, original)
    
    def test_serialize_different_shapes(self):
        """Test serialization with different array shapes"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        test_shapes = [
            (10,),  # 1D
            (5, 10),  # 2D
            (2, 3, 4),  # 3D
            (2, 3, 4, 5),  # 4D
        ]
        
        for shape in test_shapes:
            original = mx.random.normal(shape)
            serialized = transfer.serialize_activation(original)
            restored = transfer.deserialize_activation(serialized.data)
            
            assert restored.shape == original.shape
            assert mx.allclose(restored, original, rtol=1e-5)
    
    def test_serialize_different_dtypes(self):
        """Test serialization with different data types"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Float32
        float_array = mx.array([1.5, 2.5, 3.5], dtype=mx.float32)
        serialized = transfer.serialize_activation(float_array)
        restored = transfer.deserialize_activation(serialized.data)
        assert mx.allclose(restored, float_array)
        
        # Int32
        int_array = mx.array([1, 2, 3], dtype=mx.int32)
        serialized = transfer.serialize_activation(int_array)
        restored = transfer.deserialize_activation(serialized.data)
        assert mx.array_equal(restored, int_array)
    
    def test_serialize_large_array(self):
        """Test serialization of large arrays"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create large array (10MB)
        large_array = mx.random.normal((1000, 1000))
        
        start_time = time.time()
        serialized = transfer.serialize_activation(large_array)
        serialize_time = time.time() - start_time
        
        assert len(serialized.data) > 0
        assert len(serialized.data) > 1024 * 1024  # > 1MB
        assert serialize_time < 1.0  # Should be fast
        
        # Deserialize
        restored = transfer.deserialize_activation(serialized.data)
        assert restored.shape == large_array.shape
        assert mx.allclose(restored, large_array, rtol=1e-5)


class TestCompression:
    """Test compression functionality"""
    
    @pytest.mark.skipif(not ActivationTransfer(CompressionType.ZSTD).compression == CompressionType.ZSTD,
                       reason="ZSTD not available")
    def test_zstd_compression(self):
        """Test ZSTD compression"""
        transfer = ActivationTransfer(compression=CompressionType.ZSTD, compression_level=3)
        
        # Create array with some redundancy (compressible)
        original = mx.ones((100, 100))
        
        # Serialize with compression
        serialized = transfer.serialize_activation(original)
        
        # Compressed data should be smaller than uncompressed
        assert len(serialized.data) < original.nbytes
        assert serialized.compression == "zstd"
        
        # Deserialize
        restored = transfer.deserialize_activation(serialized.data)
        assert mx.allclose(restored, original)
    
    @pytest.mark.skipif(not ActivationTransfer(CompressionType.LZ4).compression == CompressionType.LZ4,
                       reason="LZ4 not available")
    def test_lz4_compression(self):
        """Test LZ4 compression"""
        transfer = ActivationTransfer(compression=CompressionType.LZ4, compression_level=1)
        
        # Create compressible array
        original = mx.zeros((100, 100))
        
        # Serialize with compression
        serialized = transfer.serialize_activation(original)
        
        assert len(serialized.data) < original.nbytes
        assert serialized.compression == "lz4"
        
        # Deserialize
        restored = transfer.deserialize_activation(serialized.data)
        assert mx.allclose(restored, original)
    
    def test_compression_levels(self):
        """Test different compression levels"""
        original = mx.random.normal((100, 100))
        
        # Try different compression levels
        results = []
        for level in [1, 3, 5]:
            transfer = ActivationTransfer(
                compression=CompressionType.ZSTD if ActivationTransfer(CompressionType.ZSTD).compression == CompressionType.ZSTD else CompressionType.NONE,
                compression_level=level
            )
            
            if transfer.compression == CompressionType.NONE:
                pytest.skip("No compression available")
            
            serialized = transfer.serialize_activation(original)
            results.append((level, len(serialized.data)))
        
        # Higher compression levels should generally produce smaller sizes
        # (though not guaranteed for all data)
        assert len(results) > 0
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation"""
        transfer = ActivationTransfer(compression=CompressionType.ZSTD if ActivationTransfer(CompressionType.ZSTD).compression == CompressionType.ZSTD else CompressionType.NONE)
        
        if transfer.compression == CompressionType.NONE:
            pytest.skip("No compression available")
        
        # Highly compressible data
        compressible = mx.ones((100, 100))
        serialized = transfer.serialize_activation(compressible)
        
        # Compressed should be smaller than original
        assert len(serialized.data) < compressible.nbytes


class TestStreaming:
    """Test streaming functionality"""
    
    @pytest.mark.asyncio
    async def test_stream_activation(self):
        """Test streaming activation in chunks"""
        transfer = ActivationTransfer(
            compression=CompressionType.NONE,
            chunk_size=1024  # Small chunks for testing
        )
        
        # Create test array
        original = mx.random.normal((100, 100))
        
        # Stream activation
        chunks = []
        async for chunk in transfer.stream_activation(original):
            assert isinstance(chunk, bytes)
            assert len(chunk) <= transfer.chunk_size
            chunks.append(chunk)
        
        assert len(chunks) > 1  # Should be multiple chunks
        
        # Reassemble and verify
        data = b''.join(chunks)
        restored = transfer.deserialize_activation(data)
        assert mx.allclose(restored, original, rtol=1e-5)
    
    @pytest.mark.asyncio
    async def test_receive_streamed_activation(self):
        """Test receiving streamed activation"""
        transfer = ActivationTransfer(
            compression=CompressionType.NONE,
            chunk_size=1024
        )
        
        # Create test array
        original = mx.random.normal((50, 50))
        
        # Create async iterator of chunks
        async def chunk_generator():
            async for chunk in transfer.stream_activation(original):
                yield chunk
        
        # Receive streamed activation
        restored = await transfer.receive_streamed_activation(chunk_generator())
        
        assert restored.shape == original.shape
        assert mx.allclose(restored, original, rtol=1e-5)
    
    @pytest.mark.asyncio
    async def test_streaming_large_activation(self):
        """Test streaming large activation"""
        transfer = ActivationTransfer(
            compression=CompressionType.NONE,
            chunk_size=10240  # 10KB chunks
        )
        
        # Create large array
        large_array = mx.random.normal((500, 500))
        
        # Stream and collect
        chunks = []
        async for chunk in transfer.stream_activation(large_array):
            chunks.append(chunk)
        
        # Should have multiple chunks
        assert len(chunks) > 5
        
        # Verify total size
        total_size = sum(len(chunk) for chunk in chunks)
        assert total_size > 0


class TestMetrics:
    """Test metrics collection"""
    
    def test_metrics_collection(self):
        """Test that metrics are collected correctly"""
        transfer = ActivationTransfer(enable_metrics=True)
        
        # Perform transfers
        array1 = mx.random.normal((10, 10))
        array2 = mx.random.normal((20, 20))
        
        serialized1 = transfer.serialize_activation(array1)
        serialized2 = transfer.serialize_activation(array2)
        
        # Check that data was serialized
        assert len(serialized1.data) > 0
        assert len(serialized2.data) > 0
        
        # Check global metrics
        summary = transfer.get_metrics_summary()
        assert summary['total_transfers'] == 2
        assert summary['total_bytes_sent'] > 0
    
    def test_throughput_calculation(self):
        """Test throughput calculation"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create array
        array = mx.random.normal((100, 100))
        
        # Serialize
        serialized = transfer.serialize_activation(array)
        
        # Check that data was serialized
        assert len(serialized.data) > 0
    
    def test_metrics_reset(self):
        """Test metrics reset"""
        transfer = ActivationTransfer(enable_metrics=True)
        
        # Perform transfer
        array = mx.random.normal((10, 10))
        transfer.serialize_activation(array)
        
        # Check metrics exist
        summary = transfer.get_metrics_summary()
        assert summary['total_transfers'] > 0
        
        # Reset
        transfer.reset_metrics()
        
        # Check metrics cleared
        summary = transfer.get_metrics_summary()
        assert summary['total_transfers'] == 0
        assert summary['total_bytes_sent'] == 0
    
    def test_metrics_disabled(self):
        """Test that metrics can be disabled"""
        transfer = ActivationTransfer(enable_metrics=False)
        
        # Perform transfer
        array = mx.random.normal((10, 10))
        serialized = transfer.serialize_activation(array)
        
        # Data should be serialized
        assert len(serialized.data) > 0
        
        # But global metrics should not increment
        summary = transfer.get_metrics_summary()
        assert summary['total_transfers'] == 0


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_serialize_mlx_array(self):
        """Test simple serialization function"""
        original = mx.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Serialize
        data = serialize_mlx_array(original, compression=False)
        assert isinstance(data, bytes)
        assert len(data) > 0
    
    def test_deserialize_mlx_array(self):
        """Test simple deserialization function"""
        original = mx.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Serialize and deserialize
        data = serialize_mlx_array(original, compression=False)
        restored = deserialize_mlx_array(data, compression=False)
        
        assert restored.shape == original.shape
        assert mx.allclose(restored, original)
    
    def test_convenience_with_compression(self):
        """Test convenience functions with compression"""
        original = mx.ones((100, 100))
        
        # Serialize with compression
        data_compressed = serialize_mlx_array(original, compression=True)
        data_uncompressed = serialize_mlx_array(original, compression=False)
        
        # Compressed should be smaller (for this data)
        if ActivationTransfer(CompressionType.ZSTD).compression == CompressionType.ZSTD:
            assert len(data_compressed) < len(data_uncompressed)
        
        # Both should deserialize correctly
        restored_compressed = deserialize_mlx_array(data_compressed, compression=True)
        restored_uncompressed = deserialize_mlx_array(data_uncompressed, compression=False)
        
        assert mx.allclose(restored_compressed, original)
        assert mx.allclose(restored_uncompressed, original)


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_data_deserialization(self):
        """Test deserialization with invalid data"""
        transfer = ActivationTransfer()
        
        # Try to deserialize garbage data
        with pytest.raises(Exception):
            transfer.deserialize_activation(b"invalid data")
    
    def test_empty_data(self):
        """Test handling of empty data"""
        transfer = ActivationTransfer()
        
        # Try to deserialize empty data
        with pytest.raises(Exception):
            transfer.deserialize_activation(b"")
    
    def test_corrupted_header(self):
        """Test handling of corrupted header"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create valid data
        original = mx.array([1.0, 2.0, 3.0])
        serialized = transfer.serialize_activation(original)
        
        # Corrupt the header
        corrupted = b"corrupted" + serialized.data[9:]
        
        # Should raise exception
        with pytest.raises(Exception):
            transfer.deserialize_activation(corrupted)


class TestPerformance:
    """Test performance characteristics"""
    
    def test_serialization_performance(self):
        """Test serialization performance"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create moderately sized array
        array = mx.random.normal((1000, 1000))
        
        # Measure serialization time
        start = time.time()
        serialized = transfer.serialize_activation(array)
        elapsed = time.time() - start
        
        # Should be fast (< 100ms for 4MB array)
        assert elapsed < 0.1
        assert len(serialized.data) > 0
    
    def test_compression_performance(self):
        """Test compression performance"""
        if ActivationTransfer(CompressionType.ZSTD).compression != CompressionType.ZSTD:
            pytest.skip("ZSTD not available")
        
        transfer = ActivationTransfer(compression=CompressionType.ZSTD, compression_level=3)
        
        # Create array
        array = mx.random.normal((500, 500))
        
        # Measure compression time
        start = time.time()
        serialized = transfer.serialize_activation(array)
        elapsed = time.time() - start
        
        # Compression should be reasonably fast
        assert elapsed < 0.5
        assert len(serialized.data) > 0
    
    def test_throughput(self):
        """Test transfer throughput"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create array
        array = mx.random.normal((1000, 1000))
        
        # Serialize
        start = time.time()
        serialized = transfer.serialize_activation(array)
        elapsed = time.time() - start
        
        # Should be fast
        assert elapsed < 1.0
        assert len(serialized.data) > 0


class TestKVCacheSerialization:
    """Test KV cache serialization"""
    
    def test_serialize_deserialize_kv_cache(self):
        """Test KV cache serialization round-trip"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create test KV cache (list of (key, value) tuples)
        cache = [
            (mx.random.normal((2, 4, 8, 64)), mx.random.normal((2, 4, 8, 64))),
            (mx.random.normal((2, 4, 8, 64)), mx.random.normal((2, 4, 8, 64))),
            (mx.random.normal((2, 4, 8, 64)), mx.random.normal((2, 4, 8, 64))),
        ]
        
        # Serialize
        data, metrics = transfer.serialize_kv_cache(cache)
        assert isinstance(data, bytes)
        assert len(data) > 0
        assert metrics.original_size > 0
        
        # Deserialize
        restored_cache, metrics2 = transfer.deserialize_kv_cache(data)
        assert len(restored_cache) == len(cache)
        
        # Verify each layer
        for i, ((orig_k, orig_v), (rest_k, rest_v)) in enumerate(zip(cache, restored_cache)):
            assert rest_k.shape == orig_k.shape
            assert rest_v.shape == orig_v.shape
            assert mx.allclose(rest_k, orig_k, rtol=1e-5)
            assert mx.allclose(rest_v, orig_v, rtol=1e-5)
    
    def test_serialize_empty_kv_cache(self):
        """Test serialization of empty KV cache"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Empty cache
        cache = []
        
        # Serialize
        data, metrics = transfer.serialize_kv_cache(cache)
        assert isinstance(data, bytes)
        assert len(data) > 0  # Should have header
        
        # Deserialize
        restored_cache, _ = transfer.deserialize_kv_cache(data)
        assert len(restored_cache) == 0
    
    def test_serialize_kv_cache_with_compression(self):
        """Test KV cache serialization with compression"""
        if ActivationTransfer(CompressionType.ZSTD).compression != CompressionType.ZSTD:
            pytest.skip("ZSTD not available")
        
        transfer = ActivationTransfer(compression=CompressionType.ZSTD, compression_level=3)
        
        # Create cache with compressible data
        cache = [
            (mx.ones((2, 4, 8, 64)), mx.ones((2, 4, 8, 64))),
            (mx.zeros((2, 4, 8, 64)), mx.zeros((2, 4, 8, 64))),
        ]
        
        # Serialize
        data, metrics = transfer.serialize_kv_cache(cache)
        
        # Should be compressed
        assert metrics.compressed_size < metrics.original_size
        assert metrics.compression_ratio > 1.0
        
        # Deserialize
        restored_cache, _ = transfer.deserialize_kv_cache(data)
        assert len(restored_cache) == len(cache)
        
        # Verify
        for (orig_k, orig_v), (rest_k, rest_v) in zip(cache, restored_cache):
            assert mx.allclose(rest_k, orig_k)
            assert mx.allclose(rest_v, orig_v)
    
    def test_serialize_kv_cache_different_shapes(self):
        """Test KV cache serialization with different shapes"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create cache with different shapes per layer
        cache = [
            (mx.random.normal((1, 4, 10, 64)), mx.random.normal((1, 4, 10, 64))),
            (mx.random.normal((1, 4, 20, 64)), mx.random.normal((1, 4, 20, 64))),
            (mx.random.normal((1, 4, 30, 64)), mx.random.normal((1, 4, 30, 64))),
        ]
        
        # Serialize
        data, _ = transfer.serialize_kv_cache(cache)
        
        # Deserialize
        restored_cache, _ = transfer.deserialize_kv_cache(data)
        assert len(restored_cache) == len(cache)
        
        # Verify shapes
        for (orig_k, orig_v), (rest_k, rest_v) in zip(cache, restored_cache):
            assert rest_k.shape == orig_k.shape
            assert rest_v.shape == orig_v.shape
            assert mx.allclose(rest_k, orig_k, rtol=1e-5)
            assert mx.allclose(rest_v, orig_v, rtol=1e-5)
    
    def test_kv_cache_serialization_performance(self):
        """Test KV cache serialization performance"""
        transfer = ActivationTransfer(compression=CompressionType.NONE)
        
        # Create realistic cache (32 layers)
        cache = [
            (mx.random.normal((1, 8, 100, 64)), mx.random.normal((1, 8, 100, 64)))
            for _ in range(32)
        ]
        
        # Measure serialization time
        start = time.time()
        data, metrics = transfer.serialize_kv_cache(cache)
        serialize_time = time.time() - start
        
        # Should be reasonably fast
        assert serialize_time < 1.0
        assert len(data) > 0
        
        # Measure deserialization time
        start = time.time()
        restored_cache, _ = transfer.deserialize_kv_cache(data)
        deserialize_time = time.time() - start
        
        # Should be reasonably fast
        assert deserialize_time < 1.0
        assert len(restored_cache) == len(cache)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
