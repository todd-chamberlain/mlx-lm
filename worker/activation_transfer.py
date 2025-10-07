"""
Activation Transfer Protocol for Distributed Inference

Handles efficient transfer of activations between pipeline stages with
compression, streaming, and metrics tracking.
"""

import asyncio
import logging
import time
import struct
from typing import Optional, AsyncIterator, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

import mlx.core as mx
import numpy as np

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    logging.warning("zstandard not available, compression disabled")

try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    logging.warning("lz4 not available, fallback compression disabled")


logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Compression algorithm types"""
    NONE = "none"
    ZSTD = "zstd"
    LZ4 = "lz4"


@dataclass
class TransferMetrics:
    """Metrics for activation transfer"""
    transfer_time: float = 0.0
    compression_time: float = 0.0
    decompression_time: float = 0.0
    serialization_time: float = 0.0
    deserialization_time: float = 0.0
    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 1.0
    throughput_mbps: float = 0.0


@dataclass
class SerializedActivation:
    """Serialized activation data"""
    data: bytes
    shape: list
    dtype: str
    compression: str


class ActivationTransfer:
    """
    Handles efficient transfer of MLX arrays between pipeline stages.
    
    Features:
    - MLX array serialization/deserialization
    - Optional compression (ZSTD, LZ4)
    - Streaming support for large activations
    - Transfer metrics and monitoring
    """
    
    def __init__(
        self,
        compression: CompressionType = CompressionType.ZSTD,
        compression_level: int = 3,
        chunk_size: int = 1024 * 1024,  # 1MB chunks for streaming
        enable_metrics: bool = True
    ):
        """
        Initialize activation transfer handler.
        
        Args:
            compression: Compression algorithm to use
            compression_level: Compression level (1-22 for ZSTD, 1-12 for LZ4)
            chunk_size: Size of chunks for streaming (bytes)
            enable_metrics: Whether to collect transfer metrics
        """
        self.compression = compression
        self.compression_level = compression_level
        self.chunk_size = chunk_size
        self.enable_metrics = enable_metrics
        
        # Validate compression availability
        if compression == CompressionType.ZSTD and not HAS_ZSTD:
            logger.warning("ZSTD not available, falling back to LZ4")
            self.compression = CompressionType.LZ4 if HAS_LZ4 else CompressionType.NONE
        
        if compression == CompressionType.LZ4 and not HAS_LZ4:
            logger.warning("LZ4 not available, disabling compression")
            self.compression = CompressionType.NONE
        
        # Initialize compressors
        self._init_compressors()
        
        # Metrics
        self.total_transfers = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        
        logger.info(f"ActivationTransfer initialized with {self.compression.value} compression")
    
    def _init_compressors(self):
        """Initialize compression contexts"""
        if self.compression == CompressionType.ZSTD and HAS_ZSTD:
            self.compressor = zstd.ZstdCompressor(level=self.compression_level)
            self.decompressor = zstd.ZstdDecompressor()
        elif self.compression == CompressionType.LZ4 and HAS_LZ4:
            # LZ4 doesn't need persistent contexts
            self.compressor = None
            self.decompressor = None
        else:
            self.compressor = None
            self.decompressor = None
    
    def serialize_kv_cache(
        self,
        cache: List[Tuple[mx.array, mx.array]]
    ) -> Tuple[bytes, TransferMetrics]:
        """
        Serialize KV cache for distributed transfer.
        
        Args:
            cache: List of (key, value) tuples for each layer
            
        Returns:
            Tuple of (serialized_bytes, metrics)
        """
        metrics = TransferMetrics()
        start_time = time.time()
        
        try:
            # Serialize number of layers
            num_layers = len(cache)
            data_parts = [struct.pack('I', num_layers)]
            
            total_original_size = 0
            
            # Serialize each (key, value) pair
            for key_cache, value_cache in cache:
                # Convert to numpy
                key_np = np.array(key_cache)
                value_np = np.array(value_cache)
                
                # Serialize key
                key_shape_bytes = struct.pack('I', len(key_np.shape))
                key_shape_bytes += struct.pack(f'{len(key_np.shape)}I', *key_np.shape)
                key_dtype_str = str(key_np.dtype)
                key_dtype_bytes = key_dtype_str.encode('utf-8')
                key_dtype_len = struct.pack('I', len(key_dtype_bytes))
                key_data = key_np.tobytes()
                
                # Serialize value
                value_shape_bytes = struct.pack('I', len(value_np.shape))
                value_shape_bytes += struct.pack(f'{len(value_np.shape)}I', *value_np.shape)
                value_dtype_str = str(value_np.dtype)
                value_dtype_bytes = value_dtype_str.encode('utf-8')
                value_dtype_len = struct.pack('I', len(value_dtype_bytes))
                value_data = value_np.tobytes()
                
                # Combine key and value
                data_parts.extend([
                    key_shape_bytes, key_dtype_len, key_dtype_bytes, key_data,
                    value_shape_bytes, value_dtype_len, value_dtype_bytes, value_data
                ])
                
                total_original_size += len(key_data) + len(value_data)
            
            # Combine all parts
            data = b''.join(data_parts)
            metrics.original_size = total_original_size
            
            # Compress if enabled
            if self.compression != CompressionType.NONE:
                compress_start = time.time()
                data = self._compress(data)
                metrics.compression_time = time.time() - compress_start
                metrics.compressed_size = len(data)
                metrics.compression_ratio = metrics.original_size / max(metrics.compressed_size, 1)
            else:
                metrics.compressed_size = len(data)
                metrics.compression_ratio = 1.0
            
            metrics.transfer_time = time.time() - start_time
            
            if self.enable_metrics:
                self.total_transfers += 1
                self.total_bytes_sent += metrics.compressed_size
            
            logger.debug(
                f"Serialized KV cache: layers={num_layers}, "
                f"original={metrics.original_size/1024:.1f}KB, "
                f"compressed={metrics.compressed_size/1024:.1f}KB, "
                f"ratio={metrics.compression_ratio:.2f}x"
            )
            
            return data, metrics
            
        except Exception as e:
            logger.error(f"Failed to serialize KV cache: {e}")
            raise
    
    def deserialize_kv_cache(
        self,
        data: bytes
    ) -> Tuple[List[Tuple[mx.array, mx.array]], TransferMetrics]:
        """
        Deserialize KV cache from distributed transfer.
        
        Args:
            data: Serialized KV cache bytes
            
        Returns:
            Tuple of (cache_list, metrics) where cache_list is [(key, value), ...]
        """
        metrics = TransferMetrics()
        start_time = time.time()
        
        try:
            metrics.compressed_size = len(data)
            
            # Decompress if needed
            if self.compression != CompressionType.NONE:
                decompress_start = time.time()
                data = self._decompress(data)
                metrics.decompression_time = time.time() - decompress_start
            
            metrics.original_size = len(data)
            
            # Parse number of layers
            offset = 0
            num_layers = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            
            cache = []
            
            # Deserialize each (key, value) pair
            for _ in range(num_layers):
                # Deserialize key
                key_ndim = struct.unpack('I', data[offset:offset+4])[0]
                offset += 4
                
                key_shape = struct.unpack(f'{key_ndim}I', data[offset:offset+4*key_ndim])
                offset += 4 * key_ndim
                
                key_dtype_len = struct.unpack('I', data[offset:offset+4])[0]
                offset += 4
                
                key_dtype_str = data[offset:offset+key_dtype_len].decode('utf-8')
                offset += key_dtype_len
                
                key_size = int(np.prod(key_shape)) * np.dtype(key_dtype_str).itemsize
                key_bytes = data[offset:offset+key_size]
                offset += key_size
                
                key_np = np.frombuffer(key_bytes, dtype=key_dtype_str).reshape(key_shape)
                key_array = mx.array(key_np)
                
                # Deserialize value
                value_ndim = struct.unpack('I', data[offset:offset+4])[0]
                offset += 4
                
                value_shape = struct.unpack(f'{value_ndim}I', data[offset:offset+4*value_ndim])
                offset += 4 * value_ndim
                
                value_dtype_len = struct.unpack('I', data[offset:offset+4])[0]
                offset += 4
                
                value_dtype_str = data[offset:offset+value_dtype_len].decode('utf-8')
                offset += value_dtype_len
                
                value_size = int(np.prod(value_shape)) * np.dtype(value_dtype_str).itemsize
                value_bytes = data[offset:offset+value_size]
                offset += value_size
                
                value_np = np.frombuffer(value_bytes, dtype=value_dtype_str).reshape(value_shape)
                value_array = mx.array(value_np)
                
                cache.append((key_array, value_array))
            
            metrics.transfer_time = time.time() - start_time
            
            if metrics.original_size > 0:
                metrics.compression_ratio = metrics.original_size / max(metrics.compressed_size, 1)
            
            if self.enable_metrics:
                self.total_bytes_received += metrics.compressed_size
            
            logger.debug(
                f"Deserialized KV cache: layers={num_layers}, "
                f"size={metrics.original_size/1024:.1f}KB"
            )
            
            return cache, metrics
            
        except Exception as e:
            logger.error(f"Failed to deserialize KV cache: {e}")
            raise
    
    def serialize_activation(self, activation: mx.array, metadata: Optional[Dict[str, Any]] = None) -> SerializedActivation:
        """
        Serialize MLX array to bytes with optional compression.
        
        Args:
            activation: MLX array to serialize
            metadata: Optional metadata dictionary (not serialized, for logging only)
            
        Returns:
            SerializedActivation object with data, shape, dtype, and compression info
        """
        start_time = time.time()
        
        try:
            # Convert MLX array to numpy for serialization
            serialize_start = time.time()
            np_array = np.array(activation)
            
            # Create header with shape and dtype info
            shape_bytes = struct.pack('I', len(np_array.shape))  # Number of dimensions
            shape_bytes += struct.pack(f'{len(np_array.shape)}I', *np_array.shape)
            dtype_str = str(np_array.dtype)
            dtype_bytes = dtype_str.encode('utf-8')
            dtype_len = struct.pack('I', len(dtype_bytes))
            
            # Serialize array data
            array_bytes = np_array.tobytes()
            original_size = len(array_bytes)
            
            # Combine header and data
            data = shape_bytes + dtype_len + dtype_bytes + array_bytes
            
            # Compress if enabled
            compression_str = self.compression.value
            if self.compression != CompressionType.NONE:
                compress_start = time.time()
                data = self._compress(data)
                compressed_size = len(data)
                compression_ratio = original_size / max(compressed_size, 1)
            else:
                compressed_size = len(data)
                compression_ratio = 1.0
            
            transfer_time = time.time() - start_time
            
            # Update global metrics
            if self.enable_metrics:
                self.total_transfers += 1
                self.total_bytes_sent += compressed_size
            
            logger.debug(
                f"Serialized activation: shape={np_array.shape}, "
                f"original={original_size/1024:.1f}KB, "
                f"compressed={compressed_size/1024:.1f}KB, "
                f"ratio={compression_ratio:.2f}x, "
                f"time={transfer_time*1000:.2f}ms"
            )
            
            return SerializedActivation(
                data=data,
                shape=list(np_array.shape),
                dtype=dtype_str,
                compression=compression_str
            )
            
        except Exception as e:
            logger.error(f"Failed to serialize activation: {e}")
            raise
    
    def deserialize_activation(
        self,
        data: bytes,
        shape: Optional[list] = None,
        dtype: Optional[str] = None,
        compression: Optional[str] = None
    ) -> mx.array:
        """
        Deserialize bytes to MLX array with optional decompression.
        
        Args:
            data: Serialized activation bytes
            shape: Optional shape (if provided, skips header parsing)
            dtype: Optional dtype (if provided, skips header parsing)
            compression: Optional compression type (if provided, overrides instance compression)
            
        Returns:
            MLX array
        """
        start_time = time.time()
        
        try:
            compressed_size = len(data)
            
            # Determine compression type
            if compression is not None:
                use_compression = compression != "none"
            else:
                use_compression = self.compression != CompressionType.NONE
            
            # Decompress if needed
            if use_compression:
                decompress_start = time.time()
                data = self._decompress(data)
            
            original_size = len(data)
            
            # Parse header if shape/dtype not provided
            if shape is None or dtype is None:
                offset = 0
                ndim = struct.unpack('I', data[offset:offset+4])[0]
                offset += 4
                
                shape = struct.unpack(f'{ndim}I', data[offset:offset+4*ndim])
                offset += 4 * ndim
                
                dtype_len = struct.unpack('I', data[offset:offset+4])[0]
                offset += 4
                
                dtype_str = data[offset:offset+dtype_len].decode('utf-8')
                offset += dtype_len
                
                array_bytes = data[offset:]
            else:
                dtype_str = dtype
                array_bytes = data
            
            # Deserialize array data
            np_array = np.frombuffer(array_bytes, dtype=dtype_str).reshape(shape)
            
            # Convert to MLX array
            mlx_array = mx.array(np_array)
            
            transfer_time = time.time() - start_time
            
            # Update global metrics
            if self.enable_metrics:
                self.total_bytes_received += compressed_size
            
            logger.debug(
                f"Deserialized activation: shape={shape}, "
                f"size={original_size/1024:.1f}KB, "
                f"time={transfer_time*1000:.2f}ms"
            )
            
            return mlx_array
            
        except Exception as e:
            logger.error(f"Failed to deserialize activation: {e}")
            raise
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data using configured algorithm"""
        if self.compression == CompressionType.ZSTD and HAS_ZSTD:
            return self.compressor.compress(data)
        elif self.compression == CompressionType.LZ4 and HAS_LZ4:
            return lz4.compress(data, compression_level=self.compression_level)
        else:
            return data
    
    def _decompress(self, data: bytes) -> bytes:
        """Decompress data using configured algorithm"""
        if self.compression == CompressionType.ZSTD and HAS_ZSTD:
            return self.decompressor.decompress(data)
        elif self.compression == CompressionType.LZ4 and HAS_LZ4:
            return lz4.decompress(data)
        else:
            return data
    
    async def stream_activation(
        self,
        activation: mx.array
    ) -> AsyncIterator[bytes]:
        """
        Stream activation in chunks for large transfers.
        
        Args:
            activation: MLX array to stream
            
        Yields:
            Chunks of serialized data
        """
        # Serialize activation
        serialized = self.serialize_activation(activation)
        data = serialized.data
        
        # Stream in chunks
        offset = 0
        while offset < len(data):
            chunk_end = min(offset + self.chunk_size, len(data))
            chunk = data[offset:chunk_end]
            yield chunk
            offset = chunk_end
            
            # Allow other tasks to run
            await asyncio.sleep(0)
    
    async def receive_streamed_activation(
        self,
        chunks: AsyncIterator[bytes]
    ) -> mx.array:
        """
        Receive and reassemble streamed activation.
        
        Args:
            chunks: Async iterator of data chunks
            
        Returns:
            MLX array
        """
        # Collect all chunks
        data_chunks = []
        async for chunk in chunks:
            data_chunks.append(chunk)
        
        # Reassemble
        data = b''.join(data_chunks)
        
        # Deserialize
        return self.deserialize_activation(data)
    
    def get_metrics_summary(self) -> dict:
        """Get summary of transfer metrics"""
        return {
            'total_transfers': self.total_transfers,
            'total_bytes_sent': self.total_bytes_sent,
            'total_bytes_received': self.total_bytes_received,
            'total_sent_mb': self.total_bytes_sent / 1024 / 1024,
            'total_received_mb': self.total_bytes_received / 1024 / 1024,
            'compression': self.compression.value,
            'compression_level': self.compression_level
        }
    
    def reset_metrics(self):
        """Reset transfer metrics"""
        self.total_transfers = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0


# Convenience functions for simple use cases

def serialize_mlx_array(array: mx.array, compression: bool = True) -> bytes:
    """
    Simple function to serialize an MLX array.
    
    Args:
        array: MLX array to serialize
        compression: Whether to use compression
        
    Returns:
        Serialized bytes
    """
    transfer = ActivationTransfer(
        compression=CompressionType.ZSTD if compression else CompressionType.NONE,
        enable_metrics=False
    )
    serialized = transfer.serialize_activation(array)
    return serialized.data


def deserialize_mlx_array(data: bytes, compression: bool = True) -> mx.array:
    """
    Simple function to deserialize an MLX array.
    
    Args:
        data: Serialized bytes
        compression: Whether data is compressed
        
    Returns:
        MLX array
    """
    transfer = ActivationTransfer(
        compression=CompressionType.ZSTD if compression else CompressionType.NONE,
        enable_metrics=False
    )
    array = transfer.deserialize_activation(data)
    return array
