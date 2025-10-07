"""
Activation Transfer Protocol for Distributed Inference

Handles efficient transfer of activations between pipeline stages with
compression, streaming, and metrics tracking.
"""

import asyncio
import logging
import time
import struct
from typing import Optional, AsyncIterator, Tuple
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
    
    def serialize_activation(self, activation: mx.array) -> Tuple[bytes, TransferMetrics]:
        """
        Serialize MLX array to bytes with optional compression.
        
        Args:
            activation: MLX array to serialize
            
        Returns:
            Tuple of (serialized_bytes, metrics)
        """
        metrics = TransferMetrics()
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
            metrics.serialization_time = time.time() - serialize_start
            metrics.original_size = len(array_bytes)
            
            # Combine header and data
            data = shape_bytes + dtype_len + dtype_bytes + array_bytes
            
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
            
            # Calculate throughput
            if metrics.transfer_time > 0:
                metrics.throughput_mbps = (metrics.original_size / 1024 / 1024) / metrics.transfer_time
            
            # Update global metrics
            if self.enable_metrics:
                self.total_transfers += 1
                self.total_bytes_sent += metrics.compressed_size
            
            logger.debug(
                f"Serialized activation: shape={np_array.shape}, "
                f"original={metrics.original_size/1024:.1f}KB, "
                f"compressed={metrics.compressed_size/1024:.1f}KB, "
                f"ratio={metrics.compression_ratio:.2f}x, "
                f"time={metrics.transfer_time*1000:.2f}ms"
            )
            
            return data, metrics
            
        except Exception as e:
            logger.error(f"Failed to serialize activation: {e}")
            raise
    
    def deserialize_activation(self, data: bytes) -> Tuple[mx.array, TransferMetrics]:
        """
        Deserialize bytes to MLX array with optional decompression.
        
        Args:
            data: Serialized activation bytes
            
        Returns:
            Tuple of (mlx_array, metrics)
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
            
            # Parse header
            offset = 0
            ndim = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            
            shape = struct.unpack(f'{ndim}I', data[offset:offset+4*ndim])
            offset += 4 * ndim
            
            dtype_len = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            
            dtype_str = data[offset:offset+dtype_len].decode('utf-8')
            offset += dtype_len
            
            # Deserialize array data
            deserialize_start = time.time()
            array_bytes = data[offset:]
            np_array = np.frombuffer(array_bytes, dtype=dtype_str).reshape(shape)
            
            # Convert to MLX array
            mlx_array = mx.array(np_array)
            metrics.deserialization_time = time.time() - deserialize_start
            
            metrics.transfer_time = time.time() - start_time
            
            # Calculate metrics
            if metrics.original_size > 0:
                metrics.compression_ratio = metrics.original_size / max(metrics.compressed_size, 1)
            
            if metrics.transfer_time > 0:
                metrics.throughput_mbps = (metrics.original_size / 1024 / 1024) / metrics.transfer_time
            
            # Update global metrics
            if self.enable_metrics:
                self.total_bytes_received += metrics.compressed_size
            
            logger.debug(
                f"Deserialized activation: shape={shape}, "
                f"size={metrics.original_size/1024:.1f}KB, "
                f"time={metrics.transfer_time*1000:.2f}ms"
            )
            
            return mlx_array, metrics
            
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
        data, metrics = self.serialize_activation(activation)
        
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
    ) -> Tuple[mx.array, TransferMetrics]:
        """
        Receive and reassemble streamed activation.
        
        Args:
            chunks: Async iterator of data chunks
            
        Returns:
            Tuple of (mlx_array, metrics)
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
    data, _ = transfer.serialize_activation(array)
    return data


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
    array, _ = transfer.deserialize_activation(data)
    return array
