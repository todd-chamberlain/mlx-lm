"""
Tests for Distributed Forward Pass Implementation

Tests distributed forward pass execution, activation flow, synchronization,
error handling, and performance metrics.
"""

import pytest
import asyncio
import mlx.core as mx
import grpc
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List

from worker.distributed_forward import DistributedForwardPass, ForwardPassMetrics
from proto import distributed_pb2, distributed_pb2_grpc


class TestForwardPassMetrics:
    """Test forward pass metrics tracking."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ForwardPassMetrics()
        
        assert metrics.total_time == 0.0
        assert metrics.layer_times == {}
        assert metrics.transfer_times == {}
        assert metrics.total_transfers == 0
        assert metrics.total_bytes_transferred == 0
        assert metrics.errors == 0
        assert metrics.retries == 0
    
    def test_metrics_tracking(self):
        """Test metrics tracking during forward pass."""
        metrics = ForwardPassMetrics()
        
        # Simulate metrics updates
        metrics.layer_times[0] = 0.1
        metrics.layer_times[1] = 0.15
        metrics.total_transfers = 2
        metrics.total_bytes_transferred = 1024 * 1024
        metrics.total_time = 0.3
        
        assert len(metrics.layer_times) == 2
        assert metrics.total_transfers == 2
        assert metrics.total_bytes_transferred == 1024 * 1024


class TestDistributedForwardPass:
    """Test distributed forward pass implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline_id = "test-pipeline"
        self.node_assignments = {
            "node1": [0, 1, 2],
            "node2": [3, 4, 5],
            "node3": [6, 7]
        }
        self.node_addresses = {
            "node1": "localhost:50001",
            "node2": "localhost:50002",
            "node3": "localhost:50003"
        }
    
    def test_initialization(self):
        """Test distributed forward pass initialization."""
        forward_pass = DistributedForwardPass(
            pipeline_id=self.pipeline_id,
            node_assignments=self.node_assignments,
            node_addresses=self.node_addresses
        )
        
        assert forward_pass.pipeline_id == self.pipeline_id
        assert forward_pass.node_assignments == self.node_assignments
        assert forward_pass.node_addresses == self.node_addresses
        assert len(forward_pass.pipeline_stages) == 3
    
    def test_pipeline_stage_ordering(self):
        """Test pipeline stages are ordered correctly."""
        forward_pass = DistributedForwardPass(
            pipeline_id=self.pipeline_id,
            node_assignments=self.node_assignments,
            node_addresses=self.node_addresses
        )
        
        # Stages should be ordered by first layer index
        stages = forward_pass.pipeline_stages
        assert len(stages) == 3
        
        # Check ordering
        assert stages[0][0] == "node1"  # Layers 0-2
        assert stages[1][0] == "node2"  # Layers 3-5
        assert stages[2][0] == "node3"  # Layers 6-7
    
    def test_pipeline_stage_ordering_unordered_input(self):
        """Test pipeline stages with unordered node assignments."""
        unordered_assignments = {
            "node3": [6, 7],
            "node1": [0, 1, 2],
            "node2": [3, 4, 5]
        }
        
        forward_pass = DistributedForwardPass(
            pipeline_id=self.pipeline_id,
            node_assignments=unordered_assignments,
            node_addresses=self.node_addresses
        )
        
        # Should still be ordered correctly
        stages = forward_pass.pipeline_stages
        assert stages[0][0] == "node1"
        assert stages[1][0] == "node2"
        assert stages[2][0] == "node3"
    
    @pytest.mark.asyncio
    async def test_forward_pass_execution(self):
        """Test basic forward pass execution."""
        forward_pass = DistributedForwardPass(
            pipeline_id=self.pipeline_id,
            node_assignments=self.node_assignments,
            node_addresses=self.node_addresses
        )
        
        # Mock gRPC stubs
        mock_stubs = {}
        for node_id in self.node_assignments.keys():
            mock_stub = AsyncMock()
            mock_response = distributed_pb2.ForwardPassResponse(
                output_data=b"mock_output",
                shape=[1, 10, 4096],
                dtype="float32",
                compression="none"
            )
            mock_stub.ForwardPass = AsyncMock(return_value=mock_response)
            mock_stubs[node_id] = mock_stub
        
        # Mock activation transfer
        with patch.object(forward_pass, '_get_stub') as mock_get_stub:
            with patch.object(forward_pass.activation_transfer, 'serialize_activation') as mock_serialize:
                with patch.object(forward_pass.activation_transfer, 'deserialize_activation') as mock_deserialize:
                    # Setup mocks
                    mock_get_stub.side_effect = lambda node_id: mock_stubs[node_id]
                    
                    mock_serialized = Mock()
                    mock_serialized.data = b"mock_data"
                    mock_serialized.compression = "none"
                    mock_serialized.shape = [1, 10, 4096]
                    mock_serialized.dtype = "float32"
                    mock_serialize.return_value = mock_serialized
                    
                    mock_deserialize.return_value = mx.zeros((1, 10, 4096))
                    
                    # Execute forward pass
                    input_ids = mx.array([[1, 2, 3, 4, 5]])
                    output = await forward_pass.forward(input_ids, request_id="test-request")
                    
                    # Verify execution
                    assert output.shape == (1, 10, 4096)
                    assert mock_serialize.call_count == 3  # One per stage
                    assert mock_deserialize.call_count == 3
    
    @pytest.mark.asyncio
    async def test_forward_pass_with_metrics(self):
        """Test forward pass metrics collection."""
        forward_pass = DistributedForwardPass(
            pipeline_id=self.pipeline_id,
            node_assignments=self.node_assignments,
            node_addresses=self.node_addresses
        )
        
        # Mock execution
        mock_stubs = {}
        for node_id in self.node_assignments.keys():
            mock_stub = AsyncMock()
            mock_response = distributed_pb2.ForwardPassResponse(
                output_data=b"x" * 1024,  # 1KB
                shape=[1, 10, 4096],
                dtype="float32",
                compression="none"
            )
            mock_stub.ForwardPass = AsyncMock(return_value=mock_response)
            mock_stubs[node_id] = mock_stub
        
        with patch.object(forward_pass, '_get_stub') as mock_get_stub:
            with patch.object(forward_pass.activation_transfer, 'serialize_activation') as mock_serialize:
                with patch.object(forward_pass.activation_transfer, 'deserialize_activation') as mock_deserialize:
                    mock_get_stub.side_effect = lambda node_id: mock_stubs[node_id]
                    
                    mock_serialized = Mock()
                    mock_serialized.data = b"x" * 1024
                    mock_serialized.compression = "none"
                    mock_serialized.shape = [1, 10, 4096]
                    mock_serialized.dtype = "float32"
                    mock_serialize.return_value = mock_serialized
                    
                    mock_deserialize.return_value = mx.zeros((1, 10, 4096))
                    
                    # Execute
                    input_ids = mx.array([[1, 2, 3, 4, 5]])
                    await forward_pass.forward(input_ids)
                    
                    # Check metrics
                    metrics = forward_pass.get_metrics()
                    assert metrics['total_time'] > 0
                    assert metrics['total_transfers'] == 3
                    assert metrics['total_bytes_transferred'] > 0
                    assert metrics['errors'] == 0
    
    @pytest.mark.asyncio
    async def test_forward_pass_error_handling(self):
        """Test error handling during forward pass."""
        forward_pass = DistributedForwardPass(
            pipeline_id=self.pipeline_id,
            node_assignments=self.node_assignments,
            node_addresses=self.node_addresses,
            max_retries=1  # Limit retries for test
        )
        
        # Mock stub that raises error
        mock_stub = AsyncMock()
        mock_stub.ForwardPass = AsyncMock(
            side_effect=grpc.aio.AioRpcError(
                code=grpc.StatusCode.UNAVAILABLE,
                initial_metadata=None,
                trailing_metadata=None,
                details="Node unavailable"
            )
        )
        
        with patch.object(forward_pass, '_get_stub', return_value=mock_stub):
            with patch.object(forward_pass.activation_transfer, 'serialize_activation') as mock_serialize:
                mock_serialized = Mock()
                mock_serialized.data = b"mock_data"
                mock_serialized.compression = "none"
                mock_serialized.shape = [1, 10, 4096]
                mock_serialized.dtype = "float32"
                mock_serialize.return_value = mock_serialized
                
                # Should raise error after retries
                input_ids = mx.array([[1, 2, 3, 4, 5]])
                
                with pytest.raises(Exception):
                    await forward_pass.forward(input_ids)
                
                # Check error metrics
                assert forward_pass.metrics.errors > 0
    
    @pytest.mark.asyncio
    async def test_forward_pass_with_cache(self):
        """Test forward pass with KV cache."""
        forward_pass = DistributedForwardPass(
            pipeline_id=self.pipeline_id,
            node_assignments=self.node_assignments,
            node_addresses=self.node_addresses
        )
        
        # Mock execution
        mock_stubs = {}
        for node_id in self.node_assignments.keys():
            mock_stub = AsyncMock()
            mock_response = distributed_pb2.ForwardPassResponse(
                output_data=b"mock_output",
                shape=[1, 1, 4096],  # Single token output
                dtype="float32",
                compression="none"
            )
            mock_stub.ForwardPass = AsyncMock(return_value=mock_response)
            mock_stubs[node_id] = mock_stub
        
        with patch.object(forward_pass, '_get_stub') as mock_get_stub:
            with patch.object(forward_pass.activation_transfer, 'serialize_activation') as mock_serialize:
                with patch.object(forward_pass.activation_transfer, 'deserialize_activation') as mock_deserialize:
                    mock_get_stub.side_effect = lambda node_id: mock_stubs[node_id]
                    
                    mock_serialized = Mock()
                    mock_serialized.data = b"mock_data"
                    mock_serialized.compression = "none"
                    mock_serialized.shape = [1, 1, 4096]
                    mock_serialized.dtype = "float32"
                    mock_serialize.return_value = mock_serialized
                    
                    mock_deserialize.return_value = mx.zeros((1, 1, 4096))
                    
                    # Execute with cache
                    input_ids = mx.array([[5]])  # Single new token
                    cache = []  # Empty cache for simplicity
                    
                    output, updated_cache = await forward_pass.forward_with_cache(
                        input_ids, cache=cache
                    )
                    
                    # Verify execution
                    assert output.shape == (1, 1, 4096)
                    assert isinstance(updated_cache, list)
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        forward_pass = DistributedForwardPass(
            pipeline_id=self.pipeline_id,
            node_assignments=self.node_assignments,
            node_addresses=self.node_addresses
        )
        
        # Set some metrics
        forward_pass.metrics.total_time = 1.0
        forward_pass.metrics.total_transfers = 10
        forward_pass.metrics.errors = 2
        
        # Reset
        forward_pass.reset_metrics()
        
        # Verify reset
        assert forward_pass.metrics.total_time == 0.0
        assert forward_pass.metrics.total_transfers == 0
        assert forward_pass.metrics.errors == 0
    
    @pytest.mark.asyncio
    async def test_close_connections(self):
        """Test closing gRPC connections."""
        forward_pass = DistributedForwardPass(
            pipeline_id=self.pipeline_id,
            node_assignments=self.node_assignments,
            node_addresses=self.node_addresses
        )
        
        # Mock channels
        mock_channels = {}
        for node_id in self.node_assignments.keys():
            mock_channel = AsyncMock()
            mock_channel.close = AsyncMock()
            mock_channels[node_id] = mock_channel
        
        forward_pass._channels = mock_channels
        
        # Close
        await forward_pass.close()
        
        # Verify all channels closed
        for channel in mock_channels.values():
            channel.close.assert_called_once()
        
        assert len(forward_pass._channels) == 0
        assert len(forward_pass._stubs) == 0


class TestForwardPassIntegration:
    """Integration tests for distributed forward pass."""
    
    @pytest.mark.asyncio
    async def test_multi_stage_pipeline(self):
        """Test multi-stage pipeline execution."""
        pipeline_id = "integration-test"
        node_assignments = {
            "node1": [0, 1],
            "node2": [2, 3],
            "node3": [4, 5]
        }
        node_addresses = {
            "node1": "localhost:50001",
            "node2": "localhost:50002",
            "node3": "localhost:50003"
        }
        
        forward_pass = DistributedForwardPass(
            pipeline_id=pipeline_id,
            node_assignments=node_assignments,
            node_addresses=node_addresses
        )
        
        # Verify pipeline structure
        assert len(forward_pass.pipeline_stages) == 3
        assert forward_pass.pipeline_stages[0][1] == [0, 1]
        assert forward_pass.pipeline_stages[1][1] == [2, 3]
        assert forward_pass.pipeline_stages[2][1] == [4, 5]
    
    @pytest.mark.asyncio
    async def test_compression_options(self):
        """Test different compression options."""
        from worker.activation_transfer import CompressionType
        
        for compression in ["none", "zstd", "lz4"]:
            forward_pass = DistributedForwardPass(
                pipeline_id="test-compression",
                node_assignments={"node1": [0, 1]},
                node_addresses={"node1": "localhost:50001"},
                compression=compression
            )
            
            # Verify compression string is stored
            assert forward_pass.compression == compression
            
            # Verify activation transfer has a valid compression type
            # (may fall back if library not available)
            assert isinstance(forward_pass.activation_transfer.compression, CompressionType)
    
    def test_metrics_aggregation(self):
        """Test metrics aggregation across stages."""
        forward_pass = DistributedForwardPass(
            pipeline_id="test-metrics",
            node_assignments={
                "node1": [0, 1, 2],
                "node2": [3, 4, 5]
            },
            node_addresses={
                "node1": "localhost:50001",
                "node2": "localhost:50002"
            }
        )
        
        # Simulate metrics
        forward_pass.metrics.layer_times = {0: 0.1, 3: 0.15}
        forward_pass.metrics.total_time = 0.3
        forward_pass.metrics.total_bytes_transferred = 2048
        
        metrics = forward_pass.get_metrics()
        
        assert metrics['avg_layer_time'] == 0.125
        assert metrics['throughput_mbps'] > 0
        assert metrics['total_time'] == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
