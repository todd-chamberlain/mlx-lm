"""
Additional tests for gRPC connection handling and error scenarios.

Tests connection management, network failures, timeout scenarios,
and edge cases in gRPC communication.
"""

import asyncio
import pytest
import grpc
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time
import threading

from proto import inference_pb2
from proto.grpc_base import (
    GRPCServerBase, 
    GRPCClientBase, 
    RetryConfig
)
from proto.errors import MLXError, MLXErrorCode
from proto.validation import ValidationError


class TestConnectionManagement:
    """Test connection establishment and management."""
    
    @pytest.mark.asyncio
    async def test_connection_reuse(self):
        """Test that existing connections are reused when healthy."""
        client = GRPCClientBase("localhost:50051")
        
        # Mock a healthy channel
        mock_channel = Mock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.READY
        client.channel = mock_channel
        
        with patch.object(client, '_wait_for_ready', new_callable=AsyncMock):
            # First connect should return existing channel
            channel1 = await client.connect()
            assert channel1 == mock_channel
            
            # Second connect should reuse the same channel
            channel2 = await client.connect()
            assert channel2 == mock_channel
            assert channel1 is channel2
    
    @pytest.mark.asyncio
    async def test_connection_recreation_on_failure(self):
        """Test that connections are recreated when unhealthy."""
        client = GRPCClientBase("localhost:50051")
        
        # Mock an unhealthy channel
        old_channel = Mock()
        old_channel.get_state.return_value = grpc.ChannelConnectivity.TRANSIENT_FAILURE
        client.channel = old_channel
        
        # Mock new channel creation
        new_channel = Mock()
        new_channel.get_state.return_value = grpc.ChannelConnectivity.READY
        
        with patch('grpc.insecure_channel', return_value=new_channel):
            with patch.object(client, '_wait_for_ready', new_callable=AsyncMock):
                channel = await client.connect()
                
                assert channel == new_channel
                assert client.channel == new_channel
                old_channel.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_connection_attempts(self):
        """Test that concurrent connection attempts are handled safely."""
        client = GRPCClientBase("localhost:50051")
        
        mock_channel = Mock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.READY
        
        connect_call_count = 0
        original_connect = client.connect
        
        async def mock_connect():
            nonlocal connect_call_count
            connect_call_count += 1
            # Simulate some delay in connection
            await asyncio.sleep(0.1)
            return await original_connect()
        
        with patch('grpc.insecure_channel', return_value=mock_channel):
            with patch.object(client, '_wait_for_ready', new_callable=AsyncMock):
                # Start multiple concurrent connection attempts
                tasks = [mock_connect() for _ in range(5)]
                results = await asyncio.gather(*tasks)
                
                # All should return the same channel
                assert all(result == mock_channel for result in results)
                # Connection should be called for each task due to mocking
                assert connect_call_count == 5
    
    @pytest.mark.asyncio
    async def test_connection_backoff(self):
        """Test connection backoff on repeated failures."""
        client = GRPCClientBase("localhost:50051")
        
        with patch('grpc.insecure_channel', side_effect=Exception("Connection failed")):
            # First failure
            with pytest.raises(MLXError):
                await client.connect()
            initial_backoff = client._connection_backoff
            
            # Second failure should increase backoff
            with pytest.raises(MLXError):
                await client.connect()
            assert client._connection_backoff > initial_backoff
            
            # Backoff should be capped
            for _ in range(10):
                try:
                    await client.connect()
                except MLXError:
                    pass
            assert client._connection_backoff <= 60.0


class TestChannelStateHandling:
    """Test handling of different channel states."""
    
    @pytest.mark.asyncio
    async def test_wait_for_ready_immediate_success(self):
        """Test immediate success when channel is already ready."""
        client = GRPCClientBase("localhost:50051")
        mock_channel = Mock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.READY
        client.channel = mock_channel
        
        # Should complete immediately
        start_time = time.time()
        await client._wait_for_ready(timeout=5.0)
        elapsed = time.time() - start_time
        
        assert elapsed < 0.1  # Should be very fast
    
    @pytest.mark.asyncio
    async def test_wait_for_ready_eventual_success(self):
        """Test eventual success when channel becomes ready."""
        client = GRPCClientBase("localhost:50051")
        mock_channel = Mock()
        
        # Mock channel state progression: CONNECTING -> READY
        state_sequence = [
            grpc.ChannelConnectivity.CONNECTING,
            grpc.ChannelConnectivity.CONNECTING,
            grpc.ChannelConnectivity.READY
        ]
        mock_channel.get_state.side_effect = state_sequence
        client.channel = mock_channel
        
        # Should eventually succeed
        await client._wait_for_ready(timeout=1.0)
        assert mock_channel.get_state.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_wait_for_ready_timeout(self):
        """Test timeout when channel never becomes ready."""
        client = GRPCClientBase("localhost:50051")
        mock_channel = Mock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.CONNECTING
        client.channel = mock_channel
        
        with pytest.raises(MLXError) as exc_info:
            await client._wait_for_ready(timeout=0.1)
        
        assert "Channel not ready" in str(exc_info.value)
        assert exc_info.value.error_code == MLXErrorCode.CONNECTION_FAILED
    
    @pytest.mark.asyncio
    async def test_wait_for_ready_transient_failure_recovery(self):
        """Test recovery from transient failures."""
        client = GRPCClientBase("localhost:50051")
        mock_channel = Mock()
        
        # Mock channel recovering from transient failure
        state_sequence = [
            grpc.ChannelConnectivity.TRANSIENT_FAILURE,
            grpc.ChannelConnectivity.TRANSIENT_FAILURE,
            grpc.ChannelConnectivity.CONNECTING,
            grpc.ChannelConnectivity.READY
        ]
        mock_channel.get_state.side_effect = state_sequence
        client.channel = mock_channel
        
        await client._wait_for_ready(timeout=1.0)
        assert mock_channel.get_state.call_count >= 4


class TestErrorScenarios:
    """Test various error scenarios and edge cases."""
    
    @pytest.mark.asyncio
    async def test_grpc_error_mapping(self):
        """Test mapping of gRPC errors to MLX errors."""
        client = GRPCClientBase("localhost:50051")
        
        # Test different gRPC status codes
        error_mappings = [
            (grpc.StatusCode.INVALID_ARGUMENT, MLXErrorCode.INVALID_REQUEST),
            (grpc.StatusCode.NOT_FOUND, MLXErrorCode.WORKER_NOT_FOUND),
            (grpc.StatusCode.PERMISSION_DENIED, MLXErrorCode.AUTHENTICATION_FAILED),
            (grpc.StatusCode.UNAVAILABLE, MLXErrorCode.SERVICE_UNAVAILABLE),
            (grpc.StatusCode.DEADLINE_EXCEEDED, MLXErrorCode.DEADLINE_EXCEEDED),
        ]
        
        for grpc_code, expected_mlx_code in error_mappings:
            mock_error = grpc.RpcError()
            mock_error.code = Mock(return_value=grpc_code)
            mock_error.details = Mock(return_value=f"Error: {grpc_code}")
            
            mock_method = AsyncMock(side_effect=mock_error)
            request = inference_pb2.InferenceRequest()
            request.request_id = "test"
            request.prompt = "test"
            request.model = "test"
            request.max_tokens = 10
            
            with patch.object(client, 'connect', new_callable=AsyncMock):
                with pytest.raises(MLXError) as exc_info:
                    await client.call_with_retry(mock_method, request)
                
                # For non-retryable errors, should match expected code
                if grpc_code in [grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.NOT_FOUND, 
                               grpc.StatusCode.PERMISSION_DENIED]:
                    assert exc_info.value.error_code == expected_mlx_code
    
    @pytest.mark.asyncio
    async def test_network_interruption_during_call(self):
        """Test handling of network interruption during call."""
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        client = GRPCClientBase("localhost:50051", retry_config=retry_config)
        
        # Mock method that fails with network error then succeeds
        network_error = grpc.RpcError()
        network_error.code = Mock(return_value=grpc.StatusCode.UNAVAILABLE)
        network_error.details = Mock(return_value="Network unavailable")
        
        mock_method = AsyncMock()
        mock_method.side_effect = [network_error, "success"]
        
        request = inference_pb2.InferenceRequest()
        request.request_id = "test"
        request.prompt = "test"
        request.model = "test"
        request.max_tokens = 10
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = Mock()
            with patch.object(client, 'close', new_callable=AsyncMock) as mock_close:
                result = await client.call_with_retry(mock_method, request)
                
                assert result == "success"
                # Should close and reconnect after network error
                assert mock_close.call_count >= 1
                assert mock_connect.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_server_shutdown_during_call(self):
        """Test handling of server shutdown during call."""
        client = GRPCClientBase("localhost:50051")
        
        # Mock server shutdown error
        shutdown_error = grpc.RpcError()
        shutdown_error.code = Mock(return_value=grpc.StatusCode.UNAVAILABLE)
        shutdown_error.details = Mock(return_value="Server shutting down")
        
        mock_method = AsyncMock(side_effect=shutdown_error)
        
        request = inference_pb2.InferenceRequest()
        request.request_id = "test"
        request.prompt = "test"
        request.model = "test"
        request.max_tokens = 10
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = Mock()
            with patch.object(client, 'close', new_callable=AsyncMock):
                with pytest.raises(MLXError) as exc_info:
                    await client.call_with_retry(mock_method, request)
                
                assert "failed after" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test handling of memory pressure scenarios."""
        client = GRPCClientBase("localhost:50051")
        
        # Mock memory error
        memory_error = grpc.RpcError()
        memory_error.code = Mock(return_value=grpc.StatusCode.RESOURCE_EXHAUSTED)
        memory_error.details = Mock(return_value="Out of memory")
        
        mock_method = AsyncMock(side_effect=memory_error)
        
        request = inference_pb2.InferenceRequest()
        request.request_id = "test"
        request.prompt = "test"
        request.model = "test"
        request.max_tokens = 10
        
        with patch.object(client, 'connect', new_callable=AsyncMock):
            with pytest.raises(MLXError):
                await client.call_with_retry(mock_method, request)


class TestTimeoutHandling:
    """Test timeout handling in various scenarios."""
    
    @pytest.mark.asyncio
    async def test_call_timeout(self):
        """Test call timeout handling."""
        client = GRPCClientBase("localhost:50051", timeout=0.1)
        
        # Mock method that takes too long
        async def slow_method(request, metadata=None):
            await asyncio.sleep(0.2)
            return "success"
        
        request = inference_pb2.InferenceRequest()
        request.request_id = "test"
        request.prompt = "test"
        request.model = "test"
        request.max_tokens = 10
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = Mock()
            
            with pytest.raises(MLXError):
                await client.call_with_retry(slow_method, request)
    
    @pytest.mark.asyncio
    async def test_custom_timeout_override(self):
        """Test custom timeout override for specific calls."""
        client = GRPCClientBase("localhost:50051", timeout=1.0)
        
        # Mock method that takes moderate time
        async def medium_method(request, metadata=None):
            await asyncio.sleep(0.05)
            return "success"
        
        request = inference_pb2.InferenceRequest()
        request.request_id = "test"
        request.prompt = "test"
        request.model = "test"
        request.max_tokens = 10
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = Mock()
            
            # Should succeed with longer timeout
            result = await client.call_with_retry(medium_method, request, timeout=0.1)
            assert result == "success"
    
    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """Test connection establishment timeout."""
        client = GRPCClientBase("localhost:50051")
        
        # Mock channel that never becomes ready
        mock_channel = Mock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.CONNECTING
        
        with patch('grpc.insecure_channel', return_value=mock_channel):
            with pytest.raises(MLXError) as exc_info:
                await client.connect()
            
            assert "Channel not ready" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""
    
    @pytest.mark.asyncio
    async def test_empty_response_handling(self):
        """Test handling of empty responses."""
        client = GRPCClientBase("localhost:50051")
        
        mock_method = AsyncMock(return_value=None)
        
        request = inference_pb2.InferenceRequest()
        request.request_id = "test"
        request.prompt = "test"
        request.model = "test"
        request.max_tokens = 10
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = Mock()
            
            result = await client.call_with_retry(mock_method, request)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_large_message_handling(self):
        """Test handling of large messages."""
        client = GRPCClientBase(
            "localhost:50051",
            max_receive_message_length=1024,  # Small limit for testing
            max_send_message_length=1024
        )
        
        # Create a request with large prompt (exceed 32KB limit)
        request = inference_pb2.InferenceRequest()
        request.request_id = "test"
        request.prompt = "x" * (33 * 1024)  # 33KB - larger than 32KB limit
        request.model = "test"
        request.max_tokens = 10
        
        # Should fail validation due to prompt size (32KB limit)
        with pytest.raises(ValidationError):
            client.validate_request(request)
    
    @pytest.mark.asyncio
    async def test_malformed_target_address(self):
        """Test handling of malformed target addresses."""
        # Test various malformed addresses
        malformed_addresses = [
            "",
            "localhost",  # Missing port
            ":50051",     # Missing host
            "localhost:abc",  # Invalid port
            "localhost:-1",   # Negative port
            "localhost:99999",  # Port too high
        ]
        
        for address in malformed_addresses:
            client = GRPCClientBase(address)
            
            with patch('grpc.insecure_channel', side_effect=Exception("Invalid address")):
                with pytest.raises(MLXError):
                    await client.connect()
    
    @pytest.mark.asyncio
    async def test_rapid_connect_disconnect_cycles(self):
        """Test rapid connect/disconnect cycles."""
        client = GRPCClientBase("localhost:50051")
        
        mock_channel = Mock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.READY
        
        with patch('grpc.insecure_channel', return_value=mock_channel):
            with patch.object(client, '_wait_for_ready', new_callable=AsyncMock):
                # Rapid connect/disconnect cycles
                for _ in range(10):
                    await client.connect()
                    await client.close()
                
                # Should handle gracefully without errors
                assert client.channel is None
    
    @pytest.mark.asyncio
    async def test_concurrent_close_operations(self):
        """Test concurrent close operations."""
        client = GRPCClientBase("localhost:50051")
        
        mock_channel = Mock()
        client.channel = mock_channel
        
        # Start multiple concurrent close operations
        tasks = [client.close() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Should handle gracefully
        assert client.channel is None
        mock_channel.close.assert_called()  # Should be called at least once


if __name__ == "__main__":
    pytest.main([__file__, "-v"])