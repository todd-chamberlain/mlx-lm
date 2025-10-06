"""
Unit tests for gRPC communication layer.

Tests message validation, serialization, connection handling, error scenarios,
retry logic, and timeout handling as specified in task 2.3.
"""

import asyncio
import pytest
import grpc
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import time

from proto import inference_pb2
from proto.grpc_base import (
    GRPCServerBase, 
    GRPCClientBase, 
    RetryConfig, 
    MessageSerializer,
    HealthChecker
)
from proto.validation import MessageValidator, ValidationError
from proto.errors import MLXError, MLXErrorCode


class TestMessageValidationAndSerialization:
    """Test message validation and serialization functionality."""
    
    def test_inference_request_validation_success(self):
        """Test successful validation of InferenceRequest."""
        request = inference_pb2.InferenceRequest()
        request.request_id = "test-123"
        request.prompt = "Hello, world!"
        request.max_tokens = 100
        request.temperature = 0.7
        request.top_p = 0.9
        request.model = "test-model"
        
        errors = MessageValidator.validate_inference_request(request)
        assert errors == []
    
    def test_inference_request_validation_failures(self):
        """Test validation failures for InferenceRequest."""
        # Empty request
        request = inference_pb2.InferenceRequest()
        errors = MessageValidator.validate_inference_request(request)
        
        assert len(errors) >= 3
        assert any("request_id is required" in error for error in errors)
        assert any("prompt is required" in error for error in errors)
        assert any("model is required" in error for error in errors)
        
        # Invalid values
        request.request_id = "test"
        request.prompt = "test"
        request.model = "test"
        request.max_tokens = -1
        request.temperature = 3.0
        request.top_p = 1.5
        
        errors = MessageValidator.validate_inference_request(request)
        assert any("max_tokens must be greater than 0" in error for error in errors)
        assert any("temperature must be between 0.0 and 2.0" in error for error in errors)
        assert any("top_p must be between 0.0 and 1.0" in error for error in errors)
    
    def test_worker_registration_validation_success(self):
        """Test successful validation of WorkerRegistration."""
        registration = inference_pb2.WorkerRegistration()
        registration.worker_id = "worker-1"
        registration.address = "localhost:8080"
        registration.max_concurrent = 4
        registration.supported_models.extend(["model-1"])
        
        device_info = registration.device_info
        device_info.model = "Mac Studio"
        device_info.memory_gb = 64
        device_info.gpu_cores = 32
        
        errors = MessageValidator.validate_worker_registration(registration)
        assert errors == []
    
    def test_worker_registration_validation_failures(self):
        """Test validation failures for WorkerRegistration."""
        registration = inference_pb2.WorkerRegistration()
        errors = MessageValidator.validate_worker_registration(registration)
        
        assert len(errors) >= 4
        assert any("worker_id is required" in error for error in errors)
        assert any("address is required" in error for error in errors)
        assert any("device model is required" in error for error in errors)
        assert any("at least one supported model" in error for error in errors)
    
    def test_message_serialization_success(self):
        """Test successful message serialization."""
        request = inference_pb2.InferenceRequest()
        request.request_id = "test-123"
        request.prompt = "Hello, world!"
        request.max_tokens = 100
        
        # Test serialization
        serialized = MessageSerializer.serialize_message(request)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Test deserialization
        deserialized = MessageSerializer.deserialize_message(
            serialized, 
            inference_pb2.InferenceRequest
        )
        assert deserialized.request_id == request.request_id
        assert deserialized.prompt == request.prompt
        assert deserialized.max_tokens == request.max_tokens
    
    def test_message_to_dict_conversion(self):
        """Test message to dictionary conversion."""
        request = inference_pb2.InferenceRequest()
        request.request_id = "test-123"
        request.prompt = "Hello, world!"
        request.max_tokens = 100
        
        result = MessageSerializer.message_to_dict(request)
        assert isinstance(result, dict)
        assert result.get("requestId") == "test-123"
        assert result.get("prompt") == "Hello, world!"
        assert result.get("maxTokens") == 100
    
    def test_dict_to_message_conversion(self):
        """Test dictionary to message conversion."""
        data = {
            "requestId": "test-123",
            "prompt": "Hello, world!",
            "maxTokens": 100,
            "temperature": 0.7,
            "model": "test-model"
        }
        
        message = MessageSerializer.dict_to_message(
            data, 
            inference_pb2.InferenceRequest
        )
        assert message.request_id == "test-123"
        assert message.prompt == "Hello, world!"
        assert message.max_tokens == 100
        assert abs(message.temperature - 0.7) < 1e-6
        assert message.model == "test-model"


class TestRetryConfig:
    """Test retry configuration and delay calculation."""
    
    def test_default_retry_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
    
    def test_custom_retry_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False
        )
        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
    
    def test_delay_calculation_without_jitter(self):
        """Test delay calculation without jitter."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=60.0,
            jitter=False
        )
        
        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 8.0
        
        # Test max delay cap
        assert config.get_delay(10) == 60.0
    
    def test_delay_calculation_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(
            base_delay=2.0,
            exponential_base=2.0,
            jitter=True
        )
        
        # With jitter, delay should be between 50% and 100% of calculated value
        delay = config.get_delay(1)  # Base calculation: 4.0
        assert 2.0 <= delay <= 4.0


class MockGRPCServer(GRPCServerBase):
    """Mock gRPC server for testing."""
    
    def __init__(self, port: int = 50051):
        super().__init__(port)
        self.servicers_added = False
    
    def add_servicers(self, server: grpc.Server) -> None:
        """Mock implementation of add_servicers."""
        self.servicers_added = True


class TestGRPCServerBase:
    """Test gRPC server base functionality."""
    
    def test_server_initialization(self):
        """Test server initialization with default parameters."""
        server = MockGRPCServer(port=50051)
        assert server.port == 50051
        assert server.max_workers == 10
        assert server.server is None
        assert len(server.options) > 0
    
    def test_server_initialization_custom_params(self):
        """Test server initialization with custom parameters."""
        server = MockGRPCServer(port=8080)
        server.max_workers = 20
        server.max_receive_message_length = 8 * 1024 * 1024
        
        assert server.port == 8080
        assert server.max_workers == 20
        assert server.max_receive_message_length == 8 * 1024 * 1024
    
    @patch('grpc.server')
    def test_create_server(self, mock_grpc_server):
        """Test server creation."""
        mock_server_instance = Mock()
        mock_grpc_server.return_value = mock_server_instance
        
        server = MockGRPCServer()
        created_server = server.create_server()
        
        assert created_server == mock_server_instance
        assert server.servicers_added is True
        mock_server_instance.add_insecure_port.assert_called_once()
    
    def test_request_validation_success(self):
        """Test successful request validation."""
        server = MockGRPCServer()
        
        request = inference_pb2.InferenceRequest()
        request.request_id = "test-123"
        request.prompt = "Hello, world!"
        request.max_tokens = 100
        request.model = "test-model"
        
        # Should not raise exception
        server.validate_request(request)
    
    def test_request_validation_failure(self):
        """Test request validation failure."""
        server = MockGRPCServer()
        
        request = inference_pb2.InferenceRequest()
        # Empty request should fail validation
        
        with pytest.raises(ValidationError):
            server.validate_request(request)
    
    @patch('grpc.ServicerContext')
    def test_error_handling_mlx_error(self, mock_context):
        """Test error handling for MLXError."""
        server = MockGRPCServer()
        error = MLXError(
            message="Test error",
            error_code=MLXErrorCode.INVALID_REQUEST,
            grpc_status=grpc.StatusCode.INVALID_ARGUMENT
        )
        
        server.handle_error(mock_context, error)
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
        mock_context.set_details.assert_called_once_with("Test error")
    
    @patch('grpc.ServicerContext')
    def test_error_handling_validation_error(self, mock_context):
        """Test error handling for ValidationError."""
        server = MockGRPCServer()
        error = ValidationError("Validation failed")
        
        server.handle_error(mock_context, error)
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
        mock_context.set_details.assert_called_once_with("Validation failed")
    
    @patch('grpc.ServicerContext')
    def test_error_handling_generic_error(self, mock_context):
        """Test error handling for generic exceptions."""
        server = MockGRPCServer()
        error = Exception("Generic error")
        
        server.handle_error(mock_context, error)
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
        mock_context.set_details.assert_called_once()


class TestGRPCClientBase:
    """Test gRPC client base functionality."""
    
    def test_client_initialization(self):
        """Test client initialization with default parameters."""
        client = GRPCClientBase("localhost:50051")
        assert client.target == "localhost:50051"
        assert client.timeout == 30.0
        assert client.channel is None
        assert isinstance(client.retry_config, RetryConfig)
    
    def test_client_initialization_custom_params(self):
        """Test client initialization with custom parameters."""
        retry_config = RetryConfig(max_attempts=5)
        client = GRPCClientBase(
            target="example.com:8080",
            retry_config=retry_config,
            timeout=60.0
        )
        assert client.target == "example.com:8080"
        assert client.timeout == 60.0
        assert client.retry_config == retry_config
    
    @patch('grpc.insecure_channel')
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_channel):
        """Test successful connection."""
        mock_channel_instance = Mock()
        mock_channel_instance.get_state.return_value = grpc.ChannelConnectivity.READY
        mock_channel.return_value = mock_channel_instance
        
        client = GRPCClientBase("localhost:50051")
        
        with patch.object(client, '_wait_for_ready', new_callable=AsyncMock):
            channel = await client.connect()
            assert channel == mock_channel_instance
            assert client.channel == mock_channel_instance
    
    @patch('grpc.insecure_channel')
    @pytest.mark.asyncio
    async def test_connect_failure(self, mock_channel):
        """Test connection failure."""
        mock_channel.side_effect = Exception("Connection failed")
        
        client = GRPCClientBase("localhost:50051")
        
        with pytest.raises(MLXError) as exc_info:
            await client.connect()
        
        assert "Failed to connect" in str(exc_info.value)
        assert exc_info.value.error_code == MLXErrorCode.CONNECTION_FAILED
    
    def test_is_channel_ready_no_channel(self):
        """Test channel ready check with no channel."""
        client = GRPCClientBase("localhost:50051")
        assert client._is_channel_ready() is False
    
    def test_is_channel_ready_with_ready_channel(self):
        """Test channel ready check with ready channel."""
        client = GRPCClientBase("localhost:50051")
        client.channel = Mock()
        client.channel.get_state.return_value = grpc.ChannelConnectivity.READY
        
        assert client._is_channel_ready() is True
    
    def test_is_channel_ready_with_not_ready_channel(self):
        """Test channel ready check with not ready channel."""
        client = GRPCClientBase("localhost:50051")
        client.channel = Mock()
        client.channel.get_state.return_value = grpc.ChannelConnectivity.CONNECTING
        
        assert client._is_channel_ready() is False
    
    @pytest.mark.asyncio
    async def test_close_with_channel(self):
        """Test closing client with active channel."""
        client = GRPCClientBase("localhost:50051")
        mock_channel = Mock()
        client.channel = mock_channel
        
        await client.close()
        
        mock_channel.close.assert_called_once()
        assert client.channel is None
    
    @pytest.mark.asyncio
    async def test_close_without_channel(self):
        """Test closing client without active channel."""
        client = GRPCClientBase("localhost:50051")
        
        # Should not raise exception
        await client.close()
    
    def test_validate_request_success(self):
        """Test successful request validation."""
        client = GRPCClientBase("localhost:50051")
        
        request = inference_pb2.InferenceRequest()
        request.request_id = "test-123"
        request.prompt = "Hello, world!"
        request.max_tokens = 100
        request.model = "test-model"
        
        # Should not raise exception
        client.validate_request(request)
    
    def test_validate_request_failure(self):
        """Test request validation failure."""
        client = GRPCClientBase("localhost:50051")
        
        request = inference_pb2.InferenceRequest()
        # Empty request should fail validation
        
        with pytest.raises(ValidationError):
            client.validate_request(request)


class TestGRPCClientRetryLogic:
    """Test gRPC client retry logic and error handling."""
    
    @pytest.mark.asyncio
    async def test_call_with_retry_success_first_attempt(self):
        """Test successful call on first attempt."""
        client = GRPCClientBase("localhost:50051")
        
        # Mock successful method call
        mock_method = AsyncMock(return_value="success")
        request = inference_pb2.InferenceRequest()
        request.request_id = "test"
        request.prompt = "test"
        request.model = "test"
        request.max_tokens = 10
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = Mock()
            
            result = await client.call_with_retry(mock_method, request)
            
            assert result == "success"
            assert mock_method.call_count == 1
            assert mock_connect.call_count == 1
    
    @pytest.mark.asyncio
    async def test_call_with_retry_success_after_retries(self):
        """Test successful call after retries."""
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        client = GRPCClientBase("localhost:50051", retry_config=retry_config)
        
        # Create mock RPC errors
        error1 = grpc.RpcError()
        error1.code = Mock(return_value=grpc.StatusCode.UNAVAILABLE)
        error1.details = Mock(return_value="Service unavailable")
        
        error2 = grpc.RpcError()
        error2.code = Mock(return_value=grpc.StatusCode.UNAVAILABLE)
        error2.details = Mock(return_value="Service unavailable")
        
        # Mock method that fails twice then succeeds
        mock_method = AsyncMock()
        mock_method.side_effect = [
            error1,     # First call fails
            error2,     # Second call fails
            "success"   # Third call succeeds
        ]
        
        request = inference_pb2.InferenceRequest()
        request.request_id = "test"
        request.prompt = "test"
        request.model = "test"
        request.max_tokens = 10
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = Mock()
            with patch.object(client, 'close', new_callable=AsyncMock):
                result = await client.call_with_retry(mock_method, request)
                
                assert result == "success"
                assert mock_method.call_count == 3
    
    @pytest.mark.asyncio
    async def test_call_with_retry_non_retryable_error(self):
        """Test non-retryable error handling."""
        client = GRPCClientBase("localhost:50051")
        
        # Mock method that returns non-retryable error
        mock_error = grpc.RpcError()
        mock_error.code = Mock(return_value=grpc.StatusCode.INVALID_ARGUMENT)
        mock_error.details = Mock(return_value="Invalid request")
        
        mock_method = AsyncMock(side_effect=mock_error)
        
        request = inference_pb2.InferenceRequest()
        request.request_id = "test"
        request.prompt = "test"
        request.model = "test"
        request.max_tokens = 10
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = Mock()
            
            with pytest.raises(MLXError) as exc_info:
                await client.call_with_retry(mock_method, request)
            
            assert "Invalid request" in str(exc_info.value) or "gRPC call failed" in str(exc_info.value)
            assert mock_method.call_count == 1  # Should not retry
    
    @pytest.mark.asyncio
    async def test_call_with_retry_max_attempts_exceeded(self):
        """Test behavior when max retry attempts are exceeded."""
        retry_config = RetryConfig(max_attempts=2, base_delay=0.01)
        client = GRPCClientBase("localhost:50051", retry_config=retry_config)
        
        # Mock method that always fails with retryable error
        mock_error = grpc.RpcError()
        mock_error.code = Mock(return_value=grpc.StatusCode.UNAVAILABLE)
        mock_error.details = Mock(return_value="Service unavailable")
        
        mock_method = AsyncMock(side_effect=mock_error)
        
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
                
                assert "failed after 2 attempts" in str(exc_info.value)
                assert mock_method.call_count == 2
    
    @pytest.mark.asyncio
    async def test_call_with_retry_timeout_handling(self):
        """Test timeout handling in retry logic."""
        client = GRPCClientBase("localhost:50051")
        
        # Mock method that times out
        mock_method = AsyncMock(side_effect=asyncio.TimeoutError())
        
        request = inference_pb2.InferenceRequest()
        request.request_id = "test"
        request.prompt = "test"
        request.model = "test"
        request.max_tokens = 10
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = Mock()
            
            with pytest.raises(MLXError):
                await client.call_with_retry(mock_method, request, timeout=0.01)


class TestHealthChecker:
    """Test health checking functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        client = GRPCClientBase("localhost:50051")
        health_checker = HealthChecker(client)
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = Mock()
            
            result = await health_checker.check_health()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure."""
        client = GRPCClientBase("localhost:50051")
        health_checker = HealthChecker(client)
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            result = await health_checker.check_health()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_wait_for_healthy_success(self):
        """Test waiting for service to become healthy."""
        client = GRPCClientBase("localhost:50051")
        health_checker = HealthChecker(client)
        
        with patch.object(health_checker, 'check_health', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True
            
            result = await health_checker.wait_for_healthy(timeout=1.0, check_interval=0.1)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_wait_for_healthy_timeout(self):
        """Test timeout when waiting for service to become healthy."""
        client = GRPCClientBase("localhost:50051")
        health_checker = HealthChecker(client)
        
        with patch.object(health_checker, 'check_health', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False
            
            result = await health_checker.wait_for_healthy(timeout=0.2, check_interval=0.1)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_wait_for_healthy_eventual_success(self):
        """Test eventual success when waiting for service to become healthy."""
        client = GRPCClientBase("localhost:50051")
        health_checker = HealthChecker(client)
        
        # Mock health check to fail first few times, then succeed
        call_count = 0
        async def mock_check_health():
            nonlocal call_count
            call_count += 1
            return call_count >= 3
        
        with patch.object(health_checker, 'check_health', side_effect=mock_check_health):
            result = await health_checker.wait_for_healthy(timeout=1.0, check_interval=0.1)
            assert result is True
            assert call_count >= 3


class TestAsyncContextManager:
    """Test async context manager functionality."""
    
    @pytest.mark.asyncio
    async def test_context_manager_success(self):
        """Test successful context manager usage."""
        client = GRPCClientBase("localhost:50051")
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            with patch.object(client, 'close', new_callable=AsyncMock) as mock_close:
                mock_connect.return_value = Mock()
                
                async with client as c:
                    assert c == client
                
                mock_connect.assert_called_once()
                mock_close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager_exception(self):
        """Test context manager with exception."""
        client = GRPCClientBase("localhost:50051")
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            with patch.object(client, 'close', new_callable=AsyncMock) as mock_close:
                mock_connect.return_value = Mock()
                
                with pytest.raises(ValueError):
                    async with client:
                        raise ValueError("Test exception")
                
                mock_connect.assert_called_once()
                mock_close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])