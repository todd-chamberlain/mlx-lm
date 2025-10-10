"""
Unit tests for FastAPI server components.

Tests OpenAI endpoint compatibility and validation, authentication and authorization flows,
streaming response handling, and error handling and edge cases as specified in task 5.5.
"""

import asyncio
import pytest
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, AsyncGenerator
from fastapi.testclient import TestClient
from fastapi import status
from httpx import AsyncClient

# Import API components
from api.main import app, get_coordinator_client
from api.models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    HealthResponse, ModelsResponse, ErrorResponse
)
from api.auth import verify_api_key, get_client_id
from api.streaming import StreamingManager, format_sse_data, format_sse_json
from api.coordinator_client import CoordinatorClient


class TestChatCompletionModels:
    """Test Pydantic models for chat completions"""
    
    def test_chat_message_validation(self):
        """Test ChatMessage model validation"""
        # Valid message
        message = ChatMessage(role="user", content="Hello, world!")
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.name is None
        
        # Message with name
        message_with_name = ChatMessage(role="assistant", content="Hi there!", name="Assistant")
        assert message_with_name.name == "Assistant"
    
    def test_chat_completion_request_validation(self):
        """Test ChatCompletionRequest model validation"""
        messages = [ChatMessage(role="user", content="Hello")]
        
        # Valid request
        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.temperature == 0.7
        assert request.max_tokens == 100
        assert request.stream is False
        assert request.id is not None  # Should be auto-generated
    
    def test_chat_completion_request_validation_errors(self):
        """Test ChatCompletionRequest validation errors"""
        # Empty messages
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            ChatCompletionRequest(model="test", messages=[])
        
        # Invalid temperature
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="test")],
                temperature=3.0  # > 2.0
            )
        
        # Invalid top_p
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="test")],
                top_p=1.5  # > 1.0
            )
    
    def test_to_prompt_conversion(self):
        """Test conversion of messages to prompt string"""
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello!"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="How are you?")
        ]
        
        request = ChatCompletionRequest(model="test", messages=messages)
        prompt = request.to_prompt()
        
        expected = (
            "System: You are a helpful assistant.\n"
            "User: Hello!\n"
            "Assistant: Hi there!\n"
            "User: How are you?\n"
            "Assistant:"
        )
        assert prompt == expected


class TestAuthentication:
    """Test authentication and authorization functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_credentials = Mock()
        self.mock_credentials.scheme = "Bearer"
        self.mock_credentials.credentials = "test-api-key-12345678901234567890"
        
        self.mock_request = Mock()
        self.mock_request.client.host = "127.0.0.1"
        self.mock_request.headers.get.return_value = "test-user-agent"
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_verify_api_key_disabled(self):
        """Test API key verification when auth is disabled"""
        # Should not raise exception when auth is disabled
        verify_api_key(None, self.mock_request)
        verify_api_key(self.mock_credentials, self.mock_request)
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'true', 'API_KEYS': 'key1,key2,test-api-key-12345678901234567890'})
    def test_verify_api_key_success(self):
        """Test successful API key verification"""
        # Should not raise exception with valid key
        verify_api_key(self.mock_credentials, self.mock_request)
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'true'})
    def test_verify_api_key_missing_credentials(self):
        """Test API key verification with missing credentials"""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(None, self.mock_request)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Missing authorization header" in exc_info.value.detail
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'true', 'API_KEYS': 'valid-key'})
    def test_verify_api_key_invalid_scheme(self):
        """Test API key verification with invalid scheme"""
        from fastapi import HTTPException
        
        self.mock_credentials.scheme = "Basic"
        
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(self.mock_credentials, self.mock_request)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid authentication scheme" in exc_info.value.detail
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'true', 'API_KEYS': 'valid-key'})
    def test_verify_api_key_invalid_key(self):
        """Test API key verification with invalid key"""
        from fastapi import HTTPException
        
        self.mock_credentials.credentials = "invalid-key"
        
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(self.mock_credentials, self.mock_request)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid API key" in exc_info.value.detail
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'true', 'API_KEYS': 'valid-key'})
    def test_verify_api_key_short_key(self):
        """Test API key verification with too short key"""
        from fastapi import HTTPException
        
        self.mock_credentials.credentials = "short"
        
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(self.mock_credentials, self.mock_request)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid API key format" in exc_info.value.detail
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'true'})
    def test_verify_api_key_no_keys_configured(self):
        """Test API key verification with no keys configured"""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(self.mock_credentials, self.mock_request)
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "API keys not configured" in exc_info.value.detail
    
    def test_get_client_id(self):
        """Test client ID extraction from credentials"""
        # With credentials
        client_id = get_client_id(self.mock_credentials)
        assert client_id == "test-api"  # First 8 characters
        
        # Without credentials
        client_id = get_client_id(None)
        assert client_id == "anonymous"


class TestStreamingManager:
    """Test streaming functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.streaming_manager = StreamingManager(timeout_seconds=10)
        self.mock_request = Mock()
        self.mock_request.client.host = "127.0.0.1"
    
    async def mock_stream_generator(self) -> AsyncGenerator[str, None]:
        """Mock stream generator for testing"""
        for i in range(3):
            yield f"data: chunk {i}\n\n"
            await asyncio.sleep(0.01)  # Small delay
    
    @pytest.mark.asyncio
    async def test_create_streaming_response(self):
        """Test streaming response creation"""
        request_id = "test-123"
        
        response = await self.streaming_manager.create_streaming_response(
            self.mock_request,
            self.mock_stream_generator(),
            request_id
        )
        
        # Check response type and headers
        assert response.media_type == "text/plain"
        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == "no-cache"
        assert response.headers["Connection"] == "keep-alive"
    
    def test_get_active_streams_count(self):
        """Test active streams counting"""
        assert self.streaming_manager.get_active_streams_count() == 0
        
        # Add mock stream
        self.streaming_manager.active_streams["test-1"] = {
            "start_time": time.time(),
            "client_ip": "127.0.0.1"
        }
        
        assert self.streaming_manager.get_active_streams_count() == 1
    
    def test_get_stream_info(self):
        """Test stream info retrieval"""
        request_id = "test-123"
        stream_info = {
            "start_time": time.time(),
            "client_ip": "127.0.0.1"
        }
        
        self.streaming_manager.active_streams[request_id] = stream_info
        
        retrieved_info = self.streaming_manager.get_stream_info(request_id)
        assert retrieved_info == stream_info
        
        # Non-existent stream
        assert self.streaming_manager.get_stream_info("nonexistent") is None
    
    def test_cleanup_expired_streams(self):
        """Test cleanup of expired streams"""
        current_time = time.time()
        
        # Add expired and active streams
        self.streaming_manager.active_streams["expired"] = {
            "start_time": current_time - 20,  # 20 seconds ago (> 10s timeout)
            "client_ip": "127.0.0.1"
        }
        self.streaming_manager.active_streams["active"] = {
            "start_time": current_time - 5,   # 5 seconds ago (< 10s timeout)
            "client_ip": "127.0.0.1"
        }
        
        self.streaming_manager.cleanup_expired_streams()
        
        assert "expired" not in self.streaming_manager.active_streams
        assert "active" in self.streaming_manager.active_streams
    
    def test_format_sse_data(self):
        """Test SSE data formatting"""
        # Simple data
        formatted = format_sse_data("Hello, world!")
        assert formatted == "data: Hello, world!\n\n"
        
        # Multi-line data
        formatted = format_sse_data("Line 1\nLine 2")
        expected = "data: Line 1\ndata: Line 2\n\n"
        assert formatted == expected
        
        # With event type
        formatted = format_sse_data("Hello", event="greeting")
        expected = "event: greeting\ndata: Hello\n\n"
        assert formatted == expected
    
    def test_format_sse_json(self):
        """Test SSE JSON formatting"""
        data = {"message": "Hello", "type": "greeting"}
        formatted = format_sse_json(data)
        
        expected_json = json.dumps(data)
        expected = f"data: {expected_json}\n\n"
        assert formatted == expected
        
        # With event type
        formatted = format_sse_json(data, event="message")
        expected = f"event: message\ndata: {expected_json}\n\n"
        assert formatted == expected


class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
        
        # Mock coordinator client
        self.mock_coordinator = AsyncMock(spec=CoordinatorClient)
        
        # Override dependency
        app.dependency_overrides[get_coordinator_client] = lambda: self.mock_coordinator
    
    def teardown_method(self):
        """Clean up after tests"""
        app.dependency_overrides.clear()
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_chat_completions_success(self):
        """Test successful chat completion request"""
        # Mock coordinator response
        mock_response = ChatCompletionResponse(
            id="test-123",
            model="test-model",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        )
        self.mock_coordinator.route_inference.return_value = mock_response
        
        # Make request
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-123"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "Hello!"
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_chat_completions_validation_error(self):
        """Test chat completion with validation error"""
        # Invalid request (empty messages)
        request_data = {
            "model": "test-model",
            "messages": [],  # Empty messages should fail validation
            "temperature": 0.7
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_chat_completions_coordinator_error(self):
        """Test chat completion with coordinator error"""
        # Mock coordinator error
        self.mock_coordinator.route_inference.side_effect = Exception("Coordinator error")
        
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "Inference failed" in data["detail"]
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'true', 'API_KEYS': 'valid-key-12345678901234567890'})
    def test_chat_completions_with_auth_success(self):
        """Test chat completion with successful authentication"""
        # Mock coordinator response
        mock_response = ChatCompletionResponse(
            id="test-123",
            model="test-model",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        )
        self.mock_coordinator.route_inference.return_value = mock_response
        
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        headers = {"Authorization": "Bearer valid-key-12345678901234567890"}
        response = self.client.post("/v1/chat/completions", json=request_data, headers=headers)
        
        assert response.status_code == 200
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'true', 'API_KEYS': 'valid-key'})
    def test_chat_completions_with_auth_failure(self):
        """Test chat completion with authentication failure"""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        # No authorization header
        response = self.client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 401
        
        # Invalid key
        headers = {"Authorization": "Bearer invalid-key"}
        response = self.client.post("/v1/chat/completions", json=request_data, headers=headers)
        assert response.status_code == 401
    
    def test_health_endpoint_success(self):
        """Test health endpoint with healthy system"""
        # Mock health response
        mock_health = Mock()
        mock_health.healthy_workers = 3
        mock_health.total_workers = 3
        mock_health.active_requests = 5
        self.mock_coordinator.get_health.return_value = mock_health
        
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["total_workers"] == 3
        assert data["healthy_workers"] == 3
        assert data["active_requests"] == 5
    
    def test_health_endpoint_unhealthy(self):
        """Test health endpoint with unhealthy system"""
        # Mock health response with no healthy workers
        mock_health = Mock()
        mock_health.healthy_workers = 0
        mock_health.total_workers = 3
        mock_health.active_requests = 0
        self.mock_coordinator.get_health.return_value = mock_health
        
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["healthy_workers"] == 0
    
    def test_health_endpoint_coordinator_error(self):
        """Test health endpoint with coordinator error"""
        # Mock coordinator error
        self.mock_coordinator.get_health.side_effect = Exception("Connection failed")
        
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["total_workers"] == 0
        assert "Connection failed" in data["error"]
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_models_endpoint_success(self):
        """Test models endpoint success"""
        # Mock available models
        self.mock_coordinator.get_available_models.return_value = ["model1", "model2"]
        
        response = self.client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "model1"
        assert data["data"][1]["id"] == "model2"
        assert all(model["object"] == "model" for model in data["data"])
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_models_endpoint_coordinator_error(self):
        """Test models endpoint with coordinator error"""
        # Mock coordinator error
        self.mock_coordinator.get_available_models.side_effect = Exception("Failed to get models")
        
        response = self.client.get("/models")
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to list models" in data["detail"]


class TestStreamingEndpoints:
    """Test streaming endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
        
        # Mock coordinator client
        self.mock_coordinator = AsyncMock(spec=CoordinatorClient)
        
        # Override dependency
        app.dependency_overrides[get_coordinator_client] = lambda: self.mock_coordinator
    
    def teardown_method(self):
        """Clean up after tests"""
        app.dependency_overrides.clear()
    
    async def mock_stream_chunks(self):
        """Mock streaming chunks from coordinator"""
        chunks = [
            {"id": "test-123", "choices": [{"index": 0, "delta": {"content": "Hello"}}]},
            {"id": "test-123", "choices": [{"index": 0, "delta": {"content": " world"}}]},
            {"id": "test-123", "choices": [{"index": 0, "delta": {"content": "!"}}]},
        ]
        
        for chunk in chunks:
            yield Mock(**chunk)
            await asyncio.sleep(0.01)
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_streaming_chat_completion(self):
        """Test streaming chat completion"""
        # Mock streaming response
        self.mock_coordinator.stream_inference.return_value = self.mock_stream_chunks()
        
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        # Check that response contains SSE data
        content = response.text
        assert "data:" in content
        assert "[DONE]" in content


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
        
        # Mock coordinator client that's not initialized
        app.dependency_overrides[get_coordinator_client] = lambda: None
    
    def teardown_method(self):
        """Clean up after tests"""
        app.dependency_overrides.clear()
    
    def test_coordinator_not_initialized(self):
        """Test behavior when coordinator client is not initialized"""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 503
        assert "Coordinator client not initialized" in response.json()["detail"]
    
    def test_invalid_json_request(self):
        """Test handling of invalid JSON requests"""
        response = self.client.post(
            "/v1/chat/completions",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_missing_required_fields(self):
        """Test handling of requests with missing required fields"""
        # Missing model field
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("model" in str(error).lower() for error in error_detail)
    
    def test_invalid_parameter_values(self):
        """Test handling of invalid parameter values"""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 5.0,  # Invalid: > 2.0
            "max_tokens": -1     # Invalid: < 1
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 422
    
    def test_nonexistent_endpoint(self):
        """Test handling of requests to non-existent endpoints"""
        response = self.client.get("/nonexistent")
        
        assert response.status_code == 404


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
        
        # Mock coordinator client
        self.mock_coordinator = AsyncMock(spec=CoordinatorClient)
        app.dependency_overrides[get_coordinator_client] = lambda: self.mock_coordinator
    
    def teardown_method(self):
        """Clean up after tests"""
        app.dependency_overrides.clear()
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_rate_limiting_models_endpoint(self):
        """Test rate limiting on models endpoint"""
        # Mock available models
        self.mock_coordinator.get_available_models.return_value = ["model1"]
        
        # Make multiple requests rapidly
        responses = []
        for _ in range(5):  # Should be within limit
            response = self.client.get("/models")
            responses.append(response)
        
        # All should succeed (assuming rate limit is higher than 5)
        assert all(r.status_code == 200 for r in responses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])