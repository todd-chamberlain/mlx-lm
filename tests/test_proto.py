"""
Tests for protocol buffer generation and validation.
"""

import pytest
from proto import inference_pb2
from proto.validation import MessageValidator, ValidationError


def test_inference_request_creation():
    """Test that InferenceRequest can be created and populated."""
    request = inference_pb2.InferenceRequest()
    request.request_id = "test-123"
    request.prompt = "Hello, world!"
    request.max_tokens = 100
    request.temperature = 0.7
    request.top_p = 0.9
    request.model = "test-model"
    
    assert request.request_id == "test-123"
    assert request.prompt == "Hello, world!"
    assert request.max_tokens == 100
    assert abs(request.temperature - 0.7) < 1e-6
    assert abs(request.top_p - 0.9) < 1e-6
    assert request.model == "test-model"


def test_worker_registration_creation():
    """Test that WorkerRegistration can be created and populated."""
    registration = inference_pb2.WorkerRegistration()
    registration.worker_id = "worker-1"
    registration.address = "localhost:8080"
    registration.max_concurrent = 4
    
    # Add device info
    device_info = registration.device_info
    device_info.model = "Mac Studio"
    device_info.memory_gb = 64
    device_info.gpu_cores = 32
    
    # Add supported models
    registration.supported_models.extend(["model-1", "model-2"])
    
    assert registration.worker_id == "worker-1"
    assert registration.address == "localhost:8080"
    assert registration.max_concurrent == 4
    assert registration.device_info.model == "Mac Studio"
    assert registration.device_info.memory_gb == 64
    assert registration.device_info.gpu_cores == 32
    assert list(registration.supported_models) == ["model-1", "model-2"]


def test_inference_request_validation_valid():
    """Test validation of valid InferenceRequest."""
    request = inference_pb2.InferenceRequest()
    request.request_id = "test-123"
    request.prompt = "Hello, world!"
    request.max_tokens = 100
    request.temperature = 0.7
    request.top_p = 0.9
    request.model = "test-model"
    
    errors = MessageValidator.validate_inference_request(request)
    assert errors == []


def test_inference_request_validation_invalid():
    """Test validation of invalid InferenceRequest."""
    request = inference_pb2.InferenceRequest()
    # Leave required fields empty
    
    errors = MessageValidator.validate_inference_request(request)
    assert len(errors) > 0
    assert any("request_id is required" in error for error in errors)
    assert any("prompt is required" in error for error in errors)
    assert any("model is required" in error for error in errors)


def test_worker_registration_validation_valid():
    """Test validation of valid WorkerRegistration."""
    registration = inference_pb2.WorkerRegistration()
    registration.worker_id = "worker-1"
    registration.address = "localhost:8080"
    registration.max_concurrent = 4
    registration.supported_models.extend(["model-1"])
    
    # Add device info
    device_info = registration.device_info
    device_info.model = "Mac Studio"
    device_info.memory_gb = 64
    device_info.gpu_cores = 32
    
    errors = MessageValidator.validate_worker_registration(registration)
    assert errors == []


def test_worker_registration_validation_invalid():
    """Test validation of invalid WorkerRegistration."""
    registration = inference_pb2.WorkerRegistration()
    # Leave required fields empty
    
    errors = MessageValidator.validate_worker_registration(registration)
    assert len(errors) > 0
    assert any("worker_id is required" in error for error in errors)
    assert any("address is required" in error for error in errors)


def test_device_info_validation():
    """Test DeviceInfo validation."""
    device_info = inference_pb2.DeviceInfo()
    device_info.model = "Mac Studio"
    device_info.memory_gb = 64
    device_info.gpu_cores = 32
    
    errors = MessageValidator.validate_device_info(device_info)
    assert errors == []
    
    # Test invalid device info
    invalid_device = inference_pb2.DeviceInfo()
    invalid_device.memory_gb = -1  # Invalid
    
    errors = MessageValidator.validate_device_info(invalid_device)
    assert len(errors) > 0
    assert any("memory_gb must be greater than 0" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__])