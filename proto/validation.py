"""
Message validation utilities for gRPC protocol buffers.
"""

import re
from typing import List, Optional, Union
from google.protobuf.message import Message

# Import generated protobuf messages (will be available after running generate_proto.py)
try:
    from . import inference_pb2
except ImportError:
    # Fallback for when proto files haven't been generated yet
    inference_pb2 = None

# Import ValidationError from errors module
from .errors import ValidationError


class MessageValidator:
    """Validates gRPC protocol buffer messages."""
    
    @staticmethod
    def validate_inference_request(request) -> List[str]:
        """
        Validate InferenceRequest message.
        
        Args:
            request: InferenceRequest protobuf message
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate request_id
        if not request.request_id or not request.request_id.strip():
            errors.append("request_id is required and cannot be empty")
        elif len(request.request_id) > 128:
            errors.append("request_id cannot exceed 128 characters")
        
        # Validate prompt
        if not request.prompt or not request.prompt.strip():
            errors.append("prompt is required and cannot be empty")
        elif len(request.prompt) > 32768:  # 32KB limit
            errors.append("prompt cannot exceed 32KB")
        
        # Validate max_tokens
        if request.max_tokens <= 0:
            errors.append("max_tokens must be greater than 0")
        elif request.max_tokens > 4096:
            errors.append("max_tokens cannot exceed 4096")
        
        # Validate temperature
        if request.temperature < 0.0 or request.temperature > 2.0:
            errors.append("temperature must be between 0.0 and 2.0")
        
        # Validate top_p
        if request.top_p < 0.0 or request.top_p > 1.0:
            errors.append("top_p must be between 0.0 and 1.0")
        
        # Validate model
        if not request.model or not request.model.strip():
            errors.append("model is required and cannot be empty")
        elif not re.match(r'^[a-zA-Z0-9_/.:-]+$', request.model):
            errors.append("model name can only contain alphanumeric characters, hyphens, underscores, periods, colons, and slashes")
        
        return errors
    
    @staticmethod
    def validate_worker_registration(registration) -> List[str]:
        """
        Validate WorkerRegistration message.
        
        Args:
            registration: WorkerRegistration protobuf message
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate worker_id
        if not registration.worker_id or not registration.worker_id.strip():
            errors.append("worker_id is required and cannot be empty")
        elif not re.match(r'^[a-zA-Z0-9_-]+$', registration.worker_id):
            errors.append("worker_id can only contain alphanumeric characters, hyphens, and underscores")
        elif len(registration.worker_id) > 64:
            errors.append("worker_id cannot exceed 64 characters")
        
        # Validate address
        if not registration.address or not registration.address.strip():
            errors.append("address is required and cannot be empty")
        elif not MessageValidator._is_valid_address(registration.address):
            errors.append("address must be in format 'host:port'")
        
        # Validate max_concurrent
        if registration.max_concurrent <= 0:
            errors.append("max_concurrent must be greater than 0")
        elif registration.max_concurrent > 64:
            errors.append("max_concurrent cannot exceed 64")
        
        # Validate device_info
        if registration.device_info:
            device_errors = MessageValidator.validate_device_info(registration.device_info)
            errors.extend(device_errors)
        else:
            errors.append("device_info is required")
        
        # Validate supported_models (optional - workers can start with no models for dynamic loading)
        if registration.supported_models:
            for model in registration.supported_models:
                if not model or not model.strip():
                    errors.append("supported model names cannot be empty")
                elif not re.match(r'^[a-zA-Z0-9_/.:-]+$', model):
                    errors.append(f"invalid model name '{model}': can only contain alphanumeric characters, hyphens, underscores, periods, colons, and slashes")
        
        return errors
    
    @staticmethod
    def validate_device_info(device_info) -> List[str]:
        """
        Validate DeviceInfo message.
        
        Args:
            device_info: DeviceInfo protobuf message
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate model
        if not device_info.model or not device_info.model.strip():
            errors.append("device model is required")
        elif len(device_info.model) > 128:
            errors.append("device model cannot exceed 128 characters")
        
        # Validate memory_gb
        if device_info.memory_gb <= 0:
            errors.append("memory_gb must be greater than 0")
        elif device_info.memory_gb > 1024:  # 1TB limit
            errors.append("memory_gb cannot exceed 1024")
        
        # Validate gpu_cores
        if device_info.gpu_cores < 0:
            errors.append("gpu_cores cannot be negative")
        elif device_info.gpu_cores > 128:  # Reasonable upper limit
            errors.append("gpu_cores cannot exceed 128")
        
        # Validate os_version (optional)
        if device_info.os_version and len(device_info.os_version) > 64:
            errors.append("os_version cannot exceed 64 characters")
        
        # Validate mlx_version (optional)
        if device_info.mlx_version and len(device_info.mlx_version) > 32:
            errors.append("mlx_version cannot exceed 32 characters")
        
        return errors
    
    @staticmethod
    def validate_heartbeat_request(heartbeat) -> List[str]:
        """
        Validate HeartbeatRequest message.
        
        Args:
            heartbeat: HeartbeatRequest protobuf message
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate worker_id
        if not heartbeat.worker_id or not heartbeat.worker_id.strip():
            errors.append("worker_id is required and cannot be empty")
        
        # Validate status
        if heartbeat.status:
            status_errors = MessageValidator.validate_worker_status(heartbeat.status)
            errors.extend(status_errors)
        
        # Validate metrics
        if heartbeat.metrics:
            metrics_errors = MessageValidator.validate_worker_metrics(heartbeat.metrics)
            errors.extend(metrics_errors)
        
        return errors
    
    @staticmethod
    def validate_worker_status(status) -> List[str]:
        """
        Validate WorkerStatus message.
        
        Args:
            status: WorkerStatus protobuf message
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate status
        valid_statuses = {"healthy", "unhealthy", "draining"}
        if status.status not in valid_statuses:
            errors.append(f"status must be one of: {', '.join(valid_statuses)}")
        
        # Validate current_load
        if status.current_load < 0:
            errors.append("current_load cannot be negative")
        
        # Validate memory_usage_percent
        if status.memory_usage_percent < 0.0 or status.memory_usage_percent > 100.0:
            errors.append("memory_usage_percent must be between 0.0 and 100.0")
        
        # Validate uptime_seconds
        if status.uptime_seconds < 0:
            errors.append("uptime_seconds cannot be negative")
        
        return errors
    
    @staticmethod
    def validate_worker_metrics(metrics) -> List[str]:
        """
        Validate WorkerMetrics message.
        
        Args:
            metrics: WorkerMetrics protobuf message
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate avg_latency_ms
        if metrics.avg_latency_ms < 0.0:
            errors.append("avg_latency_ms cannot be negative")
        
        # Validate tokens_per_second
        if metrics.tokens_per_second < 0.0:
            errors.append("tokens_per_second cannot be negative")
        
        # Validate request counts
        if metrics.total_requests < 0:
            errors.append("total_requests cannot be negative")
        if metrics.failed_requests < 0:
            errors.append("failed_requests cannot be negative")
        if metrics.active_requests < 0:
            errors.append("active_requests cannot be negative")
        
        # Validate failed_requests <= total_requests
        if metrics.failed_requests > metrics.total_requests:
            errors.append("failed_requests cannot exceed total_requests")
        
        # Validate memory_usage_gb
        if metrics.memory_usage_gb < 0.0:
            errors.append("memory_usage_gb cannot be negative")
        
        return errors
    
    @staticmethod
    def _is_valid_address(address: str) -> bool:
        """
        Check if address is in valid 'host:port' format.
        
        Args:
            address: Address string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            host, port_str = address.rsplit(':', 1)
            port = int(port_str)
            return (
                len(host) > 0 and 
                1 <= port <= 65535 and
                not host.startswith(':') and
                not host.endswith(':')
            )
        except (ValueError, AttributeError):
            return False


def validate_message(message: Message) -> List[str]:
    """
    Validate a protobuf message based on its type.
    
    Args:
        message: Protobuf message to validate
        
    Returns:
        List of validation error messages (empty if valid)
        
    Raises:
        ValidationError: If message type is not supported
    """
    if inference_pb2 is None:
        return ["Protocol buffer definitions not available - run generate_proto.py first"]
    
    message_type = type(message).__name__
    
    if message_type == "InferenceRequest":
        return MessageValidator.validate_inference_request(message)
    elif message_type == "WorkerRegistration":
        return MessageValidator.validate_worker_registration(message)
    elif message_type == "HeartbeatRequest":
        return MessageValidator.validate_heartbeat_request(message)
    elif message_type == "WorkerStatus":
        return MessageValidator.validate_worker_status(message)
    elif message_type == "WorkerMetrics":
        return MessageValidator.validate_worker_metrics(message)
    elif message_type == "DeviceInfo":
        return MessageValidator.validate_device_info(message)
    else:
        raise ValidationError(f"Validation not implemented for message type: {message_type}")


def ensure_valid_message(message: Message) -> None:
    """
    Validate a message and raise ValidationError if invalid.
    
    Args:
        message: Protobuf message to validate
        
    Raises:
        ValidationError: If message validation fails
    """
    errors = validate_message(message)
    if errors:
        raise ValidationError(f"Message validation failed: {'; '.join(errors)}")