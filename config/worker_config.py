"""
Worker Configuration Classes
"""

from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import Field, validator
from .base import BaseConfig, NetworkConfig


class ModelConfig(BaseConfig):
    """Model configuration."""
    
    path: str = Field(env="MODEL_PATH")
    name: str = Field(env="MODEL_NAME")
    cache_size: int = Field(default=2, env="MODEL_CACHE_SIZE")
    validation_enabled: bool = Field(default=True, env="MODEL_VALIDATION_ENABLED")
    
    @validator("path")
    def validate_model_path(cls, v):
        if not v:
            raise ValueError("model_path is required")
        path = Path(v)
        if not path.exists():
            raise ValueError(f"model_path does not exist: {v}")
        return str(path.absolute())
    
    @validator("name")
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("model_name is required")
        return v.strip()
    
    @validator("cache_size")
    def validate_cache_size(cls, v):
        if v <= 0:
            raise ValueError("cache_size must be positive")
        return v


class DeviceConfig(BaseConfig):
    """Device and hardware configuration."""
    
    memory_limit_gb: Optional[int] = Field(default=None, env="DEVICE_MEMORY_LIMIT_GB")
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    auto_detect: bool = Field(default=True, env="DEVICE_AUTO_DETECT")
    
    @validator("memory_limit_gb")
    def validate_memory_limit(cls, v):
        if v is not None and v <= 0:
            raise ValueError("memory_limit_gb must be positive")
        return v
    
    @validator("gpu_memory_fraction")
    def validate_gpu_memory_fraction(cls, v):
        if not (0.0 < v <= 1.0):
            raise ValueError("gpu_memory_fraction must be between 0 and 1")
        return v


class PerformanceConfig(BaseConfig):
    """Performance and concurrency configuration."""
    
    max_concurrent: int = Field(default=4, env="MAX_CONCURRENT")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")
    cleanup_interval: int = Field(default=60, env="CLEANUP_INTERVAL")
    metrics_history_size: int = Field(default=100, env="METRICS_HISTORY_SIZE")
    
    @validator("max_concurrent")
    def validate_max_concurrent(cls, v):
        if v <= 0:
            raise ValueError("max_concurrent must be positive")
        return v
    
    @validator("request_timeout")
    def validate_request_timeout(cls, v):
        if v <= 0:
            raise ValueError("request_timeout must be positive")
        return v
    
    @validator("cleanup_interval")
    def validate_cleanup_interval(cls, v):
        if v <= 0:
            raise ValueError("cleanup_interval must be positive")
        return v
    
    @validator("metrics_history_size")
    def validate_metrics_history_size(cls, v):
        if v <= 0:
            raise ValueError("metrics_history_size must be positive")
        return v


class CoordinatorConnectionConfig(BaseConfig):
    """Coordinator connection configuration."""
    
    host: str = Field(default="coordinator.local:50051", env="COORDINATOR_HOST")
    auto_discover: bool = Field(default=True, env="AUTO_DISCOVER_COORDINATOR")
    heartbeat_interval: int = Field(default=10, env="HEARTBEAT_INTERVAL")
    connection_timeout: int = Field(default=30, env="COORDINATOR_CONNECTION_TIMEOUT")
    max_retries: int = Field(default=3, env="COORDINATOR_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="COORDINATOR_RETRY_DELAY")
    
    @validator("heartbeat_interval")
    def validate_heartbeat_interval(cls, v):
        if v <= 0:
            raise ValueError("heartbeat_interval must be positive")
        return v
    
    @validator("connection_timeout")
    def validate_connection_timeout(cls, v):
        if v <= 0:
            raise ValueError("connection_timeout must be positive")
        return v
    
    @validator("max_retries")
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        return v


class HealthConfig(BaseConfig):
    """Health monitoring configuration."""
    
    check_interval: float = Field(default=30.0, env="HEALTH_CHECK_INTERVAL")
    report_interval: int = Field(default=60, env="HEALTH_REPORT_INTERVAL")
    metrics_enabled: bool = Field(default=True, env="HEALTH_METRICS_ENABLED")
    
    @validator("check_interval")
    def validate_check_interval(cls, v):
        if v <= 0:
            raise ValueError("check_interval must be positive")
        return v
    
    @validator("report_interval")
    def validate_report_interval(cls, v):
        if v <= 0:
            raise ValueError("report_interval must be positive")
        return v


class WorkerConfig(NetworkConfig):
    """Complete worker configuration."""
    
    # Worker identification
    worker_id: Optional[str] = Field(default=None, env="WORKER_ID")
    
    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    coordinator: CoordinatorConnectionConfig = Field(default_factory=CoordinatorConnectionConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    
    @validator("worker_id", pre=True, always=True)
    def generate_worker_id(cls, v):
        if v is None or not v.strip():
            import socket
            hostname = socket.gethostname()
            return f"worker-{hostname}"
        return v.strip()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "WorkerConfig":
        """Load configuration from YAML file."""
        from .base import load_config_from_file
        return load_config_from_file(config_path, cls)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def get_model_path(self) -> Path:
        """Get model path as Path object."""
        return Path(self.model.path)
    
    def get_coordinator_address(self) -> str:
        """Get coordinator address."""
        return self.coordinator.host