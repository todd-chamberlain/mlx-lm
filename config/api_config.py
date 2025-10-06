"""
API Server Configuration Classes
"""

from typing import List, Optional
from pydantic import Field, validator
from .base import BaseConfig, NetworkConfig, SecurityConfig


class ServerConfig(NetworkConfig):
    """API server configuration."""
    
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=False, env="RELOAD")
    worker_processes: int = Field(default=1, env="WORKER_PROCESSES")
    
    @validator("worker_processes")
    def validate_worker_processes(cls, v):
        if v < 1:
            raise ValueError("worker_processes must be at least 1")
        return v


class CoordinatorConfig(BaseConfig):
    """Coordinator connection configuration."""
    
    host: str = Field(default="coordinator", env="COORDINATOR_HOST")
    port: int = Field(default=50051, env="COORDINATOR_PORT")
    timeout: int = Field(default=30, env="COORDINATOR_TIMEOUT")
    max_retries: int = Field(default=3, env="COORDINATOR_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="COORDINATOR_RETRY_DELAY")
    
    @property
    def address(self) -> str:
        """Get full coordinator address."""
        return f"{self.host}:{self.port}"


class RateLimitingConfig(BaseConfig):
    """Rate limiting configuration."""
    
    enabled: bool = Field(default=True, env="RATE_LIMITING_ENABLED")
    requests_per_minute: int = Field(default=100, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    burst_size: int = Field(default=20, env="RATE_LIMIT_BURST_SIZE")
    per_client: bool = Field(default=True, env="RATE_LIMIT_PER_CLIENT")
    
    @validator("requests_per_minute")
    def validate_requests_per_minute(cls, v):
        if v <= 0:
            raise ValueError("requests_per_minute must be positive")
        return v


class CorsConfig(BaseConfig):
    """CORS configuration."""
    
    enabled: bool = Field(default=True, env="CORS_ENABLED")
    allow_origins: List[str] = Field(default=["*"], env="CORS_ALLOW_ORIGINS")
    allow_methods: List[str] = Field(default=["GET", "POST", "OPTIONS"], env="CORS_ALLOW_METHODS")
    allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    allow_credentials: bool = Field(default=False, env="CORS_ALLOW_CREDENTIALS")
    
    @validator("allow_origins", pre=True)
    def parse_allow_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("allow_methods", pre=True)
    def parse_allow_methods(cls, v):
        if isinstance(v, str):
            return [method.strip().upper() for method in v.split(",")]
        return v
    
    @validator("allow_headers", pre=True)
    def parse_allow_headers(cls, v):
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v


class StreamingConfig(BaseConfig):
    """Streaming configuration."""
    
    chunk_size: int = Field(default=1024, env="STREAMING_CHUNK_SIZE")
    timeout: int = Field(default=300, env="STREAMING_TIMEOUT")
    buffer_size: int = Field(default=8192, env="STREAMING_BUFFER_SIZE")
    heartbeat_interval: int = Field(default=30, env="STREAMING_HEARTBEAT_INTERVAL")
    
    @validator("chunk_size")
    def validate_chunk_size(cls, v):
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v
    
    @validator("buffer_size")
    def validate_buffer_size(cls, v):
        if v <= 0:
            raise ValueError("buffer_size must be positive")
        return v


class PerformanceConfig(BaseConfig):
    """Performance configuration."""
    
    max_connections: int = Field(default=1000, env="MAX_CONNECTIONS")
    connection_timeout: int = Field(default=30, env="CONNECTION_TIMEOUT")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")
    
    @validator("max_connections")
    def validate_max_connections(cls, v):
        if v <= 0:
            raise ValueError("max_connections must be positive")
        return v


class APIConfig(BaseConfig):
    """Complete API server configuration."""
    
    # Sub-configurations
    server: ServerConfig = Field(default_factory=ServerConfig)
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "APIConfig":
        """Load configuration from YAML file."""
        from .base import load_config_from_file
        return load_config_from_file(config_path, cls)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()