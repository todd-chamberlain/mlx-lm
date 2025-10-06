"""
Base configuration classes for MLX Distributed Inference Server.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings


class BaseConfig(PydanticBaseSettings):
    """Base configuration class with common settings."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    
    # gRPC settings
    grpc_max_message_length: int = Field(default=4 * 1024 * 1024, env="GRPC_MAX_MESSAGE_LENGTH")  # 4MB
    grpc_keepalive_time_ms: int = Field(default=30000, env="GRPC_KEEPALIVE_TIME_MS")
    grpc_keepalive_timeout_ms: int = Field(default=5000, env="GRPC_KEEPALIVE_TIMEOUT_MS")
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of: {', '.join(valid_levels)}")
        return v.upper()
    
    @validator("log_format")
    def validate_log_format(cls, v):
        valid_formats = {"json", "text"}
        if v.lower() not in valid_formats:
            raise ValueError(f"log_format must be one of: {', '.join(valid_formats)}")
        return v.lower()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class NetworkConfig(BaseConfig):
    """Network and connection configuration."""
    
    # Host and port settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Connection limits
    max_connections: int = Field(default=1000, env="MAX_CONNECTIONS")
    connection_timeout: int = Field(default=30, env="CONNECTION_TIMEOUT")
    
    # Retry settings
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="RETRY_DELAY")
    retry_backoff: float = Field(default=2.0, env="RETRY_BACKOFF")
    
    @validator("port")
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError("port must be between 1 and 65535")
        return v


class SecurityConfig(BaseConfig):
    """Security and authentication configuration."""
    
    # Authentication
    enable_auth: bool = Field(default=False, env="ENABLE_AUTH")
    api_keys: List[str] = Field(default_factory=list, env="API_KEYS")
    jwt_secret: Optional[str] = Field(default=None, env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration: int = Field(default=3600, env="JWT_EXPIRATION")  # seconds
    
    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # TLS
    enable_tls: bool = Field(default=False, env="ENABLE_TLS")
    tls_cert_file: Optional[str] = Field(default=None, env="TLS_CERT_FILE")
    tls_key_file: Optional[str] = Field(default=None, env="TLS_KEY_FILE")
    
    @validator("api_keys", pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return v or []
    
    @validator("api_keys")
    def validate_api_keys(cls, v):
        for key in v:
            if len(key) < 32:
                raise ValueError("API keys must be at least 32 characters long")
        return v


def load_config_from_file(config_path: Path, config_class: type = BaseConfig) -> BaseConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        config_class: Configuration class to instantiate
        
    Returns:
        Configuration instance
    """
    import yaml
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Convert nested dict to flat dict with env var style keys
    flat_config = {}
    
    def flatten_dict(d: Dict[str, Any], prefix: str = ""):
        for key, value in d.items():
            env_key = f"{prefix}{key}".upper()
            if isinstance(value, dict):
                flatten_dict(value, f"{prefix}{key}_")
            else:
                flat_config[env_key] = value
    
    flatten_dict(config_data)
    
    # Set environment variables temporarily
    original_env = {}
    for key, value in flat_config.items():
        if key not in os.environ:
            original_env[key] = os.environ.get(key)
            os.environ[key] = str(value)
    
    try:
        config = config_class()
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
    
    return config


def get_config_path(service_name: str, config_name: str = "config.yaml") -> Path:
    """
    Get configuration file path for a service.
    
    Args:
        service_name: Name of the service (api, coordinator, worker)
        config_name: Name of the configuration file
        
    Returns:
        Path to configuration file
    """
    # Try service-specific config first
    service_config = Path(f"config/{service_name}/{config_name}")
    if service_config.exists():
        return service_config
    
    # Fall back to shared config
    shared_config = Path(f"config/{config_name}")
    if shared_config.exists():
        return shared_config
    
    # Return service-specific path for creation
    return service_config