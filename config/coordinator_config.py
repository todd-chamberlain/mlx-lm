"""
Coordinator Service Configuration Classes
"""

from typing import Dict, Optional
from enum import Enum
from pydantic import Field, validator
from .base import BaseConfig, NetworkConfig, ShardingConfig


class LoadBalancingAlgorithm(str, Enum):
    """Load balancing algorithm options."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_LATENCY = "least_latency"


class ServerConfig(BaseConfig):
    """Coordinator server configuration."""
    
    port: int = Field(default=50051, env="COORDINATOR_PORT")
    max_workers: int = Field(default=10, env="COORDINATOR_MAX_WORKERS")
    enable_reflection: bool = Field(default=False, env="ENABLE_GRPC_REFLECTION")
    max_message_size: int = Field(default=4194304, env="GRPC_MAX_MESSAGE_SIZE")  # 4MB
    
    @validator("port")
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError("port must be between 1 and 65535")
        return v
    
    @validator("max_workers")
    def validate_max_workers(cls, v):
        if v <= 0:
            raise ValueError("max_workers must be positive")
        return v


class WorkerPoolConfig(BaseConfig):
    """Worker pool management configuration."""
    
    heartbeat_timeout: float = Field(default=30.0, env="HEARTBEAT_TIMEOUT")
    cleanup_interval: float = Field(default=60.0, env="CLEANUP_INTERVAL")
    max_workers: int = Field(default=100, env="MAX_WORKERS")
    worker_timeout: int = Field(default=300, env="WORKER_TIMEOUT")
    registration_timeout: int = Field(default=30, env="REGISTRATION_TIMEOUT")
    
    @validator("heartbeat_timeout")
    def validate_heartbeat_timeout(cls, v):
        if v <= 0:
            raise ValueError("heartbeat_timeout must be positive")
        return v
    
    @validator("cleanup_interval")
    def validate_cleanup_interval(cls, v):
        if v <= 0:
            raise ValueError("cleanup_interval must be positive")
        return v
    
    @validator("max_workers")
    def validate_max_workers(cls, v):
        if v <= 0:
            raise ValueError("max_workers must be positive")
        return v


class LoadBalancingConfig(BaseConfig):
    """Load balancing configuration."""
    
    algorithm: LoadBalancingAlgorithm = Field(default=LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN, env="LOAD_BALANCING_ALGORITHM")
    weights: Dict[str, float] = Field(default={
        "latency": 0.4,
        "load": 0.3,
        "reliability": 0.3
    })
    health_check_interval: int = Field(default=10, env="HEALTH_CHECK_INTERVAL")
    
    @validator("weights")
    def validate_weights(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError("weights must sum to 1.0")
        
        for key, weight in v.items():
            if weight < 0 or weight > 1:
                raise ValueError(f"weight for {key} must be between 0 and 1")
        
        return v
    
    @validator("health_check_interval")
    def validate_health_check_interval(cls, v):
        if v <= 0:
            raise ValueError("health_check_interval must be positive")
        return v


class CircuitBreakerConfig(BaseConfig):
    """Circuit breaker configuration."""
    
    failure_threshold: int = Field(default=5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    recovery_timeout: float = Field(default=60.0, env="CIRCUIT_BREAKER_RECOVERY_TIMEOUT")
    half_open_max_calls: int = Field(default=3, env="CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS")
    
    @validator("failure_threshold")
    def validate_failure_threshold(cls, v):
        if v <= 0:
            raise ValueError("failure_threshold must be positive")
        return v
    
    @validator("recovery_timeout")
    def validate_recovery_timeout(cls, v):
        if v <= 0:
            raise ValueError("recovery_timeout must be positive")
        return v
    
    @validator("half_open_max_calls")
    def validate_half_open_max_calls(cls, v):
        if v <= 0:
            raise ValueError("half_open_max_calls must be positive")
        return v


class RoutingConfig(BaseConfig):
    """Request routing configuration."""
    
    max_retries: int = Field(default=3, env="ROUTING_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="ROUTING_RETRY_DELAY")
    retry_backoff: float = Field(default=2.0, env="ROUTING_RETRY_BACKOFF")
    timeout: int = Field(default=300, env="ROUTING_TIMEOUT")
    queue_size: int = Field(default=1000, env="ROUTING_QUEUE_SIZE")
    
    @validator("max_retries")
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        return v
    
    @validator("retry_delay")
    def validate_retry_delay(cls, v):
        if v <= 0:
            raise ValueError("retry_delay must be positive")
        return v
    
    @validator("retry_backoff")
    def validate_retry_backoff(cls, v):
        if v < 1.0:
            raise ValueError("retry_backoff must be at least 1.0")
        return v
    
    @validator("queue_size")
    def validate_queue_size(cls, v):
        if v <= 0:
            raise ValueError("queue_size must be positive")
        return v


class HealthConfig(BaseConfig):
    """Health monitoring configuration."""
    
    check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    unhealthy_threshold: int = Field(default=3, env="HEALTH_UNHEALTHY_THRESHOLD")
    recovery_threshold: int = Field(default=2, env="HEALTH_RECOVERY_THRESHOLD")
    metrics_retention: int = Field(default=300, env="HEALTH_METRICS_RETENTION")
    
    @validator("check_interval")
    def validate_check_interval(cls, v):
        if v <= 0:
            raise ValueError("check_interval must be positive")
        return v
    
    @validator("unhealthy_threshold")
    def validate_unhealthy_threshold(cls, v):
        if v <= 0:
            raise ValueError("unhealthy_threshold must be positive")
        return v
    
    @validator("recovery_threshold")
    def validate_recovery_threshold(cls, v):
        if v <= 0:
            raise ValueError("recovery_threshold must be positive")
        return v


class DiscoveryConfig(BaseConfig):
    """Service discovery configuration."""
    
    enabled: bool = Field(default=True, env="DISCOVERY_ENABLED")
    announce_interval: int = Field(default=30, env="DISCOVERY_ANNOUNCE_INTERVAL")
    ttl: int = Field(default=90, env="DISCOVERY_TTL")
    
    @validator("announce_interval")
    def validate_announce_interval(cls, v):
        if v <= 0:
            raise ValueError("announce_interval must be positive")
        return v
    
    @validator("ttl")
    def validate_ttl(cls, v):
        if v <= 0:
            raise ValueError("ttl must be positive")
        return v


class CoordinatorConfig(BaseConfig):
    """Complete coordinator configuration."""
    
    # Sub-configurations
    server: ServerConfig = Field(default_factory=ServerConfig)
    worker_pool: WorkerPoolConfig = Field(default_factory=WorkerPoolConfig)
    load_balancing: LoadBalancingConfig = Field(default_factory=LoadBalancingConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    sharding: ShardingConfig = Field(default_factory=ShardingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "CoordinatorConfig":
        """Load configuration from YAML file."""
        from .base import load_config_from_file
        return load_config_from_file(config_path, cls)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()